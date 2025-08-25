//! The Scheduler uses a BlockEngine to schedule and automatically batch sequences. The
//! primary method `schedule` returns the batched sequences as inputs, as well as the
//! operations to be executed on the cache by the CacheEngine.

pub mod block_engine;
pub mod cache_engine;
pub mod sequence;
use tracing::warn;
type CPUBlockFrom = usize;
type GPUBlockFrom = usize;
type CPUBlockTo = usize;
type GPUBlockTo = usize;
type SrcBlockFrom = usize;
type DstBlocksTo = Vec<usize>;

use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
    sync::atomic::{AtomicUsize, Ordering},
};

use crate::scheduler::{block_engine::AllocStatus, sequence::SequenceStatus};

use self::{block_engine::BlockEngine, cache_engine::CacheConfig, sequence::SequenceGroup};

pub struct SchedulerOutput {
    pub scheduled: Arc<VecDeque<Arc<SequenceGroup>>>,
    pub blocks_to_swap_in: HashMap<CPUBlockFrom, GPUBlockTo>,
    pub blocks_to_swap_out: HashMap<GPUBlockFrom, CPUBlockTo>,
    pub blocks_to_copy: HashMap<SrcBlockFrom, DstBlocksTo>,
    pub ignored_seq_groups: Arc<VecDeque<Arc<SequenceGroup>>>,
}

pub struct SchedulerConfig {
    pub max_num_seqs: usize,
}

pub struct Scheduler {
    waiting: VecDeque<Arc<SequenceGroup>>,
    running: VecDeque<Arc<SequenceGroup>>,
    swapped_out: VecDeque<Arc<SequenceGroup>>,
    config: SchedulerConfig,
    pub block_engine: BlockEngine,
    scheduling_counter: AtomicUsize,
    batch_counter: AtomicUsize,
}

impl Scheduler {
    pub fn new(config: SchedulerConfig, cache_config: &CacheConfig) -> Self {
        assert!(cache_config.fully_init);
        Self {
            waiting: VecDeque::new(),
            running: VecDeque::new(),
            swapped_out: VecDeque::new(),
            config,
            block_engine: BlockEngine::new(
                cache_config.block_size,
                cache_config.num_gpu_blocks.unwrap(),
                cache_config.num_cpu_blocks.unwrap(),
            ),
            scheduling_counter: AtomicUsize::new(0),
            batch_counter: AtomicUsize::new(0),
        }
    }

    pub fn add_sequence(&mut self, seq_group: SequenceGroup) {
        if self.waiting.len() >= self.waiting.capacity() {
            self.waiting.reserve(self.waiting.len() + 10);
        }
        self.waiting.push_back(Arc::new(seq_group));
    }

    pub fn schedule(&mut self) -> SchedulerOutput {
        self.scheduling_counter.fetch_add(1, Ordering::Relaxed);
        
        if self.swapped_out.is_empty() {
            let mut scheduled = VecDeque::with_capacity(self.waiting.len());
            let mut ignored_seq_groups = VecDeque::with_capacity(4);
            
            let mut batch_size = 0;
            let max_batch_size = std::cmp::min(self.waiting.len(), 8);
            
            while !self.waiting.is_empty() && batch_size < max_batch_size {
                let seq_group = self.waiting.front().unwrap().clone();

                let current_running_count: usize = self.running
                    .iter()
                    .map(|group| group.get_seqs().len())
                    .sum();
                
                if self.config.max_num_seqs == current_running_count + 1 {
                    break;
                }

                if seq_group.get_status() != SequenceStatus::Pending {
                    let can_allocate = self.block_engine.can_allocate(&seq_group);
                    match can_allocate {
                        AllocStatus::Later => break,
                        AllocStatus::Impossible => {
                            warn!(
                                "Input prompt with length of {} tokens is too long and exceeds capacity of block engine.",
                                seq_group.get_prompt_len()
                            );
                            seq_group.set_status(SequenceStatus::FinishedIgnored);
                            ignored_seq_groups.push_back(self.waiting.pop_front().unwrap());
                            continue;
                        }
                        _ => {}
                    }

                    self._allocate(&seq_group);
                }

                seq_group.set_status(SequenceStatus::Running);
                let seq_group = self.waiting.pop_front().unwrap();
                self.running.push_back(seq_group.clone());
                scheduled.push_back(seq_group);
                batch_size += 1;
            }
            
            if batch_size > 1 {
                self.batch_counter.fetch_add(1, Ordering::Relaxed);
            }

            if !scheduled.is_empty() || !ignored_seq_groups.is_empty() {
                return SchedulerOutput {
                    scheduled: Arc::new(scheduled),
                    blocks_to_swap_in: HashMap::new(),
                    blocks_to_copy: HashMap::new(),
                    blocks_to_swap_out: HashMap::new(),
                    ignored_seq_groups: Arc::new(ignored_seq_groups),
                };
            }
        }

        let mut blocks_to_swap_out = HashMap::new();
        let mut blocks_to_swap_in = HashMap::new();
        let mut blocks_to_copy = HashMap::new();

        self.sort_running_by_priority_fcfs();

        let mut running = VecDeque::new();
        let mut preempted = VecDeque::new();
        while !self.running.is_empty() {
            let seq_group = self.running.pop_front().unwrap();
            let mut finished_with_break = false;
            while !self.block_engine.can_append_token_to_seq(&seq_group) {
                if !self.running.is_empty() {
                    let seq_to_preempt = self.running.pop_back().unwrap();
                    self._preempt(seq_to_preempt.clone(), &mut blocks_to_swap_out);
                    preempted.push_back(seq_to_preempt);
                } else {
                    self._preempt(seq_group.clone(), &mut blocks_to_swap_out);
                    preempted.push_back(seq_group.clone());
                    finished_with_break = true;
                    break;
                }
            }
            if !finished_with_break {
                self._append_token_slot_to_seq_group(&seq_group, &mut blocks_to_copy);
                running.push_back(seq_group);
            }
        }
        self.running = running;

        self.sort_swapped_out_by_priority_fcfs();

        if preempted.is_empty() {
            while !self.swapped_out.is_empty() {
                let seq_group = self.swapped_out.front().unwrap();

                if !self.block_engine.can_swap_in_seq_group(seq_group) {
                    break;
                }

                let seq_group = self.swapped_out.pop_front().unwrap();
                let to_swap_in = self.block_engine.swap_in(&seq_group);
                blocks_to_swap_in.extend(to_swap_in);
                self._append_token_slot_to_seq_group(&seq_group, &mut blocks_to_copy);
                self.running.push_back(seq_group);
            }
        }

        SchedulerOutput {
            scheduled: self.running.clone().into(),
            blocks_to_swap_in,
            blocks_to_copy,
            blocks_to_swap_out,
            ignored_seq_groups: Arc::new(VecDeque::new()),
        }
    }

    pub fn has_unfinished_sequences(&self) -> bool {
        !self.running.is_empty() || !self.waiting.is_empty()
    }

    pub fn free_finished_sequence_groups(&mut self) {
        let mut to_free = Vec::new();
        let clone = self.running.clone();
        self.running = clone
            .iter()
            .filter(|group| {
                if group.is_finished() {
                    to_free.push((*group).clone());
                    false
                } else {
                    true
                }
            })
            .cloned()
            .collect::<VecDeque<_>>();
        for group in to_free {
            self._free(&group);
        }
    }

    pub fn print_free_blocks(&self) {
        let free_blocks = self.block_engine.get_num_free_blocks();
        tracing::info!(
            "Available kvcache blocks {} (for {} tokens)",
            free_blocks,
            free_blocks * self.block_engine.get_block_size()
        );
    }

    pub fn get_available_kv_tokens(&self) -> usize {
        let free_blocks = self.block_engine.get_num_free_blocks();
        free_blocks * self.block_engine.get_block_size()
    }

    pub fn filter_prefill_finished(
        &mut self,
        scheduled: &VecDeque<Arc<SequenceGroup>>,
        chunk_size: usize,
    ) -> (Vec<u32>, VecDeque<Arc<SequenceGroup>>) {
        let mut finished_indices = Vec::new();
        let mut remove_ids = Vec::new();
        assert!(chunk_size > 0, "Invalid prefill chunk size!");
        for (i, group) in scheduled.iter().enumerate() {
            let seq = group.get_seqs().values().nth(0).unwrap();
            let prompt_len = seq.deref().get_prompt_len();
            let num_cached_tokens = seq.deref().get_num_cached_tokens();
            if prompt_len < chunk_size || num_cached_tokens + chunk_size >= prompt_len {
                if prompt_len > chunk_size {
                    tracing::info!(
                        "Seq {} chunk prefill finished ({} tokens)",
                        seq.deref().get_id(),
                        prompt_len
                    );
                }
                finished_indices.push(i as u32);
            } else {
                remove_ids.push(seq.deref().get_id());
                //unfinished due to chunked_prefill, push back to waiting list
                let group = group.clone();
                let seq = group.get_seqs().values().nth(0).unwrap();
                seq.deref_mut()
                    .set_num_cached_tokens(num_cached_tokens + chunk_size); //current prefilled CHUNK_SIZE
                group.set_status(SequenceStatus::Pending);
                tracing::info!(
                    "Seq {} chunk prefilled {}/{} tokens",
                    seq.deref().get_id(),
                    seq.deref().get_num_cached_tokens(),
                    prompt_len
                );
                self.waiting.push_back(group);
            }
        }
        self.running.retain(|s| {
            !remove_ids.contains(&s.get_seqs().values().nth(0).unwrap().deref().get_id())
        });

        let finished_groups: VecDeque<Arc<SequenceGroup>> = finished_indices
            .iter()
            .map(|&i| Arc::clone(&scheduled[i as usize]))
            .collect();
        (finished_indices, finished_groups)
    }

    pub fn get_performance_stats(&self) -> (usize, usize) {
        (
            self.scheduling_counter.load(Ordering::Relaxed),
            self.batch_counter.load(Ordering::Relaxed)
        )
    }
}

impl Scheduler {
    fn remove_seq_group(&mut self, seq_group: &SequenceGroup) {
        if let Some(idx) = self
            .waiting
            .iter()
            .position(|grp| grp.get_id() == seq_group.get_id())
        {
            self.waiting.remove(idx);
        };
        if let Some(idx) = self
            .running
            .iter()
            .position(|grp| grp.get_id() == seq_group.get_id())
        {
            self.running.remove(idx);
        };
        if let Some(idx) = self
            .swapped_out
            .iter()
            .position(|grp| grp.get_id() == seq_group.get_id())
        {
            self.swapped_out.remove(idx);
        };
    }
    fn _append_token_slot_to_seq_group(
        &mut self,
        seq_group: &SequenceGroup,
        blocks_to_copy: &mut HashMap<usize, Vec<usize>>,
    ) {
        for seq in seq_group.get_seqs().values() {
            let op = self.block_engine.append_token_slot_to_seq(seq);
            if let Some((src_block, dst_block)) = op {
                if let std::collections::hash_map::Entry::Vacant(e) =
                    blocks_to_copy.entry(src_block)
                {
                    e.insert(vec![dst_block]);
                } else {
                    blocks_to_copy.get_mut(&src_block).unwrap().push(dst_block);
                }
            }
        }
    }

    fn _abort_seq_group(&mut self, seq_group: &SequenceGroup) {
        self.remove_seq_group(seq_group);
        seq_group.set_status(SequenceStatus::FinishedAborted);
        self._free(seq_group);
    }

    fn _preempt(
        &mut self,
        seq_group: Arc<SequenceGroup>,
        blocks_to_swap_out: &mut HashMap<usize, usize>,
    ) {
        match seq_group.get_seqs().len() {
            1 => self._preempt_by_recompute(seq_group),
            _ => self._preempt_by_swap(seq_group, blocks_to_swap_out),
        }
    }

    fn _preempt_by_recompute(&mut self, seq_group: Arc<SequenceGroup>) {
        seq_group.set_status(SequenceStatus::Waiting);
        self._free(&seq_group);
        self.waiting.push_front(seq_group);
    }

    fn _preempt_by_swap(
        &mut self,
        seq_group: Arc<SequenceGroup>,
        blocks_to_swap_out: &mut HashMap<usize, usize>,
    ) {
        if !self.block_engine.can_swap_out_seq_group(&seq_group) {
            self._abort_seq_group(&seq_group);
            return;
        }
        let new_to_swap = self.block_engine.swap_out(&seq_group);
        blocks_to_swap_out.extend(new_to_swap);
        seq_group.set_status(SequenceStatus::Swapped);

        self.swapped_out.push_back(seq_group);
    }

    fn _allocate(&mut self, seq_group: &SequenceGroup) {
        self.block_engine.allocate(seq_group)
    }

    fn _free(&mut self, seq_group: &SequenceGroup) {
        for seq in seq_group.get_seqs().values() {
            self.block_engine.free_sequence(seq);
        }
    }

    fn sort_running_by_priority_fcfs(&mut self) {
        self.running
            .make_contiguous()
            .sort_by_key(|seq_group| seq_group.arrival_time());
        self.running.make_contiguous().reverse();
    }

    fn sort_swapped_out_by_priority_fcfs(&mut self) {
        self.swapped_out
            .make_contiguous()
            .sort_by_key(|seq_group| seq_group.arrival_time());
        self.swapped_out.make_contiguous().reverse();
    }
}
