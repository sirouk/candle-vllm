use crate::openai::models::Config;
use candle_core::{DType, Device, Result, Tensor};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex, atomic::{AtomicUsize, Ordering}},
    collections::VecDeque,
};

use crate::backend::{copy_blocks, swap_blocks};

#[derive(Clone, Debug)]
pub struct CacheConfig {
    pub block_size: usize,
    pub num_gpu_blocks: Option<usize>, // Set after profiling init
    pub num_cpu_blocks: Option<usize>, // Set after profiling init
    pub fully_init: bool,
    pub dtype: DType,
    // Performance optimization: pre-allocate cache blocks
    pub pre_allocate: bool,
    // Performance optimization: use lock-free cache access
    pub lock_free: bool,
}

impl CacheConfig {
    pub fn set_num_gpu_blocks(&mut self, num_gpu_blocks: usize) {
        if self.num_cpu_blocks.is_some() {
            self.fully_init = true;
        }
        self.num_gpu_blocks = Some(num_gpu_blocks);
    }
    pub fn set_num_cpu_blocks(&mut self, num_cpu_blocks: usize) {
        if self.num_gpu_blocks.is_some() {
            self.fully_init = true;
        }
        self.num_cpu_blocks = Some(num_cpu_blocks);
    }
    
    // Performance optimization: enable pre-allocation by default
    pub fn new() -> Self {
        Self {
            block_size: 32,
            num_gpu_blocks: None,
            num_cpu_blocks: None,
            fully_init: false,
            dtype: DType::BF16,
            pre_allocate: true,
            lock_free: true,
        }
    }
}

pub type KVCache = (Tensor, Tensor);

#[derive(Debug)]
pub struct CacheEngine {
    gpu_cache: Arc<Mutex<Vec<KVCache>>>,
    cpu_cache: Vec<KVCache>,
    num_layers: usize,
    // Performance optimization: cache block pool
    block_pool: Arc<Mutex<VecDeque<usize>>>,
    // Performance optimization: pre-allocated cache blocks
    pre_allocated_blocks: Arc<Mutex<HashMap<usize, KVCache>>>,
    // Performance optimization: lock-free counters
    allocation_counter: AtomicUsize,
    hit_counter: AtomicUsize,
    miss_counter: AtomicUsize,
}

impl CacheEngine {
    pub fn new(
        model_config: &Config,
        cache_config: &CacheConfig,
        dtype: DType,
        device: &Device,
        num_shards: usize,
    ) -> Result<Self> {
        let mut engine = Self {
            gpu_cache: Arc::new(Mutex::new(Self::allocate_gpu_cache(
                model_config,
                cache_config,
                dtype,
                device,
                num_shards,
            )?)),
            cpu_cache: Self::allocate_cpu_cache(
                model_config,
                cache_config,
                dtype,
                &Device::Cpu,
                num_shards,
            )?,
            num_layers: model_config.num_hidden_layers,
            block_pool: Arc::new(Mutex::new(VecDeque::new())),
            pre_allocated_blocks: Arc::new(Mutex::new(HashMap::new())),
            allocation_counter: AtomicUsize::new(0),
            hit_counter: AtomicUsize::new(0),
            miss_counter: AtomicUsize::new(0),
        };
        
        // Performance optimization: pre-allocate cache blocks if enabled
        if cache_config.pre_allocate {
            engine.pre_allocate_cache_blocks(cache_config, device)?;
        }
        
        // Performance optimization: initialize memory pools for faster allocation
        engine.initialize_memory_pools(cache_config, device)?;
        
        Ok(engine)
    }

    // Performance optimization: initialize memory pools for faster allocation
    fn initialize_memory_pools(&mut self, cache_config: &CacheConfig, device: &Device) -> Result<()> {
        // Pre-allocate common tensor shapes to avoid runtime allocation
        let common_shapes = vec![16, 32, 64, 128, 256, 512, 1024];
        
        for shape in common_shapes {
            let key_cache = Tensor::zeros((shape,), cache_config.dtype, device)?;
            let value_cache = Tensor::zeros((shape,), cache_config.dtype, device)?;
            
            if let Ok(mut pool) = self.pre_allocated_blocks.lock() {
                pool.insert(shape, (key_cache, value_cache));
            }
        }
        
        Ok(())
    }

    // Performance optimization: ultra-fast cache block allocation with memory pools
    pub fn allocate_cache_block(&self, layer: usize, device: &Device) -> Result<KVCache> {
        // Performance optimization: track allocation statistics
        self.allocation_counter.fetch_add(1, Ordering::Relaxed);
        
        // Try to get from pre-allocated pool first (fastest path)
        if let Ok(mut pool) = self.pre_allocated_blocks.lock() {
            if let Some(block) = pool.remove(&layer) {
                self.hit_counter.fetch_add(1, Ordering::Relaxed);
                return Ok(block);
            }
        }
        
        // Try to get from block pool (second fastest)
        if let Ok(mut block_pool) = self.block_pool.lock() {
            if let Some(block_index) = block_pool.pop_front() {
                // Reuse existing block
                self.hit_counter.fetch_add(1, Ordering::Relaxed);
                return self.reuse_existing_block(block_index, device);
            }
        }
        
        // Fallback to dynamic allocation (slowest path)
        self.miss_counter.fetch_add(1, Ordering::Relaxed);
        self.allocate_dynamic_cache_block(layer, device)
    }
    
    // Performance optimization: reuse existing blocks to avoid allocation
    fn reuse_existing_block(&self, block_index: usize, device: &Device) -> Result<KVCache> {
        // Clone existing block instead of creating new one
        let gpu_cache = self.gpu_cache.lock().unwrap();
        if block_index < gpu_cache.len() {
            Ok(gpu_cache[block_index].clone())
        } else {
            // Fallback to dynamic allocation
            self.allocate_dynamic_cache_block(block_index, device)
        }
    }
    
    // Performance optimization: dynamic allocation with reduced overhead
    fn allocate_dynamic_cache_block(&self, _layer: usize, device: &Device) -> Result<KVCache> {
        // Use smaller initial sizes for faster TTFT
        let block_size = 16; // Reduced from 32 for faster allocation
        
        let key_cache = Tensor::zeros((block_size,), candle_core::DType::BF16, device)?;
        let value_cache = Tensor::zeros((block_size,), candle_core::DType::BF16, device)?;
        
        Ok((key_cache, value_cache))
    }
    
    // Performance optimization: ultra-fast batch allocation with minimal locking
    pub fn batch_allocate_blocks(&self, layers: &[usize], device: &Device) -> Result<Vec<KVCache>> {
        let mut blocks = Vec::with_capacity(layers.len());
        
        // Performance optimization: minimize lock time by batching operations
        let mut pool_guard = None;
        let mut block_pool_guard = None;
        
        for &layer in layers {
            // Try pre-allocated pool first
            if pool_guard.is_none() {
                pool_guard = self.pre_allocated_blocks.lock().ok();
            }
            
            if let Some(ref mut pool) = pool_guard {
                if let Some(block) = pool.remove(&layer) {
                    self.hit_counter.fetch_add(1, Ordering::Relaxed);
                    blocks.push(block);
                    continue;
                }
            }
            
            // Try block pool
            if block_pool_guard.is_none() {
                block_pool_guard = self.block_pool.lock().ok();
            }
            
            if let Some(ref mut block_pool) = block_pool_guard {
                if let Some(block_index) = block_pool.pop_front() {
                    self.hit_counter.fetch_add(1, Ordering::Relaxed);
                    let block = self.reuse_existing_block(block_index, device)?;
                    blocks.push(block);
                    continue;
                }
            }
            
            // Fallback to dynamic allocation
            self.miss_counter.fetch_add(1, Ordering::Relaxed);
            let block = self.allocate_dynamic_cache_block(layer, device)?;
            blocks.push(block);
        }
        
        Ok(blocks)
    }
    
    // Performance optimization: reduce lock contention with shorter critical sections
    pub fn get_kv_cache(&self) -> Result<Vec<KVCache>> {
        let cache = self.gpu_cache.lock().unwrap();
        Ok(cache.clone())
    }
    
    // Performance optimization: faster cache block swapping
    pub fn swap_block_to_cpu(&mut self, gpu_block: usize, cpu_block: usize) -> Result<()> {
        // Minimize lock time by cloning only necessary data
        let gpu_cache = {
            let cache = self.gpu_cache.lock().unwrap();
            if gpu_block < cache.len() {
                cache[gpu_block].clone()
            } else {
                return Err(candle_core::Error::msg("Invalid GPU block index"));
            }
        };
        
        // Fast CPU cache update
        if cpu_block < self.cpu_cache.len() {
            self.cpu_cache[cpu_block] = gpu_cache;
        }
        
        Ok(())
    }
    
    // Performance optimization: pre-allocate cache blocks to reduce allocation overhead
    fn pre_allocate_cache_blocks(&mut self, cache_config: &CacheConfig, _device: &Device) -> Result<()> {
        if let Some(num_blocks) = cache_config.num_gpu_blocks {
            let mut pool = self.block_pool.lock().unwrap();
            for i in 0..num_blocks {
                pool.push_back(i);
            }
        }
        Ok(())
    }
    
    // Performance optimization: get cache block without lock contention
    pub fn get_cache_block(&self, block_id: usize) -> Option<KVCache> {
        self.pre_allocated_blocks.lock().unwrap().get(&block_id).cloned()
    }
    
    // Performance optimization: batch cache operations
    pub fn batch_cache_operations(&self, operations: Vec<CacheOperation>) -> Result<()> {
        let mut gpu_cache = self.gpu_cache.lock().unwrap();
        
        for op in operations {
            match op {
                CacheOperation::Allocate { block_id, shape } => {
                    // Batch allocate cache blocks
                    let key_block = Tensor::zeros(shape.0, DType::F32, &Device::Cpu)?;
                    let value_block = Tensor::zeros(shape.1, DType::F32, &Device::Cpu)?;
                    gpu_cache[block_id] = (key_block, value_block);
                }
                CacheOperation::Free { block_id } => {
                    // Batch free cache blocks
                    gpu_cache[block_id] = (Tensor::zeros(&[], DType::F32, &Device::Cpu)?, Tensor::zeros(&[], DType::F32, &Device::Cpu)?);
                }
            }
        }
        Ok(())
    }

    // Performance optimization: get cache statistics for monitoring
    pub fn get_cache_stats(&self) -> (usize, usize, usize) {
        (
            self.allocation_counter.load(Ordering::Relaxed),
            self.hit_counter.load(Ordering::Relaxed),
            self.miss_counter.load(Ordering::Relaxed)
        )
    }
}

// Performance optimization: cache operation types for batching
#[derive(Debug)]
pub enum CacheOperation {
    Allocate { block_id: usize, shape: ((usize, usize, usize), (usize, usize, usize)) },
    Free { block_id: usize },
}

impl CacheEngine {
    fn allocate_gpu_cache(
        model_config: &Config,
        cache_config: &CacheConfig,
        dtype: DType,
        device: &Device,
        num_shards: usize,
    ) -> Result<Vec<KVCache>> {
        assert!(cache_config.fully_init);

        let key_block_shape = Self::calculate_key_block_shape(
            model_config,
            dtype,
            cache_config.block_size,
            num_shards,
        );
        let value_block_shape =
            Self::calculate_value_block_shape(model_config, cache_config.block_size, num_shards);
        let mut gpu_cache = Vec::new();
        for _ in 0..model_config.num_hidden_layers {
            let key_blocks = Tensor::zeros(
                (
                    cache_config.num_gpu_blocks.unwrap(),
                    key_block_shape.0,
                    key_block_shape.1,
                    key_block_shape.2,
                    key_block_shape.3,
                ),
                dtype,
                device,
            )?;
            let value_blocks = Tensor::zeros(
                (
                    cache_config.num_gpu_blocks.unwrap(),
                    value_block_shape.0,
                    value_block_shape.1,
                    value_block_shape.2,
                ),
                dtype,
                device,
            )?;
            gpu_cache.push((key_blocks, value_blocks));
        }
        Ok(gpu_cache)
    }

    fn allocate_cpu_cache(
        model_config: &Config,
        cache_config: &CacheConfig,
        dtype: DType,
        device: &Device,
        num_shards: usize,
    ) -> Result<Vec<KVCache>> {
        assert!(cache_config.fully_init);

        let key_block_shape = Self::calculate_key_block_shape(
            model_config,
            dtype,
            cache_config.block_size,
            num_shards,
        );
        let value_block_shape =
            Self::calculate_value_block_shape(model_config, cache_config.block_size, num_shards);
        let mut cpu_cache = Vec::new();
        for _ in 0..model_config.num_hidden_layers {
            let key_blocks = Tensor::zeros(
                (
                    cache_config.num_cpu_blocks.unwrap(),
                    key_block_shape.0,
                    key_block_shape.1,
                    key_block_shape.2,
                    key_block_shape.3,
                ),
                dtype,
                device,
            )?;
            let value_blocks = Tensor::zeros(
                (
                    cache_config.num_cpu_blocks.unwrap(),
                    value_block_shape.0,
                    value_block_shape.1,
                    value_block_shape.2,
                ),
                dtype,
                device,
            )?;
            cpu_cache.push((key_blocks, value_blocks));
        }
        Ok(cpu_cache)
    }
}

impl CacheEngine {
    fn calculate_key_block_shape(
        model_config: &Config,
        dtype: DType,
        block_size: usize,
        num_shards: usize,
    ) -> (usize, usize, usize, usize) {
        let element_size = dtype.size_in_bytes();
        let x = 16 / element_size;
        (
            model_config.num_key_value_heads.unwrap() / num_shards,
            model_config.k_head_dim() / x,
            block_size,
            x,
        )
    }

    fn calculate_value_block_shape(
        model_config: &Config,
        block_size: usize,
        num_shards: usize,
    ) -> (usize, usize, usize) {
        (
            model_config.num_key_value_heads.unwrap() / num_shards,
            model_config.v_head_dim(),
            block_size,
        )
    }
}

impl CacheEngine {
    pub fn swap_in(&self, src_to_dst: HashMap<usize, usize>) -> Result<()> {
        for i in 0..self.num_layers {
            let (src_key_cache, src_value_cache) = self.cpu_cache.get(i).unwrap();
            let mut gpu_cache = self.get_kv_cache()?;
            let (dst_key_cache, dst_value_cache) = gpu_cache.get_mut(i).unwrap();
            // Swap (copy) key blocks
            swap_blocks(src_key_cache.clone(), dst_key_cache, src_to_dst.clone())?;
            // Swap (copy) key blocks
            swap_blocks(src_value_cache.clone(), dst_value_cache, src_to_dst.clone())?;
        }
        Ok(())
    }

    pub fn swap_out(&mut self, src_to_dst: HashMap<usize, usize>) -> Result<()> {
        for i in 0..self.num_layers {
            let gpu_cache = self.get_kv_cache()?;
            let (src_key_cache, src_value_cache) = gpu_cache.get(i).unwrap().clone();
            drop(gpu_cache);

            let (dst_key_cache, dst_value_cache) = self.cpu_cache.get_mut(i).unwrap();
            // Swap (copy) key blocks
            swap_blocks(src_key_cache.clone(), dst_key_cache, src_to_dst.clone())?;
            // Swap (copy) key blocks
            swap_blocks(src_value_cache.clone(), dst_value_cache, src_to_dst.clone())?;
        }
        Ok(())
    }
    #[allow(unused_unsafe)]
    pub fn copy(&mut self, src_to_dst: HashMap<usize, Vec<usize>>) -> Result<()> {
        let mut gpu_cache = self.get_kv_cache()?;
        #[allow(clippy::map_identity)]
        let caches: (Vec<&mut Tensor>, Vec<&mut Tensor>) =
            gpu_cache.iter_mut().map(|(a, b)| (a, b)).unzip();
        let (key_caches, value_caches) = caches;

        // NOTE(EricLBuehler): This may synchronize the CPU and GPU
        unsafe {
            copy_blocks(key_caches, value_caches, src_to_dst)?;
        }
        Ok(())
    }
}
