#[cfg(feature = "cuda")]
use crate::backend::custom_ops::sort::ArgSortOp; //Use our custom sort kernel, fix kernel crash on A100
use crate::backend::custom_ops::moe::{TopKLastDimOp, TopKOutput};
use crate::candle::D;
use crate::candle::{DType, Error, Result, Tensor};
use crate::openai::sampling_params::{SamplingParams, TopLogprob};
use rand::{distr::Distribution, SeedableRng};
use std::sync::Arc;
use std::sync::Mutex;

#[derive(Clone, Debug)]
pub struct SamplingResult {
    pub token: u32,
    pub logprob: f32,
    pub top_logprobs: Vec<TopLogprob>,
}

#[derive(Clone, PartialEq, Debug)]
pub enum Sampling {
    ArgMax,
    All { temperature: f32 },
    TopK { k: usize, temperature: f32 },
    TopP { p: f32, temperature: f32 },
    TopKThenTopP { k: usize, p: f32, temperature: f32 },
}

pub struct LogitsProcessor {
    rng: Arc<Mutex<rand::rngs::StdRng>>,
    pub sampling: Sampling,
}

impl LogitsProcessor {
    // Performance optimization: cache computed logprobs to avoid recomputation
    pub fn sample_with_logprobs_cached(
        &self,
        logits: &Tensor,
        sampling_params: &Option<SamplingParams>,
        tokenizer: Option<&tokenizers::Tokenizer>,
        seeds: Vec<u64>,
        cache: &mut Option<Tensor>, // Cache for computed logprobs
    ) -> Result<Vec<SamplingResult>> {
        let logits = logits.to_dtype(DType::F32)?;
        let batch = logits.layout().dims()[0];
        
        // Performance optimization: use cached logprobs if available
        let log_probs = if let Some(cached) = cache {
            if cached.dims() == logits.dims() {
                cached.clone()
            } else {
                let computed = self.compute_log_softmax(&logits)?;
                *cache = Some(computed.clone());
                computed
            }
        } else {
            let computed = self.compute_log_softmax(&logits)?;
            *cache = Some(computed.clone());
            computed
        };
        
        // Get top logprobs if requested (from raw logits)
        // Only return top_logprobs if explicitly requested via top_logprobs parameter
        let num_top_logprobs = sampling_params
            .as_ref()
            .and_then(|p| p.top_logprobs)
            .unwrap_or(0);
        
        let top_logprobs = if num_top_logprobs > 0 {
            self.extract_top_logprobs(&log_probs, num_top_logprobs, tokenizer)?
        } else {
            vec![vec![]; batch]
        };

        let next_tokens = self.sample_with_seeds(&logits, sampling_params, seeds)?;
        
        let log_probs_vec: Vec<Vec<f32>> = log_probs.to_vec2()?;
        
        let results: Vec<SamplingResult> = next_tokens
            .into_iter()
            .enumerate()
            .map(|(b, token)| {
                let logprob = log_probs_vec[b][token as usize];
                SamplingResult {
                    token,
                    logprob,
                    top_logprobs: top_logprobs[b].clone(),
                }
            })
            .collect();
        
        Ok(results)
    }
    
    pub fn compute_log_softmax_fast(&self, logits: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        {
            if let Ok(_device) = logits.device().as_cuda_device() {
                return candle_nn::ops::softmax_last_dim(logits)?.log();
            }
        }
        
        candle_nn::ops::softmax_last_dim(logits)?.log()
    }
    
    pub fn compute_log_softmax_with_temperature(&self, logits: &Tensor, temperature: f32) -> Result<Tensor> {
        if temperature == 1.0 {
            self.compute_log_softmax_fast(logits)
        } else if temperature < 0.01 {
            let safe_temperature = 0.1;
            let scaled_logits = (logits / safe_temperature as f64)?;
            self.compute_log_softmax_fast(&scaled_logits)
        } else {
            let scaled_logits = (logits / temperature as f64)?;
            self.compute_log_softmax_fast(&scaled_logits)
        }
    }
    
    pub fn compute_log_softmax(&self, logits: &Tensor) -> Result<Tensor> {
        self.compute_log_softmax_fast(logits)
    }
    
    pub fn sample_batch_optimized(
        &self,
        logits: &Tensor,
        sampling_params: &Option<SamplingParams>,
        seeds: Vec<u64>,
    ) -> Result<Vec<u32>> {
        let logits = logits.to_dtype(DType::F32)?;
        let batch = logits.layout().dims()[0];
        
        let sampling = sampling_params.as_ref().map_or_else(
            || self.sampling.to_owned(),
            |param| LogitsProcessor::get_strategy(param.temperature, param.top_k, param.top_p),
        );
        
        match &sampling {
            Sampling::ArgMax => Ok(self.sample_argmax(&logits)?),
            Sampling::All { temperature } => {
                // Vectorized temperature scaling
                let scaled_logits = (&logits / *temperature as f64)?;
                let prs = candle_nn::ops::softmax_last_dim(&scaled_logits)?;
                let prs_vec: Vec<Vec<f32>> = prs.to_vec2()?;
                
                // Parallel sampling with pre-computed probabilities
                let result: Vec<u32> = (0..batch)
                    .into_iter()
                    .zip(seeds.iter())
                    .map(|(b, seed)| Self::sample_multinomial_with_seed(&prs_vec[b], *seed).unwrap())
                    .collect();
                Ok(result)
            }
            Sampling::TopP { p, temperature } => {
                let scaled_logits = (&logits / *temperature as f64)?;
                let prs = candle_nn::ops::softmax_last_dim(&scaled_logits)?;
                
                if *p <= 0.0 || *p >= 1.0 {
                    let prs_vec: Vec<Vec<f32>> = prs.to_vec2()?;
                    let result: Vec<u32> = (0..batch)
                        .into_iter()
                        .zip(seeds.iter())
                        .map(|(b, seed)| Self::sample_multinomial_with_seed(&prs_vec[b], *seed).unwrap())
                        .collect();
                    Ok(result)
                } else {
                    Ok(self.sample_topp_with_seeds(&prs, *p as f32, seeds)?)
                }
            }
            Sampling::TopK { k, temperature } => {
                let scaled_logits = (&logits / *temperature as f64)?;
                let prs = candle_nn::ops::softmax_last_dim(&scaled_logits)?;
                Ok(self.sample_topk_with_seeds(&prs, *k, seeds)?)
            }
            Sampling::TopKThenTopP { k, p, temperature } => {
                let scaled_logits = (&logits / *temperature as f64)?;
                let prs = candle_nn::ops::softmax_last_dim(&scaled_logits)?;
                Ok(self.sample_topk_topp_with_seeds(&prs, *k, *p as f32, seeds)?)
            }
        }
    }

    pub fn from_sampling(seed: u64, sampling: Sampling) -> Self {
        let rng = rand::rngs::StdRng::seed_from_u64(seed);
        Self {
            rng: Arc::new(Mutex::new(rng)),
            sampling,
        }
    }

    pub fn new(
        seed: u64,
        temperature: Option<f32>,
        top_k: Option<isize>,
        top_p: Option<f32>,
    ) -> Self {
        let strategy = LogitsProcessor::get_strategy(temperature, top_k, top_p);
        Self::from_sampling(seed, strategy)
    }
    
    pub fn set_seed(&self, seed: u64) {
        let new_rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut rng = self.rng.lock().unwrap();
        *rng = new_rng;
    }

    pub fn get_strategy(
        temperature: Option<f32>,
        top_k: Option<isize>,
        top_p: Option<f32>,
    ) -> Sampling {
        let temperature = temperature.unwrap_or(1.0);
        let temperature = if temperature < 1e-7 { None } else { Some(temperature) };
        let top_k: Option<usize> = top_k.filter(|&k| k > 0).map(|k| k as usize);

        let temperature: Option<f32> = temperature.filter(|&t| t > 0.0);

        match (temperature, top_k, top_p) {
            (None, _, _) => Sampling::ArgMax,
            (Some(temperature), None, None) => Sampling::All { temperature },
            (Some(temperature), Some(k), None) => Sampling::TopK { k, temperature },
            (Some(temperature), None, Some(p)) => Sampling::TopP { p, temperature },
            (Some(temperature), Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
        }
    }

    fn sample_argmax(&self, logits: &Tensor) -> Result<Vec<u32>> {
        let next_tokens = logits.argmax(D::Minus1)?.to_vec1::<u32>()?;
        Ok(next_tokens)
    }

    fn sample_multinomial(&self, prs: &Vec<f32>) -> Result<u32> {
        let distr = rand::distr::weighted::WeightedIndex::new(prs).map_err(Error::wrap)?;
        let mut rng = self.rng.lock().unwrap();
        let next_token = distr.sample(&mut *rng) as u32;
        Ok(next_token)
    }
    
    fn sample_multinomial_with_seed(prs: &Vec<f32>, seed: u64) -> Result<u32> {
        let distr = rand::distr::weighted::WeightedIndex::new(prs).map_err(Error::wrap)?;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let next_token = distr.sample(&mut rng) as u32;
        Ok(next_token)
    }

    fn sample_topp_with_seeds(&self, logits: &Tensor, top_p: f32, seeds: Vec<u64>) -> Result<Vec<u32>> {
        #[cfg(feature = "cuda")]
        let asort = logits.arg_sort(false)?;
        #[cfg(not(feature = "cuda"))]
        let asort = logits
            .to_device(&candle_core::Device::Cpu)?
            .arg_sort_last_dim(false)?;
        let asort: Vec<Vec<u32>> = asort.to_vec2()?;
        let sorted: Vec<Vec<f32>> = logits.to_vec2()?;
        let batch = logits.layout().dims()[0];
        let vec_ret: Vec<u32> = (0..batch)
            .into_iter()
            .zip(seeds.iter())
            .map(|(b, seed)| {
                let indices: Vec<u32> = asort[b].to_vec();
                let mut prs: Vec<f32> = sorted[b].to_vec();
                let mut cumsum = 0.;
                for index in &indices {
                    if cumsum >= top_p {
                        prs[*index as usize] = 0.0;
                    } else {
                        cumsum += prs[*index as usize];
                    }
                }
                Self::sample_multinomial_with_seed(&prs, *seed).unwrap()
            })
            .collect();
        Ok(vec_ret)
    }

    fn sample_topp(&self, logits: &Tensor, top_p: f32) -> Result<Vec<u32>> {
        #[cfg(feature = "cuda")]
        let asort = logits.arg_sort(false)?;
        #[cfg(not(feature = "cuda"))]
        let asort = logits
            .to_device(&candle_core::Device::Cpu)?
            .arg_sort_last_dim(false)?;
        let asort: Vec<Vec<u32>> = asort.to_vec2()?;
        let sorted: Vec<Vec<f32>> = logits.to_vec2()?;
        let batch = logits.layout().dims()[0];
        let vec_ret: Vec<u32> = (0..batch)
            .into_iter()
            .map(|b| {
                let indices: Vec<u32> = asort[b].to_vec();
                let mut prs: Vec<f32> = sorted[b].to_vec();
                let mut cumsum = 0.;
                for index in &indices {
                    if cumsum >= top_p {
                        prs[*index as usize] = 0.0;
                    } else {
                        cumsum += prs[*index as usize];
                    }
                }
                // Sample with clamped probabilities.
                self.sample_multinomial(&prs).unwrap()
            })
            .collect();
        Ok(vec_ret)
    }

    fn sample_topk_with_seeds(&self, logits: &Tensor, top_k: usize, seeds: Vec<u64>) -> Result<Vec<u32>> {
        #[cfg(feature = "cuda")]
        let (sorted, asort) = logits.sort(false)?;
        #[cfg(not(feature = "cuda"))]
        let (sorted, asort) = logits
            .to_device(&candle_core::Device::Cpu)?
            .sort_last_dim(false)?;
        let asort: Vec<Vec<u32>> = asort.to_vec2()?;
        let sorted: Vec<Vec<f32>> = sorted.to_vec2()?;
        let batch = logits.layout().dims()[0];
        let vec_ret: Vec<u32> = (0..batch)
            .into_iter()
            .zip(seeds.iter())
            .map(|(b, seed)| {
                let indices: Vec<u32> = asort[b][0..top_k].to_vec();
                let prs: Vec<f32> = sorted[b][0..top_k].to_vec();
                let index = Self::sample_multinomial_with_seed(&prs, *seed).unwrap();
                indices[index as usize] as u32
            })
            .collect();
        Ok(vec_ret)
    }

    // top-k sampling samples from the k tokens with the largest probabilities.
    fn sample_topk(&self, logits: &Tensor, top_k: usize) -> Result<Vec<u32>> {
        #[cfg(feature = "cuda")]
        let (sorted, asort) = logits.sort(false)?;
        #[cfg(not(feature = "cuda"))]
        let (sorted, asort) = logits
            .to_device(&candle_core::Device::Cpu)?
            .sort_last_dim(false)?;
        let asort: Vec<Vec<u32>> = asort.to_vec2()?;
        let sorted: Vec<Vec<f32>> = sorted.to_vec2()?;
        let batch = logits.layout().dims()[0];
        let vec_ret: Vec<u32> = (0..batch)
            .into_iter()
            .map(|b| {
                let indices: Vec<u32> = asort[b][0..top_k].to_vec();
                let prs: Vec<f32> = sorted[b][0..top_k].to_vec();
                let index = self.sample_multinomial(&prs).unwrap();
                indices[index as usize] as u32
            })
            .collect();
        Ok(vec_ret)
    }

    fn sample_topk_topp_with_seeds(&self, logits: &Tensor, top_k: usize, top_p: f32, seeds: Vec<u64>) -> Result<Vec<u32>> {
        #[cfg(feature = "cuda")]
        let (sorted, asort) = logits.sort(false)?;
        #[cfg(not(feature = "cuda"))]
        let (sorted, asort) = logits
            .to_device(&candle_core::Device::Cpu)?
            .sort_last_dim(false)?;
        let asort: Vec<Vec<u32>> = asort.to_vec2()?;
        let sorted: Vec<Vec<f32>> = sorted.to_vec2()?;
        let batch = logits.layout().dims()[0];
        let vec_ret: Vec<u32> = (0..batch)
            .into_iter()
            .zip(seeds.iter())
            .map(|(b, seed)| {
                let indices: Vec<u32> = asort[b][0..top_k].to_vec();
                let mut prs: Vec<f32> = sorted[b][0..top_k].to_vec();
                let sum_p = prs.iter().sum::<f32>();
                let index = if top_p <= 0.0 || top_p >= sum_p {
                    Self::sample_multinomial_with_seed(&prs, *seed).unwrap()
                } else {
                    let mut cumsum = 0.;
                    for i in 0..prs.len() {
                        if cumsum >= top_p {
                            prs[i] = 0.0;
                        } else {
                            cumsum += prs[i];
                        }
                    }
                    // Sample with clamped probabilities using per-request seed
                    Self::sample_multinomial_with_seed(&prs, *seed).unwrap()
                };
                indices[index as usize] as u32
            })
            .collect();
        Ok(vec_ret)
    }

    fn sample_topk_topp(&self, logits: &Tensor, top_k: usize, top_p: f32) -> Result<Vec<u32>> {
        #[cfg(feature = "cuda")]
        let (sorted, asort) = logits.sort(false)?;
        #[cfg(not(feature = "cuda"))]
        let (sorted, asort) = logits
            .to_device(&candle_core::Device::Cpu)?
            .sort_last_dim(false)?;
        let asort: Vec<Vec<u32>> = asort.to_vec2()?;
        let sorted: Vec<Vec<f32>> = sorted.to_vec2()?;
        let batch = logits.layout().dims()[0];
        let vec_ret: Vec<u32> = (0..batch)
            .into_iter()
            .map(|b| {
                let indices: Vec<u32> = asort[b][0..top_k].to_vec();
                let mut prs: Vec<f32> = sorted[b][0..top_k].to_vec();
                let sum_p = prs.iter().sum::<f32>();
                let index = if top_p <= 0.0 || top_p >= sum_p {
                    self.sample_multinomial(&prs).unwrap()
                } else {
                    let mut cumsum = 0.;
                    for i in 0..prs.len() {
                        if cumsum >= top_p {
                            prs[i] = 0.0;
                        } else {
                            cumsum += prs[i];
                        }
                    }
                    self.sample_multinomial(&prs).unwrap()
                };
                indices[index as usize] as u32
            })
            .collect();
        Ok(vec_ret)
    }

    pub fn extract_top_logprobs(
        &self,
        log_probs: &Tensor,
        top_n: usize,
        tokenizer: Option<&tokenizers::Tokenizer>,
    ) -> Result<Vec<Vec<TopLogprob>>> {
        let batch = log_probs.dims()[0];
        let vocab_size = log_probs.dims()[1];
        
        let (top_values, top_indices) = if top_n >= vocab_size {
            log_probs.sort_last_dim(false)?
        } else {
            let TopKOutput { values, indices } = log_probs.topk(top_n)?;
            (values, indices)
        };
        
        let top_values: Vec<Vec<f32>> = top_values.to_vec2()?;
        let top_indices: Vec<Vec<u32>> = top_indices.to_vec2()?;
        
        let results: Vec<Vec<TopLogprob>> = (0..batch)
            .into_iter()
            .map(|b| {
                let mut top_logprobs = Vec::new();
                for i in 0..top_n.min(top_indices[b].len()) {
                    let token = top_indices[b][i] as usize;
                    let logprob = top_values[b][i];
                    
                    if token < vocab_size && logprob > -1000.0 && logprob < 1000.0 {
                        let bytes = if let Some(tok) = tokenizer {
                            match tok.id_to_token(token as u32) {
                                Some(token_text) => token_text,
                                None => format!("<{}>", token)
                            }
                        } else {
                            format!("<{}>", token)
                        };
                        
                        if !bytes.starts_with("<") || bytes.len() < 10 {
                            let bytes_vec: Vec<i32> = bytes.bytes().map(|b| b as i32).collect();
                            top_logprobs.push(TopLogprob {
                                token_id: token,
                                logprob,
                                bytes: bytes_vec,
                            });
                        }
                    }
                }
                top_logprobs
            })
            .collect();
        
        Ok(results)
    }

    pub fn sample_with_logprobs(
        &self,
        logits: &Tensor,
        sampling_params: &Option<SamplingParams>,
        tokenizer: Option<&tokenizers::Tokenizer>,
        seeds: Vec<u64>, 
    ) -> Result<Vec<SamplingResult>> {
        let logits = logits.to_dtype(DType::F32)?;
        let batch = logits.layout().dims()[0];
        
        let log_probs = self.compute_log_softmax(&logits)?;
        
        let num_top_logprobs = sampling_params
            .as_ref()
            .and_then(|p| p.top_logprobs)
            .unwrap_or(0);
        
        let top_logprobs = if num_top_logprobs > 0 {
            self.extract_top_logprobs(&log_probs, num_top_logprobs, tokenizer)?
        } else {
            vec![vec![]; batch]
        };

        let next_tokens = self.sample_with_seeds(&logits, sampling_params, seeds)?;
        
        let log_probs_vec: Vec<Vec<f32>> = log_probs.to_vec2()?;
        
        let results: Vec<SamplingResult> = next_tokens
            .into_iter()
            .enumerate()
            .map(|(b, token)| {
                let logprob = log_probs_vec[b][token as usize];
                SamplingResult {
                    token,
                    logprob,
                    top_logprobs: top_logprobs[b].clone(),
                }
            })
            .collect();
        
        Ok(results)
    }

    pub fn sample_with_logprobs_and_penalties(
        &self,
        logits: &Tensor,
        sampling_params: &Option<SamplingParams>,
        penalties: Vec<f32>,
        reference_tokens: Vec<Vec<u32>>,
        tokenizer: Option<&tokenizers::Tokenizer>,
        seeds: Vec<u64>,  // Per-request seeds for vLLM V1 compatibility
    ) -> Result<Vec<SamplingResult>> {
        let logits = logits.to_dtype(DType::F32)?;
        let batch = logits.layout().dims()[0];
        
        // Apply penalties first to get penalized logits
        let penalized_logits = if penalties.iter().any(|&v| v != 1.0 && v != 0.) {
            self.apply_batch_repeat_penalty(&logits, penalties, reference_tokens)?
        } else {
            logits.clone()
        };

        // vLLM V1: Compute log probabilities from raw penalized logits (not temperature-scaled)
        // This ensures logprobs match vLLM's behavior exactly
        let log_probs = self.compute_log_softmax(&penalized_logits)?;
        
        // Get top logprobs if requested (from penalized logits)
        // Only return top_logprobs if explicitly requested via top_logprobs parameter
        let num_top_logprobs = sampling_params
            .as_ref()
            .and_then(|p| p.top_logprobs)
            .unwrap_or(0);
        
        let top_logprobs = if num_top_logprobs > 0 {
            self.extract_top_logprobs(&log_probs, num_top_logprobs, tokenizer)?
        } else {
            vec![vec![]; batch]
        };

        let next_tokens = self.sample_with_seeds(&penalized_logits, sampling_params, seeds)?;
        
        let log_probs_vec: Vec<Vec<f32>> = log_probs.to_vec2()?;
        
        let results: Vec<SamplingResult> = next_tokens
            .into_iter()
            .enumerate()
            .map(|(b, token)| {
                let logprob = log_probs_vec[b][token as usize];
                SamplingResult {
                    token,
                    logprob,
                    top_logprobs: top_logprobs[b].clone(),
                }
            })
            .collect();
        
        Ok(results)
    }

    pub fn sample_with_seeds(
        &self,
        logits: &Tensor,
        sampling_params: &Option<SamplingParams>,
        seeds: Vec<u64>,
    ) -> Result<Vec<u32>> {
        let logits = logits.to_dtype(DType::F32)?;
        let batch = logits.layout().dims()[0];
        
        // Ensure we have the right number of seeds
        if seeds.len() != batch {
            return Err(Error::msg(format!(
                "Expected {} seeds for batch size {}, got {}",
                batch, batch, seeds.len()
            )));
        }
        
        let prs = |temperature: f64| -> Result<Tensor> {
            let logits = (&logits / temperature)?;
            let prs = candle_nn::ops::softmax_last_dim(&logits)?;
            Ok(prs)
        };

        let sampling = sampling_params.as_ref().map_or_else(
            || self.sampling.to_owned(),
            |param| LogitsProcessor::get_strategy(param.temperature, param.top_k, param.top_p),
        );

        let next_tokens = match &sampling {
            Sampling::ArgMax => self.sample_argmax(&logits)?,
            Sampling::All { temperature } => {
                let prs = prs(*temperature as f64)?.to_vec2()?;
                (0..batch)
                    .zip(seeds.into_iter())
                    .map(|(b, seed)| Self::sample_multinomial_with_seed(&prs[b], seed).unwrap())
                    .collect()
            }
            Sampling::TopP { p, temperature } => {
                let prs = prs(*temperature as f64)?;
                if *p <= 0.0 || *p >= 1.0 {
                    // simply sample from the predicted probability distribution
                    let prs = prs.to_vec2()?;
                    (0..batch)
                        .zip(seeds.into_iter())
                        .map(|(b, seed)| Self::sample_multinomial_with_seed(&prs[b], seed).unwrap())
                        .collect()
                } else {
                    // top-p (nucleus) sampling with per-request seeds
                    self.sample_topp_with_seeds(&prs, *p as f32, seeds)?
                }
            }
            Sampling::TopK { k, temperature } => {
                let prs = prs(*temperature as f64)?;
                self.sample_topk_with_seeds(&prs, *k, seeds)?
            }
            Sampling::TopKThenTopP { k, p, temperature } => {
                let prs = prs(*temperature as f64)?;
                self.sample_topk_topp_with_seeds(&prs, *k, *p as f32, seeds)?
            }
        };
        Ok(next_tokens)
    }

    pub fn sample(
        &self,
        logits: &Tensor,
        sampling_params: &Option<SamplingParams>,
    ) -> Result<Vec<u32>> {
        let logits = logits.to_dtype(DType::F32)?;
        let batch = logits.layout().dims()[0];
        let prs = |temperature: f64| -> Result<Tensor> {
            let logits = (&logits / temperature)?;
            let prs = candle_nn::ops::softmax_last_dim(&logits)?;
            Ok(prs)
        };

        let sampling = sampling_params.as_ref().map_or_else(
            || self.sampling.to_owned(),
            |param| LogitsProcessor::get_strategy(param.temperature, param.top_k, param.top_p),
        );

        let next_tokens = match &sampling {
            Sampling::ArgMax => self.sample_argmax(&logits)?,
            Sampling::All { temperature } => {
                let prs = prs(*temperature as f64)?.to_vec2()?;
                (0..batch)
                    .map(|b| self.sample_multinomial(&prs[b]).unwrap())
                    .collect()
            }
            Sampling::TopP { p, temperature } => {
                let prs = prs(*temperature as f64)?;
                if *p <= 0.0 || *p >= 1.0 {
                    let prs = prs.to_vec2()?;
                    (0..batch)
                        .map(|b| self.sample_multinomial(&prs[b]).unwrap())
                        .collect()
                } else {
                    self.sample_topp(&prs, *p as f32)?
                }
            }
            Sampling::TopK { k, temperature } => {
                let prs = prs(*temperature as f64)?;
                self.sample_topk(&prs, *k)?
            }
            Sampling::TopKThenTopP { k, p, temperature } => {
                let prs = prs(*temperature as f64)?;
                self.sample_topk_topp(&prs, *k, *p as f32)?
            }
        };
        Ok(next_tokens)
    }

    pub fn apply_batch_repeat_penalty(
        &self,
        logits: &Tensor,
        penalties: Vec<f32>,
        context: Vec<Vec<u32>>,
    ) -> Result<Tensor> {
        let device = logits.device();
        let batch = logits.layout().dims()[0];
        let logits_len = logits.layout().dims()[1];
        let logits: Vec<Vec<f32>> = logits.to_dtype(candle_core::DType::F32)?.to_vec2::<f32>()?;
        let vec_ret: Vec<Vec<f32>> = (0..batch)
            .into_iter()
            .map(|b| {
                let mut logits = logits[b].to_vec();
                let mut already_seen = std::collections::HashSet::new();
                if penalties[b] != 1.0 && penalties[b] != 0. && context[b].len() > 1 {
                    for token_id in &context[b] {
                        if already_seen.contains(&token_id) {
                            continue;
                        }
                        already_seen.insert(token_id);
                        if let Some(logit) = logits.get_mut(*token_id as usize) {
                            if *logit >= 0. {
                                *logit /= penalties[b]
                            } else {
                                *logit *= penalties[b]
                            }
                        }
                    }
                }
                logits
            })
            .collect();

        let logits = vec_ret.into_iter().flatten().collect();
        Tensor::from_vec(logits, (batch, logits_len), device)
    }
}
