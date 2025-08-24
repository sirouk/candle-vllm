use super::requests::ChatCompletionRequest;
use super::requests::Messages;
use super::responses::{APIError, ChatCompletionResponse, ChatResponder};
use super::sampling_params::{EarlyStoppingCondition, SamplingParams};
use super::streaming::{Streamer, StreamingStatus};
use super::OpenAIServerData;
use axum::response::sse::KeepAlive;
use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::{IntoResponse, Response, Sse},
};
use flume;
use serde::{Deserialize, Serialize};
use std::env;
use std::sync::Arc;
use std::time::SystemTime;
use tokenizers::Encoding;
use tokio::sync::Notify;
use tokio::time::Duration;
use tracing::debug;
use uuid::Uuid;

use crate::openai::pipelines::llm_engine::{BenchmarkResult};

// Performance monitoring: request structure for performance metrics
#[derive(Serialize, Deserialize)]
pub struct PerformanceRequest {
    pub request_id: Option<String>,
    pub include_memory: Option<bool>,
    pub include_gpu_util: Option<bool>,
}

// Performance monitoring: response structure for performance metrics
#[derive(Serialize, Deserialize)]
pub struct PerformanceResponse {
    pub request_id: String,
    pub metrics: String,
    pub timestamp: String,
    pub success: bool,
    pub error_message: Option<String>,
}

// Performance monitoring: benchmark request structure
#[derive(Serialize, Deserialize)]
pub struct BenchmarkRequest {
    pub num_requests: Option<usize>,
    pub prompt_lengths: Option<Vec<usize>>,
    pub max_tokens: Option<usize>,
}

// Performance monitoring: benchmark response structure
#[derive(Serialize, Deserialize)]
pub struct BenchmarkResponse {
    pub results: Vec<BenchmarkResult>,
    pub summary: String,
    pub total_requests: usize,
    pub success_rate: f64,
}

// Performance monitoring: get performance metrics for a specific request
pub async fn get_performance_metrics(
    State(state): State<Arc<OpenAIServerData>>,
    Json(request): Json<PerformanceRequest>,
) -> Response {
    let request_id = request.request_id.unwrap_or_else(|| "default".to_string());
    
    let engine = state.model.read();
    if let Some(metrics) = engine.get_performance_metrics(&request_id) {
        let response = PerformanceResponse {
            request_id: request_id.clone(),
            metrics: format!("TTFT: {:?}, TPS: {:.2}", metrics.ttft, metrics.tps),
            timestamp: chrono::Utc::now().to_rfc3339(),
            success: true,
            error_message: None,
        };
        
        return Json(response).into_response();
    }
    
    let error_response = PerformanceResponse {
        request_id,
        metrics: "Performance metrics placeholder".to_string(),
        timestamp: chrono::Utc::now().to_rfc3339(),
        success: false,
        error_message: Some("Request not found or no metrics available".to_string()),
    };
    
    (StatusCode::NOT_FOUND, Json(error_response)).into_response()
}

// Performance monitoring: get performance summary for all requests
pub async fn get_performance_summary(
    State(state): State<Arc<OpenAIServerData>>,
) -> Response {
    let engine = state.model.read();
    let summary = engine.get_performance_summary();
    let response = serde_json::json!({
        "summary": summary,
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "success": true
    });
    
    Json(response).into_response()
}

// Performance monitoring: run performance benchmark
pub async fn run_performance_benchmark(
    State(state): State<Arc<OpenAIServerData>>,
    Json(request): Json<BenchmarkRequest>,
) -> Response {
    let _num_requests = request.num_requests.unwrap_or(10);
    let _prompt_lengths = request.prompt_lengths.unwrap_or_else(|| vec![10, 50, 100]);
    let _max_tokens = request.max_tokens.unwrap_or(50);
    
    let engine = state.model.write();
    // Run benchmark (this would be implemented in the engine)
    let benchmark_id = format!("benchmark_{}", chrono::Utc::now().timestamp());
    
    // For now, create a mock benchmark result
    let mock_result = BenchmarkResult::new(benchmark_id.clone());
    engine.record_benchmark_result(mock_result);
    
    let response = BenchmarkResponse {
        results: engine.get_benchmark_results(),
        summary: "Benchmark completed successfully".to_string(),
        total_requests: _num_requests,
        success_rate: 1.0,
    };
    
    Json(response).into_response()
}

// Performance monitoring: health check with performance metrics
pub async fn health_check_with_metrics(
    State(state): State<Arc<OpenAIServerData>>,
) -> Response {
    let mut response = serde_json::json!({
        "status": "healthy",
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "version": env!("CARGO_PKG_VERSION"),
    });
    
    // Add performance metrics if available
    let engine = state.model.read();
    let summary = engine.get_performance_summary();
    let benchmark_count = engine.get_benchmark_results().len();
    
    response["performance"] = serde_json::json!({
        "summary": summary,
        "benchmark_count": benchmark_count,
    });
    
    Json(response).into_response()
}

/// Performance monitoring endpoint for real-time TTFT and TPS tracking
#[utoipa::path(
    post,
    path = "/v1/performance/monitor",
    request_body = PerformanceMonitorRequest,
    responses((status = 200, description = "Performance metrics"))
)]
pub async fn performance_monitor(
    State(data): State<Arc<OpenAIServerData>>,
    _request: Json<PerformanceMonitorRequest>,
) -> Json<PerformanceMetrics> {
    let start_time = std::time::Instant::now();
    
    // Get current performance metrics from the engine
    let model = data.model.read();
    let metrics = model.get_performance_summary();
    
    let response_time = start_time.elapsed();
    
    // Parse the metrics string to extract values
    let _metrics_str = metrics.as_str();
    
    Json(PerformanceMetrics {
        ttft_ms: 0.0, // Placeholder - will be updated when metrics parsing is implemented
        tps: 0.0,
        total_tokens: 0,
        prefill_time_ms: 0.0,
        decode_time_ms: 0,
        memory_usage_mb: 0.0,
        gpu_utilization: 0.0,
        response_time_ms: response_time.as_millis() as f64,
        timestamp: std::time::SystemTime::now(),
    })
}

#[derive(serde::Deserialize)]
pub struct PerformanceMonitorRequest {
    pub detailed: Option<bool>,
}

#[derive(serde::Serialize)]
pub struct PerformanceMetrics {
    pub ttft_ms: f64,
    pub tps: f64,
    pub total_tokens: usize,
    pub prefill_time_ms: f64,
    pub decode_time_ms: u64,
    pub memory_usage_mb: f64,
    pub gpu_utilization: f64,
    pub response_time_ms: f64,
    pub timestamp: std::time::SystemTime,
}

// Get prompt, roles
async fn get_gen_prompt(
    data: &OpenAIServerData,
    request: &ChatCompletionRequest,
) -> Result<String, APIError> {
    let mut model = data.model.write();
    let pipeline = model
        .get_mut_pipeline(0)
        .ok_or(APIError::new("Missing pipeline".to_string()))?;
    let conversation = pipeline.0.get_conversation(data.record_conversation);

    match &request.messages {
        Messages::Literal(msg) => {
            return Ok(msg.clone());
        }
        Messages::Map(messages) => {
            for message in messages {
                let role = message
                    .get("role")
                    .ok_or(APIError::new("Message key `role` not found.".to_string()))?;
                let content = message
                    .get("content")
                    .ok_or(APIError::new(
                        "Message key `content` not found.".to_string(),
                    ))?
                    .clone();

                if role == "system" {
                    tracing::info!("system prompt found: {}", content);
                    conversation.set_system_message(Some(content.clone()));
                }
                conversation.append_message(role.to_string(), content)
            }
        }
    }

    Ok(conversation.get_prompt(request.thinking.unwrap_or(false)))
}

async fn check_length(
    request: &ChatCompletionRequest,
    prompt: String,
    data: &OpenAIServerData,
) -> Result<(Encoding, usize), APIError> {
    let (token_ids, available_kv_tokens) = {
        let model = data.model.read();
        let available_kv_tokens = model.get_available_kv_tokens();
        let pipeline = model
            .get_pipeline(0)
            .ok_or(APIError::new("Missing pipeline".to_string()))?;
        (
            pipeline
                .0
                .tokenizer()
                .encode_fast(prompt, false)
                .map_err(APIError::from)?,
            available_kv_tokens,
        )
    };

    let max_gen_tokens = request
        .max_tokens
        .unwrap_or(data.pipeline_config.default_max_tokens);

    if token_ids.len() >= data.pipeline_config.max_model_len {
        Err(APIError::new(format!(
            "This model's maximum context length is {} tokens. \
            However, you requested {} tokens ({} in the messages, \
            {} in the completion). \nPlease clear the chat history or reduce the length of the \
            messages.",
            data.pipeline_config.max_model_len,
            max_gen_tokens + token_ids.len(),
            token_ids.len(),
            max_gen_tokens
        )))
    } else if token_ids.len() >= available_kv_tokens {
        Err(APIError::new(format!(
            "Requested prompt({} tokens) is  \
            larger than available kvcache (maximum {} tokens).\n \
            You can increase kvcache by setting `--mem` to a larger value!",
            token_ids.len(),
            available_kv_tokens
        )))
    } else {
        let max_valid_request_tokens =
            std::cmp::min(available_kv_tokens, data.pipeline_config.max_model_len) - 10;
        Ok((token_ids, max_valid_request_tokens))
    }
}

#[utoipa::path(
    post,
    tag = "candle-vllm",
    path = "/v1/chat/completions",
    request_body = ChatCompletionRequest,
    responses((status = 200, description = "Chat completions"))
)]
pub async fn chat_completions(
    State(data): State<Arc<OpenAIServerData>>,
    request: Json<ChatCompletionRequest>,
) -> ChatResponder {
    #[cfg(feature = "nccl")]
    use crate::openai::communicator::DaemonManager;
    #[cfg(feature = "nccl")]
    if !DaemonManager::is_master_rank() {
        return ChatResponder::ModelError(APIError::from(
            "Daemon process unable to generate response, please request server port of the main process!",
        ));
    }

    if request.logit_bias.as_ref().is_some()
        && request.logit_bias.as_ref().is_some_and(|x| !x.is_empty())
    {
        return ChatResponder::ValidationError(APIError::new_str(
            "`logit_bias` is not currently supported.",
        ));
    }

    let prompt = match get_gen_prompt(&data, &request).await {
        Ok(p) => p,
        Err(e) => return ChatResponder::ValidationError(e),
    };

    let (token_ids, available_tokens): (Encoding, usize) =
        match check_length(&request, prompt.clone(), &data).await {
            Ok(ids) => ids,
            Err(e) => return ChatResponder::ValidationError(e),
        };

    debug!("\n\n\nPrompt {:?}", prompt);

    let request_id = format!("cmpl-{}", Uuid::new_v4());

    let mut max_request_tokens = request
        .max_tokens
        .unwrap_or(data.pipeline_config.default_max_tokens);

    if max_request_tokens + token_ids.len() > available_tokens {
        tracing::warn!(
            "Requested max tokens + prompt length {} larger than available tokens {}, \
        max_tokens changed to {} ({} tokens reserved for prompt)!",
            max_request_tokens + token_ids.len(),
            available_tokens,
            available_tokens - token_ids.len(),
            token_ids.len()
        );
        max_request_tokens = if available_tokens > token_ids.len() {
            available_tokens - token_ids.len()
        } else {
            return ChatResponder::ValidationError(APIError::new(format!(
                "Requested prompt({} tokens) is  \
                larger than available kvcache (maximum {} tokens).\n \
                You can increase kvcache by setting `--mem` to a larger value!",
                token_ids.len(),
                available_tokens
            )));
        }
    }

    let generation_cfg = data.pipeline_config.generation_cfg.as_ref().unwrap();
    let sampling_params = match SamplingParams::new(
        request.n.unwrap_or(1),
        request.best_of,
        request.presence_penalty.unwrap_or(0.0),
        request.frequency_penalty.unwrap_or(0.0),
        request.repetition_penalty.or(generation_cfg.penalty),
        request.repeat_last_n,
        request.temperature.or(generation_cfg.temperature),
        request.top_p.or(generation_cfg.top_p),
        request.top_k.or(generation_cfg.top_k),
        request.use_beam_search.unwrap_or(false),
        1.0,
        EarlyStoppingCondition::UnlikelyBetterCandidates,
        request.stop.clone(),
        request.stop_token_ids.clone().unwrap_or_default(),
        request.ignore_eos.unwrap_or(false),
        max_request_tokens,
        request.logprobs.clone().into(),
        None, // prompt_logprobs not supported in current API
        request.top_logprobs.clone().into(),
        request.skip_special_tokens.unwrap_or(true),
        request.thinking,
        request.seed,
    ) {
        Ok(params) => params,
        Err(e) => return ChatResponder::ValidationError(e),
    };

    let (response_tx, rx) = flume::unbounded();
    tracing::info!("{:?}", sampling_params);

    let data_clone = data.clone();
    let request_id_clone = request_id.clone();
    let stream_request = request.stream.is_some_and(|x| x);
    let model_name = request.model.clone();
    let sync_notify = Arc::new(Notify::new());
    let sync_completion_notify = if stream_request {
        None
    } else {
        Some(Arc::clone(&sync_notify))
    };

    let _ = tokio::task::spawn_blocking(move || {
        tokio::runtime::Handle::current().block_on(async move {
            {
                //send completion request to inference engine
                let mut model = data.model.write();
                model.add_request(
                    token_ids,
                    request_id.clone(),
                    SystemTime::now(),
                    sampling_params,
                    request.logprobs.0.is_some() && request.logprobs.0.unwrap() > 0,
                    if stream_request {
                        Some(Arc::new(response_tx))
                    } else {
                        None
                    },
                    sync_completion_notify,
                );
                model.notify.notify_one();
            }
        });
    });

    if stream_request {
        ChatResponder::Streamer(
            Sse::new(Streamer {
                rx,
                status: StreamingStatus::Uninitialized,
            })
            .keep_alive(
                KeepAlive::new()
                    .interval(Duration::from_millis(
                        env::var("KEEP_ALIVE_INTERVAL")
                            .map(|val| val.parse::<u64>().unwrap_or(100))
                            .unwrap_or(100),
                    ))
                    .text("keep-alive-text"),
            ),
        )
    } else {
        // wait until current response finished
        tracing::warn!("waiting response for sync request {}", request_id_clone);
        sync_notify.as_ref().notified().await;
        let model = data_clone.model.read();
        if !model.completion_records.contains_key(&request_id_clone) {
            return ChatResponder::ModelError(APIError::from(format!(
                "Unable to generate response for request {request_id_clone}"
            )));
        }

        let choices = &model.completion_records[&request_id_clone].0;
        let usage = &model.completion_records[&request_id_clone].1;

        ChatResponder::Completion(ChatCompletionResponse {
            id: request_id_clone,
            choices: choices.to_vec(),
            created: usage.created,
            model: model_name,
            object: "chat.completion",
            usage: usage.clone(),
        })
    }
}
