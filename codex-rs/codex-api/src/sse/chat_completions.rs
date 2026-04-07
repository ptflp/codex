use crate::common::ResponseEvent;
use crate::common::ResponseStream;
use crate::error::ApiError;
use crate::telemetry::SseTelemetry;
use codex_client::StreamResponse;
use codex_protocol::models::ContentItem;
use codex_protocol::models::MessagePhase;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::TokenUsage;
use eventsource_stream::Eventsource;
use futures::StreamExt;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::OnceLock;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::time::Instant;
use tokio::time::timeout;
use tracing::debug;
use tracing::trace;

#[derive(Debug, Deserialize)]
struct ChatCompletionsStreamChunk {
    id: Option<String>,
    choices: Option<Vec<ChatCompletionsChoice>>,
    usage: Option<ChatCompletionsUsage>,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionsChoice {
    delta: Option<ChatCompletionsDelta>,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionsDelta {
    role: Option<String>,
    content: Option<String>,
    tool_calls: Option<Vec<ChatCompletionsToolCallDelta>>,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionsToolCallDelta {
    index: Option<usize>,
    id: Option<String>,
    function: Option<ChatCompletionsToolFunctionDelta>,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionsToolFunctionDelta {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionsUsage {
    prompt_tokens: Option<i64>,
    completion_tokens: Option<i64>,
    total_tokens: Option<i64>,
    #[serde(default)]
    prompt_tokens_details: Option<ChatCompletionsPromptTokensDetails>,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionsPromptTokensDetails {
    cached_tokens: Option<i64>,
}

#[derive(Clone, Debug, Default)]
struct PendingToolCall {
    id: Option<String>,
    name: Option<String>,
    arguments: String,
}

pub fn spawn_chat_completions_stream(
    stream_response: StreamResponse,
    idle_timeout: Duration,
    telemetry: Option<Arc<dyn SseTelemetry>>,
    turn_state: Option<Arc<OnceLock<String>>>,
) -> ResponseStream {
    if let Some(turn_state) = turn_state.as_ref()
        && let Some(header_value) = stream_response
            .headers
            .get("x-codex-turn-state")
            .and_then(|v| v.to_str().ok())
    {
        let _ = turn_state.set(header_value.to_string());
    }

    let (tx_event, rx_event) = mpsc::channel::<Result<ResponseEvent, ApiError>>(1600);
    tokio::spawn(async move {
        process_chat_completions_sse(stream_response.bytes, tx_event, idle_timeout, telemetry)
            .await;
    });

    ResponseStream { rx_event }
}

async fn process_chat_completions_sse(
    bytes: codex_client::ByteStream,
    tx_event: mpsc::Sender<Result<ResponseEvent, ApiError>>,
    idle_timeout: Duration,
    telemetry: Option<Arc<dyn SseTelemetry>>,
) {
    let mut tx_event = tx_event;

    // Eagerly emit Created so downstream session state matches Responses API behavior.
    let _ = tx_event.send(Ok(ResponseEvent::Created)).await;

    let mut event_stream = bytes.eventsource();
    let mut response_id: Option<String> = None;
    let mut assistant_message_id: Option<String> = None;
    let mut assistant_message_seq: u64 = 0;
    let mut assistant_text = String::new();
    let mut pending_tool_calls: BTreeMap<usize, PendingToolCall> = BTreeMap::new();
    let mut final_usage: Option<TokenUsage> = None;

    loop {
        let start = Instant::now();
        let response = timeout(idle_timeout, event_stream.next()).await;
        if let Some(t) = telemetry.as_ref() {
            t.on_sse_poll(&response, start.elapsed());
        }

        let event = match response {
            Ok(Some(Ok(event))) => event,
            Ok(Some(Err(err))) => {
                let _ = tx_event
                    .send(Err(ApiError::Stream(format!(
                        "chat completions SSE stream error: {err}"
                    ))))
                    .await;
                return;
            }
            Ok(None) => break,
            Err(_) => {
                let _ = tx_event
                    .send(Err(ApiError::Stream(
                        "chat completions SSE idle timeout".to_string(),
                    )))
                    .await;
                return;
            }
        };

        if event.data.trim() == "[DONE]" {
            break;
        }

        trace!(data = %event.data, "chat completions SSE event");

        let chunk: ChatCompletionsStreamChunk = match serde_json::from_str(&event.data) {
            Ok(chunk) => chunk,
            Err(err) => {
                let _ = tx_event
                    .send(Err(ApiError::Stream(format!(
                        "failed to parse chat completions chunk: {err}"
                    ))))
                    .await;
                return;
            }
        };

        if response_id.is_none() {
            response_id = chunk.id.clone();
        }

        if let Some(usage) = chunk.usage.as_ref() {
            final_usage = Some(TokenUsage {
                input_tokens: usage.prompt_tokens.unwrap_or(0),
                cached_input_tokens: usage
                    .prompt_tokens_details
                    .as_ref()
                    .and_then(|d| d.cached_tokens)
                    .unwrap_or(0),
                output_tokens: usage.completion_tokens.unwrap_or(0),
                reasoning_output_tokens: 0,
                total_tokens: usage.total_tokens.unwrap_or_else(|| {
                    usage.prompt_tokens.unwrap_or(0) + usage.completion_tokens.unwrap_or(0)
                }),
            });
        }

        let Some(choice) = chunk.choices.as_ref().and_then(|choices| choices.first()) else {
            continue;
        };

        if let Some(delta) = choice.delta.as_ref() {
            if let Some(role) = delta.role.as_ref() {
                debug!(role, "chat completions delta role");
            }

            if let Some(content) = delta.content.as_ref() {
                if assistant_message_id.is_none() {
                    assistant_message_seq += 1;
                    let id = format!("msg_{assistant_message_seq}");
                    assistant_message_id = Some(id.clone());
                    let _ = tx_event
                        .send(Ok(ResponseEvent::OutputItemAdded(ResponseItem::Message {
                            id: Some(id),
                            role: "assistant".to_string(),
                            content: vec![ContentItem::OutputText {
                                text: String::new(),
                            }],
                            end_turn: None,
                            phase: Some(MessagePhase::Commentary),
                        })))
                        .await;
                }

                assistant_text.push_str(content);
                let _ = tx_event
                    .send(Ok(ResponseEvent::OutputTextDelta(content.clone())))
                    .await;
            }

            if let Some(tool_calls) = delta.tool_calls.as_ref() {
                for tool_call in tool_calls {
                    let index = tool_call.index.unwrap_or(0);
                    let entry = pending_tool_calls.entry(index).or_default();
                    if entry.id.is_none() {
                        entry.id = tool_call.id.clone();
                    }
                    if let Some(function) = tool_call.function.as_ref() {
                        if entry.name.is_none() {
                            entry.name = function.name.clone();
                        }
                        if let Some(arguments) = function.arguments.as_ref() {
                            entry.arguments.push_str(arguments);
                        }
                    }
                }
            }
        }

        if let Some(finish_reason) = choice.finish_reason.as_ref() {
            if finish_reason == "tool_calls" {
                emit_pending_tool_calls(&mut tx_event, &pending_tool_calls).await;
                pending_tool_calls.clear();
            }
        }
    }

    if !pending_tool_calls.is_empty() {
        emit_pending_tool_calls(&mut tx_event, &pending_tool_calls).await;
    }

    if let Some(id) = assistant_message_id.take()
        && !assistant_text.trim().is_empty()
    {
        let _ = tx_event
            .send(Ok(ResponseEvent::OutputItemDone(ResponseItem::Message {
                id: Some(id),
                role: "assistant".to_string(),
                content: vec![ContentItem::OutputText {
                    text: assistant_text,
                }],
                end_turn: Some(true),
                phase: Some(MessagePhase::FinalAnswer),
            })))
            .await;
    }

    let _ = tx_event
        .send(Ok(ResponseEvent::Completed {
            response_id: response_id.unwrap_or_else(|| "chatcmpl_unknown".to_string()),
            token_usage: final_usage,
        }))
        .await;
}

async fn emit_pending_tool_calls(
    tx_event: &mut mpsc::Sender<Result<ResponseEvent, ApiError>>,
    pending: &BTreeMap<usize, PendingToolCall>,
) {
    for call in pending.values() {
        let call_id = call
            .id
            .clone()
            .unwrap_or_else(|| "call_unknown".to_string());
        let name = call.name.clone().unwrap_or_else(|| "unknown".to_string());
        let _ = tx_event
            .send(Ok(ResponseEvent::OutputItemDone(
                ResponseItem::FunctionCall {
                    id: None,
                    name,
                    namespace: None,
                    arguments: call.arguments.clone(),
                    call_id,
                },
            )))
            .await;
    }
}
