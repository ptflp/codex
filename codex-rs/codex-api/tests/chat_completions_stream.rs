use anyhow::Result;
use async_trait::async_trait;
use bytes::Bytes;
use codex_api::AuthProvider;
use codex_api::ChatCompletionsClient;
use codex_api::ChatCompletionsOptions;
use codex_api::Provider;
use codex_api::common::ResponseEvent;
use codex_api::requests::responses::Compression;
use codex_client::HttpTransport;
use codex_client::Request;
use codex_client::Response;
use codex_client::StreamResponse;
use codex_client::TransportError;
use http::HeaderMap;
use http::Method;
use http::StatusCode;
use pretty_assertions::assert_eq;

#[derive(Clone, Default)]
struct NoAuth;

impl AuthProvider for NoAuth {
    fn bearer_token(&self) -> Option<String> {
        None
    }
}

#[derive(Clone)]
struct StaticStreamTransport {
    bytes: Vec<Bytes>,
}

#[async_trait]
impl HttpTransport for StaticStreamTransport {
    async fn execute(&self, _req: Request) -> Result<Response, TransportError> {
        Err(TransportError::Build("execute should not run".to_string()))
    }

    async fn stream(&self, req: Request) -> Result<StreamResponse, TransportError> {
        assert_eq!(req.method, Method::POST);
        assert!(
            req.url.ends_with("/v1/chat/completions"),
            "unexpected url: {}",
            req.url
        );

        let stream = futures::stream::iter(
            self.bytes
                .clone()
                .into_iter()
                .map(Ok::<Bytes, TransportError>),
        );
        Ok(StreamResponse {
            status: StatusCode::OK,
            headers: HeaderMap::new(),
            bytes: Box::pin(stream),
        })
    }
}

#[tokio::test]
async fn chat_completions_stream_emits_response_events() -> Result<()> {
    let provider = Provider {
        name: "mock".to_string(),
        base_url: "http://example.test/v1".to_string(),
        query_params: None,
        headers: HeaderMap::new(),
        retry: codex_api::provider::RetryConfig {
            max_attempts: 1,
            base_delay: std::time::Duration::from_millis(1),
            retry_429: false,
            retry_5xx: false,
            retry_transport: false,
        },
        stream_idle_timeout: std::time::Duration::from_secs(5),
    };

    let sse = |json: &str| Bytes::from(format!("data: {json}\n\n"));

    let transport = StaticStreamTransport {
        bytes: vec![
            sse(
                r#"{"id":"chatcmpl-1","choices":[{"delta":{"role":"assistant","content":"hi"},"finish_reason":null}]}"#,
            ),
            sse(
                r#"{"id":"chatcmpl-1","choices":[{"delta":{"content":" there"},"finish_reason":null}]}"#,
            ),
            sse(
                r#"{"id":"chatcmpl-1","choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}"#,
            ),
            Bytes::from("data: [DONE]\n\n"),
        ],
    };

    let client = ChatCompletionsClient::new(transport, provider, NoAuth)
        .with_telemetry(/*request*/ None, /*sse*/ None);
    let mut stream = client
        .stream(
            serde_json::json!({
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": true,
            }),
            ChatCompletionsOptions {
                conversation_id: None,
                session_source: None,
                extra_headers: HeaderMap::new(),
                compression: Compression::None,
                turn_state: None,
            },
        )
        .await?;

    let mut events = Vec::new();
    while let Ok(event) = stream.rx_event.recv().await.unwrap() {
        events.push(event);
        if matches!(events.last(), Some(ResponseEvent::Completed { .. })) {
            break;
        }
    }

    assert!(matches!(events[0], ResponseEvent::Created));
    assert_eq!(
        events
            .iter()
            .filter(|e| matches!(e, ResponseEvent::OutputTextDelta(_)))
            .count(),
        2
    );
    Ok(())
}
