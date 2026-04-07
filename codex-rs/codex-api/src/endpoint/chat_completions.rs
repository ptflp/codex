use crate::auth::AuthProvider;
use crate::common::ResponseStream;
use crate::endpoint::session::EndpointSession;
use crate::error::ApiError;
use crate::provider::Provider;
use crate::requests::headers::build_conversation_headers;
use crate::requests::headers::insert_header;
use crate::requests::headers::subagent_header;
use crate::requests::responses::Compression;
use crate::sse::spawn_chat_completions_stream;
use crate::telemetry::SseTelemetry;
use codex_client::HttpTransport;
use codex_client::RequestCompression;
use codex_client::RequestTelemetry;
use codex_protocol::protocol::SessionSource;
use http::HeaderMap;
use http::Method;
use serde_json::Value;
use std::sync::Arc;
use std::sync::OnceLock;
use tracing::instrument;

pub struct ChatCompletionsClient<T: HttpTransport, A: AuthProvider> {
    session: EndpointSession<T, A>,
    sse_telemetry: Option<Arc<dyn SseTelemetry>>,
}

#[derive(Default)]
pub struct ChatCompletionsOptions {
    pub conversation_id: Option<String>,
    pub session_source: Option<SessionSource>,
    pub extra_headers: HeaderMap,
    pub compression: Compression,
    pub turn_state: Option<Arc<OnceLock<String>>>,
}

impl<T: HttpTransport, A: AuthProvider> ChatCompletionsClient<T, A> {
    pub fn new(transport: T, provider: Provider, auth: A) -> Self {
        Self {
            session: EndpointSession::new(transport, provider, auth),
            sse_telemetry: None,
        }
    }

    pub fn with_telemetry(
        self,
        request: Option<Arc<dyn RequestTelemetry>>,
        sse: Option<Arc<dyn SseTelemetry>>,
    ) -> Self {
        Self {
            session: self.session.with_request_telemetry(request),
            sse_telemetry: sse,
        }
    }

    fn path() -> &'static str {
        "chat/completions"
    }

    #[instrument(
        name = "chat_completions.stream",
        level = "info",
        skip_all,
        fields(
            transport = "chat_completions_http",
            http.method = "POST",
            api.path = "chat_completions"
        )
    )]
    pub async fn stream(
        &self,
        body: Value,
        mut options: ChatCompletionsOptions,
    ) -> Result<ResponseStream, ApiError> {
        if let Some(ref conv_id) = options.conversation_id {
            insert_header(&mut options.extra_headers, "x-client-request-id", conv_id);
        }
        options
            .extra_headers
            .extend(build_conversation_headers(options.conversation_id.clone()));
        if let Some(subagent) = subagent_header(&options.session_source) {
            insert_header(&mut options.extra_headers, "x-openai-subagent", &subagent);
        }

        let stream = self
            .session
            .stream_with(
                Method::POST,
                Self::path(),
                options.extra_headers,
                Some(body),
                |req| match options.compression {
                    Compression::None => {}
                    Compression::Zstd => {
                        req.compression = RequestCompression::Zstd;
                    }
                },
            )
            .await?;

        Ok(spawn_chat_completions_stream(
            stream,
            self.session.provider().stream_idle_timeout,
            self.sse_telemetry.clone(),
            options.turn_state,
        ))
    }
}
