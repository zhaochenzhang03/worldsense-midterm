"""
Provider abstraction for model calls.

Experimental logic (prompts, conditions, parsing, saving) lives
elsewhere; everything specific to the CMU AI Gateway is confined to
CMUGatewayClient, so the transport can change without touching the
experiment.
"""

import base64
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from src.answer_parser import parse_answer
from src.prompting import Prompt


@dataclass
class ModelResponse:
    raw_response: str | None
    parsed_prediction: str | None
    parse_status: str
    provider: str
    model: str
    error: str | None = None
    # True if the failed request returned no completion and can be
    # re-sent without risk of duplicating a success.
    transient: bool = False
    latency_seconds: float | None = None
    # Non-secret request/response metadata, e.g. HTTP status,
    # provider response id, finish reason, token usage.
    request_metadata: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None and self.raw_response is not None


class GatewayNotConfigured(RuntimeError):
    pass


class TransientError(RuntimeError):
    """
    A failed request that returned no completion (HTTP 408/429/5xx,
    timeout, connection error). Re-sending it cannot duplicate a
    saved success.
    """


class ModelClient:
    provider = "base"

    def __init__(self, model: str, temperature: float, max_tokens: int | None = None):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def check_ready(self) -> None:
        """
        Raise if the client cannot send requests. Called once before
        any example is processed, so misconfiguration fails fast.
        """

    def _send(self, prompt: Prompt) -> tuple[str, dict]:
        """
        Send one request. Return (raw response text, metadata).
        Must not retry, so a successful request is never duplicated.
        """
        raise NotImplementedError

    def generate(
        self,
        example: dict,
        condition: str,
        prompt: Prompt,
        media_paths: list[str],
    ) -> ModelResponse:
        if [m.path for m in prompt.media] != list(media_paths):
            raise ValueError("media_paths do not match the prompt's media")
        if prompt.condition != condition:
            raise ValueError("prompt was built for a different condition")

        start = time.monotonic()

        try:
            raw, metadata = self._send(prompt)
        except Exception as error:
            return ModelResponse(
                raw_response=None,
                parsed_prediction=None,
                parse_status="invalid",
                provider=self.provider,
                model=self.model,
                error=f"{type(error).__name__}: {error}",
                transient=isinstance(error, TransientError),
                latency_seconds=time.monotonic() - start,
            )

        parsed = parse_answer(raw, prompt.valid_letters)

        return ModelResponse(
            raw_response=raw,
            parsed_prediction=parsed.prediction,
            parse_status=parsed.status,
            provider=self.provider,
            model=self.model,
            latency_seconds=time.monotonic() - start,
            request_metadata=metadata,
        )


class CMUGatewayClient(ModelClient):
    """
    CMU AI Gateway transport (LiteLLM, OpenAI-compatible).

    One POST /v1/chat/completions per request. Media is sent inline as
    base64 data URLs in OpenAI/LiteLLM "file" content parts
    ({"type": "file", "file": {"file_data": "data:<mime>;base64,...",
    "format": "<mime>"}}), which LiteLLM forwards to Gemini as inline
    data with the given MIME type. Media parts come before the text
    part. The API key is read from the environment variable named in
    configs/eval_config.json and is never logged or stored.
    """

    provider = "cmu_ai_gateway"
    timeout_seconds = 300
    transient_http_statuses = {408, 429, 500, 502, 503, 504}

    def __init__(
        self,
        model: str,
        temperature: float,
        max_tokens: int | None,
        gateway_config: dict,
        gateway_model_id: str | None,
    ):
        super().__init__(model, temperature, max_tokens)
        self.gateway_config = gateway_config
        self.gateway_model_id = gateway_model_id

    def check_ready(self) -> None:
        problems = []

        api_key_env_var = self.gateway_config["api_key_env_var"]
        if not os.environ.get(api_key_env_var):
            problems.append(f"environment variable {api_key_env_var} not set")

        for key in ("base_url", "interface", "media_request_format"):
            if not self.gateway_config.get(key):
                problems.append(
                    f"gateway.{key} is not specified in configs/eval_config.json"
                )
        if not self.gateway_model_id:
            problems.append(
                "model.gateway_model_id is not specified in configs/eval_config.json"
            )

        if problems:
            raise GatewayNotConfigured(
                "CMU AI Gateway is not configured: " + "; ".join(problems)
            )

    def _api_key(self) -> str:
        return os.environ[self.gateway_config["api_key_env_var"]]

    def _redact(self, text: str) -> str:
        key = self._api_key()
        return text.replace(key, "[REDACTED]") if key else text

    def _build_payload(self, prompt: Prompt) -> dict:
        content = []

        for media in prompt.media:
            with open(media.path, "rb") as f:
                data = base64.b64encode(f.read()).decode("ascii")
            content.append({
                "type": "file",
                "file": {
                    "file_data": f"data:{media.mime_type};base64,{data}",
                    "format": media.mime_type,
                },
            })

        content.append({"type": "text", "text": prompt.text})

        payload = {
            "model": self.gateway_model_id,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        return payload

    def _send(self, prompt: Prompt) -> tuple[str, dict]:
        body = json.dumps(self._build_payload(prompt)).encode("utf-8")
        request = urllib.request.Request(
            self.gateway_config["base_url"].rstrip("/") + "/v1/chat/completions",
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self._api_key()}",
                "Content-Type": "application/json",
            },
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                status = response.status
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = self._redact(error.read().decode("utf-8", "replace"))[:2000]
            message = f"HTTP {error.code}: {detail}"
            if error.code in self.transient_http_statuses:
                raise TransientError(message) from None
            raise RuntimeError(message) from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            raise TransientError(self._redact(f"{type(error).__name__}: {error}")) from None

        choice = payload["choices"][0]
        # A completion without text (e.g. blocked, or cut off by
        # max_tokens before any text) is still a model outcome, not a
        # transport failure: keep it as an empty response so it is
        # saved, parsed as invalid and never re-sent.
        text = choice["message"].get("content") or ""

        metadata = {
            "http_status": status,
            "request_bytes": len(body),
            "response_id": payload.get("id"),
            "response_model": payload.get("model"),
            "finish_reason": choice.get("finish_reason"),
            "usage": payload.get("usage"),
            "response_json": payload,
        }

        return text, metadata


def make_client(config: dict) -> ModelClient:
    model = config["model"]

    if model["provider"] == "cmu_ai_gateway":
        return CMUGatewayClient(
            model=model["model_id"],
            temperature=config["generation"]["temperature"],
            max_tokens=config["generation"].get("max_tokens"),
            gateway_config=config["gateway"],
            gateway_model_id=model.get("gateway_model_id"),
        )

    raise ValueError(f"unknown provider {model['provider']!r}")
