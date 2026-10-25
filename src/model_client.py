"""
Provider abstraction for model calls.

Experimental logic (prompts, conditions, parsing, saving) lives
elsewhere; everything specific to the CMU AI Gateway is confined to
CMUGatewayClient, so the transport can change without touching the
experiment.
"""

import os
import time
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
    latency_seconds: float | None = None
    # Non-secret request/response metadata, e.g. HTTP status,
    # provider response id, finish reason, token usage.
    request_metadata: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None and self.raw_response is not None


class GatewayNotConfigured(RuntimeError):
    pass


class ModelClient:
    provider = "base"

    def __init__(self, model: str, temperature: float):
        self.model = model
        self.temperature = temperature

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
    CMU AI Gateway transport.

    The Gateway's API style (endpoint path, auth header, request
    schema, how video/audio parts are encoded, model identifier) is
    not yet specified, so this client refuses to send anything
    rather than guess. Credentials are read from environment
    variables named in configs/eval_config.json and are never
    logged or stored.
    """

    provider = "cmu_ai_gateway"

    def __init__(self, model: str, temperature: float, gateway_config: dict, gateway_model_id: str | None):
        super().__init__(model, temperature)
        self.gateway_config = gateway_config
        self.gateway_model_id = gateway_model_id

    def check_ready(self) -> None:
        missing = [
            name
            for name in (
                self.gateway_config["base_url_env_var"],
                self.gateway_config["api_key_env_var"],
            )
            if not os.environ.get(name)
        ]
        problems = []

        if missing:
            problems.append(f"environment variables not set: {missing}")
        if not self.gateway_config.get("interface"):
            problems.append(
                "gateway.interface is not specified in configs/eval_config.json"
            )
        if not self.gateway_model_id:
            problems.append(
                "model.gateway_model_id is not specified in configs/eval_config.json"
            )

        if problems:
            raise GatewayNotConfigured(
                "CMU AI Gateway is not configured: " + "; ".join(problems)
            )

    def _send(self, prompt: Prompt) -> tuple[str, dict]:
        # Intentionally unimplemented until the Gateway interface is
        # documented. Implement here only: build the request from
        # prompt.text and prompt.media (path + mime_type), send it once
        # with temperature=self.temperature, and return the response
        # text plus non-secret metadata.
        raise GatewayNotConfigured(
            "CMU AI Gateway request format is not implemented yet"
        )


def make_client(config: dict) -> ModelClient:
    model = config["model"]

    if model["provider"] == "cmu_ai_gateway":
        return CMUGatewayClient(
            model=model["model_id"],
            temperature=config["generation"]["temperature"],
            gateway_config=config["gateway"],
            gateway_model_id=model.get("gateway_model_id"),
        )

    raise ValueError(f"unknown provider {model['provider']!r}")
