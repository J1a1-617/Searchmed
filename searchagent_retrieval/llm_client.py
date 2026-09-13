from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional

from .tool_pipeline import ToolExecutionPipeline

DEFAULT_BASE_URL = "https://yeysai.com/v1"
DEFAULT_MODEL_NAME = "gpt-5"


class LLMClientError(RuntimeError):
    """Raised when LLM configuration or invocation fails."""


class LLMClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: Optional[float] = None,
        connect_timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        structured_attempts: Optional[int] = None,
        max_calls_per_trace: Optional[int] = None,
        stage_call_budgets: Optional[Dict[str, int]] = None,
    ) -> None:
        self.api_key = (
            api_key
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("DEFAULT_OPENAI_API_KEY")
            or ""
        ).strip()
        if not self.api_key:
            raise LLMClientError(
                "No API key is configured. Export OPENAI_API_KEY (preferred) or DEFAULT_OPENAI_API_KEY."
            )
        self.base_url = (
            base_url
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("DEFAULT_BASE_URL")
            or DEFAULT_BASE_URL
        ).strip()
        self.model_name = (model_name or os.environ.get("MODEL_NAME") or DEFAULT_MODEL_NAME).strip()
        self.timeout = float(timeout or os.environ.get("LLM_TIMEOUT") or 60)
        self.connect_timeout = max(
            1.0,
            min(
                self.timeout,
                float(connect_timeout or os.environ.get("LLM_CONNECT_TIMEOUT") or 20),
            ),
        )
        self.max_retries = max(
            0,
            int(max_retries if max_retries is not None else os.environ.get("LLM_MAX_RETRIES") or 1),
        )
        self.structured_attempts = max(
            1,
            int(
                structured_attempts
                if structured_attempts is not None
                else os.environ.get("LLM_STRUCTURED_ATTEMPTS") or 3
            ),
        )
        self.max_calls_per_trace = max(0, int(
            max_calls_per_trace
            if max_calls_per_trace is not None
            else os.environ.get("LLM_MAX_CALLS_PER_CASE") or 0
        ))
        self.stage_call_budgets = {
            str(stage): max(0, int(limit))
            for stage, limit in (stage_call_budgets or {}).items()
            if str(stage) and int(limit) > 0
        }
        self._trace_case_id = ""
        self._trace_events: List[Dict[str, Any]] = []
        self._trace_event_sink: Optional[Callable[[str, Dict[str, Any]], Any]] = None
        self._trace_call_index = 0
        self._trace_tool_index = 0
        self._tool_pipeline = ToolExecutionPipeline(self._emit_trace_event)

    def begin_trace(
        self,
        case_id: str,
        event_sink: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
    ) -> None:
        self._trace_case_id = str(case_id)
        self._trace_events = []
        self._trace_event_sink = event_sink
        self._trace_call_index = 0
        self._trace_tool_index = 0

    def set_trace_event_sink(
        self,
        event_sink: Optional[Callable[[str, Dict[str, Any]], Any]],
    ) -> None:
        self._trace_event_sink = event_sink

    def _emit_trace_event(self, event_type: str, data: Dict[str, Any]) -> None:
        if self._trace_event_sink is not None:
            self._trace_event_sink(event_type, self._trace_jsonable(data))

    def emit_trace_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit a durable domain event through the active trace sink."""
        self._emit_trace_event(event_type, data)

    def active_tool_call_id(self) -> Optional[str]:
        """Return the model-facing tool call currently being executed."""
        return self._tool_pipeline.active_call_id

    def emit_tool_stage(
        self,
        stage: str,
        data: Dict[str, Any],
        *,
        duration_ms: Optional[float] = None,
    ) -> None:
        self._tool_pipeline.stage(stage, data, duration_ms=duration_ms)

    def execute_tool_call(
        self,
        *,
        tool: str,
        arguments: Dict[str, Any],
        handler: Callable[[], Any],
        operation: str,
        turn: int = 1,
    ) -> Dict[str, Any]:
        """Execute one logical tool call through the durable pipeline."""
        self._trace_tool_index += 1
        call_id = f"{self._trace_case_id or 'trace'}:tool:{self._trace_tool_index}"
        return self._tool_pipeline.execute(
            call_id=call_id,
            tool=tool,
            arguments=arguments,
            handler=handler,
            turn=turn,
            operation=operation,
        )

    def trace_events(self) -> List[Dict[str, Any]]:
        return [dict(event) for event in self._trace_events]

    def calls_used(self) -> int:
        return len(self._trace_events)

    def calls_remaining(self) -> Optional[int]:
        if not self.max_calls_per_trace:
            return None
        return max(0, self.max_calls_per_trace - len(self._trace_events))

    @staticmethod
    def _operation_stage(operation: str) -> str:
        if operation in {
            "function:submit_query_understanding",
            "function:submit_retrieval_plan",
            "function:submit_multistep_plan",
        }:
            return "planning"
        if operation in {
            "function:submit_safety_reflection",
            "function:summarize_answer_context",
            "function:submit_predictive_benchmark_result",
        }:
            return "finalization"
        return "retrieval"

    def stage_calls_used(self, stage: str) -> int:
        return sum(
            1 for event in self._trace_events
            if event.get("stage") == stage
        )

    @staticmethod
    def _extract_json_object(text: Optional[str]) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        candidate = str(text).strip()
        if not candidate:
            return None
        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", candidate, flags=re.IGNORECASE | re.DOTALL)
        if fenced:
            candidate = fenced.group(1).strip()
        else:
            brace = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
            if brace:
                candidate = brace.group(0).strip()
        try:
            payload = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _trace_jsonable(value: Any) -> Any:
        """Convert SDK response objects to lossless JSON-compatible data."""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(key): LLMClient._trace_jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [LLMClient._trace_jsonable(item) for item in value]
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            try:
                return LLMClient._trace_jsonable(model_dump(mode="json"))
            except TypeError:
                return LLMClient._trace_jsonable(model_dump())
        return str(value)

    def _client(self):
        from openai import OpenAI
        from httpx import Timeout

        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            # A model may legitimately need minutes to generate, but an
            # unreachable endpoint should not occupy a benchmark case for the
            # full read timeout.
            timeout=Timeout(self.timeout, connect=self.connect_timeout),
            # Retry at the logical-call layer below. Some compatible gateways
            # mislabel invalid function schemas as 429; SDK retries would then
            # wait before the client can inspect the real error body.
            max_retries=0,
        )

    @staticmethod
    def _is_non_retryable_request_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return any(marker in message for marker in (
            "invalid_function_parameters",
            "invalid schema for function",
            "invalid_request_error",
            "unsupported schema",
        ))

    @staticmethod
    def _is_connection_failure(exc: Exception) -> bool:
        """Recognize transport failures before a response was established."""
        current: Optional[BaseException] = exc
        seen = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            name = type(current).__name__.lower()
            message = str(current).lower()
            if name in {"connecttimeout", "connecterror", "apiconnectionerror"}:
                return True
            if any(marker in message for marker in (
                "connection timeout",
                "connect timeout",
                "failed to connect",
                "connection refused",
                "name or service not known",
            )):
                return True
            current = current.__cause__ or current.__context__
        return False

    def _responses_create(self, *, client: Any, **kwargs: Any) -> Any:
        operation = str(kwargs.pop("_trace_operation", "responses.create"))
        stage = self._operation_stage(operation)
        if self.max_calls_per_trace and len(self._trace_events) >= self.max_calls_per_trace:
            raise LLMClientError(
                f"LLM call budget exhausted for {self._trace_case_id or 'current trace'}: "
                f"limit={self.max_calls_per_trace}"
            )
        stage_limit = self.stage_call_budgets.get(stage)
        if stage_limit and self.stage_calls_used(stage) >= stage_limit:
            raise LLMClientError(
                f"LLM stage call budget exhausted for {stage}: limit={stage_limit}"
            )
        request: Dict[str, Any] = {
            "model": self.model_name,
            **kwargs,
        }
        request.setdefault("parallel_tool_calls", False)
        # GPT reasoning-family models on this endpoint reject temperature.
        if not self.model_name.lower().startswith(("gpt-5", "gpt-6")):
            request.setdefault("temperature", 0.3)
        else:
            request.pop("temperature", None)
        if "max_output_tokens" not in request:
            configured_limit = int(os.environ.get("LLM_MAX_OUTPUT_TOKENS") or 0)
            request["max_output_tokens"] = max(2048, configured_limit)
        started = time.perf_counter()
        self._trace_call_index += 1
        call_id = f"{self._trace_case_id or 'trace'}:llm:{self._trace_call_index}"
        event: Dict[str, Any] = {
            "call_id": call_id,
            "case_id": self._trace_case_id,
            "operation": operation,
            "stage": stage,
            "model": self.model_name,
            "max_output_tokens": request.get("max_output_tokens"),
            "input_chars": len(str(request.get("input") or "")),
            "instruction_chars": len(str(request.get("instructions") or "")),
        }
        if os.environ.get("LLM_TRACE_FULL") == "1":
            # Keep the complete API request except credentials, which are not
            # part of the Responses request object. This includes prompts,
            # conversation history, tool schemas, and tool choice.
            event["request"] = self._trace_jsonable(request)
        self._emit_trace_event("llm/request", dict(event))
        try:
            response = None
            for retry_index in range(self.max_retries + 1):
                try:
                    response = client.responses.create(**request)
                    event["retry_count"] = retry_index
                    break
                except Exception as exc:
                    if (
                        self._is_non_retryable_request_error(exc)
                        or self._is_connection_failure(exc)
                        or retry_index >= self.max_retries
                    ):
                        raise
                    self._emit_trace_event("llm/retry", {
                        "call_id": call_id,
                        "case_id": self._trace_case_id,
                        "operation": operation,
                        "stage": stage,
                        "model": self.model_name,
                        "retry_index": retry_index + 1,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    })
                    time.sleep(min(2.0, 0.5 * (2 ** retry_index)))
            if response is None:
                raise LLMClientError("LLM request completed without a response")
            event["status"] = str(getattr(response, "status", "completed") or "completed")
            event["response_id"] = str(getattr(response, "id", "") or "")
            event["output_types"] = [str(getattr(item, "type", "")) for item in getattr(response, "output", [])]
            if os.environ.get("LLM_TRACE_FULL") == "1":
                # model_dump preserves ordinary message content, reasoning
                # items, function names/arguments/call IDs, status, and usage.
                event["response"] = self._trace_jsonable(response)
                event["response_output_text"] = str(getattr(response, "output_text", None) or "")
            usage = getattr(response, "usage", None)
            for name in ("input_tokens", "output_tokens", "total_tokens"):
                value = getattr(usage, name, None) if usage is not None else None
                if value is not None:
                    event[name] = int(value)
            event["duration_seconds"] = round(time.perf_counter() - started, 4)
            self._emit_trace_event("llm/response", dict(event))
            return response
        except Exception as exc:
            event["status"] = "error"
            event["error_type"] = type(exc).__name__
            event["error"] = str(exc)[:500]
            if os.environ.get("LLM_TRACE_FULL") == "1":
                event["error_full"] = str(exc)
            event["duration_seconds"] = round(time.perf_counter() - started, 4)
            self._emit_trace_event("llm/error", dict(event))
            raise
        finally:
            event.setdefault("duration_seconds", round(time.perf_counter() - started, 4))
            self._trace_events.append(event)

    @staticmethod
    def _normalize_tool(tool: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(tool, dict):
            raise LLMClientError("Tool spec must be a dict.")
        if tool.get("type") != "function":
            return tool
        if "function" in tool and isinstance(tool["function"], dict):
            fn = tool["function"]
            return {
                "type": "function",
                "name": fn.get("name"),
                "description": fn.get("description"),
                "parameters": fn.get("parameters"),
                "strict": fn.get("strict"),
            }
        return {
            "type": "function",
            "name": tool.get("name"),
            "description": tool.get("description"),
            "parameters": tool.get("parameters"),
            "strict": tool.get("strict"),
        }

    def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.3,
        max_output_tokens: Optional[int] = None,
    ) -> str:
        client = self._client()
        response = self._responses_create(
            client=client,
            _trace_operation="chat",
            instructions=system,
            input=user,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        text = getattr(response, "output_text", None)
        if not text or not str(text).strip():
            raise LLMClientError("LLM returned empty content.")
        return str(text).strip()

    def call_function(
        self,
        *,
        system: str,
        user: str,
        function_name: str,
        description: str,
        parameters: Dict[str, Any],
        temperature: float = 0.2,
        max_output_tokens: Optional[int] = None,
        max_attempts: Optional[int] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """Force one strict function call and return its decoded arguments."""
        client = self._client()
        token_limit = max(int(max_output_tokens or 2048), int(os.environ.get("LLM_MAX_OUTPUT_TOKENS") or 0))
        attempts = max(1, int(max_attempts if max_attempts is not None else self.structured_attempts))
        last_error: Optional[Exception] = None
        for _ in range(attempts):
            response = self._responses_create(
                client=client,
                _trace_operation=f"function:{function_name}",
                instructions=system,
                input=user,
                tools=[{
                    "type": "function",
                    "name": function_name,
                    "description": description,
                    "strict": strict,
                    "parameters": parameters,
                }],
                tool_choice="required",
                temperature=temperature,
                max_output_tokens=token_limit,
            )
            calls = [item for item in getattr(response, "output", []) if getattr(item, "type", None) == "function_call"]
            matching = [call for call in calls if getattr(call, "name", None) == function_name]
            if len(matching) != 1:
                fallback = self._extract_json_object(getattr(response, "output_text", None))
                if fallback is not None:
                    return fallback
                last_error = LLMClientError(
                    f"Expected exactly one {function_name} tool call, received {len(matching)}."
                )
                continue
            try:
                arguments = json.loads(matching[0].arguments)
            except (json.JSONDecodeError, TypeError) as exc:
                last_error = LLMClientError(f"Tool arguments are not valid JSON: {exc}")
                continue
            if isinstance(arguments, dict):
                return arguments
            last_error = LLMClientError("Tool arguments must decode to a JSON object.")
        raise last_error or LLMClientError(f"Structured call {function_name} failed.")

    def run_function_tool_loop(
        self,
        *,
        system: str,
        user: str,
        tools: List[Dict[str, Any]],
        execute_tool: Callable[[str, Dict[str, Any]], Dict[str, Any]],
        terminal_tool_name: str,
        max_turns: int = 6,
        temperature: float = 0.1,
        max_output_tokens: Optional[int] = None,
        initial_tool_choice: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Run real function tools until the model calls the terminal report tool."""
        if max_turns < 1:
            raise ValueError("max_turns must be positive")

        client = self._client()
        configured_limit = int(os.environ.get("LLM_MAX_OUTPUT_TOKENS") or 0)
        token_limit = max(int(max_output_tokens or 1600), configured_limit)
        normalized_tools = [self._normalize_tool(item) for item in tools]
        tool_names = {
            str(item.get("name") or item.get("function", {}).get("name") or "")
            for item in normalized_tools
            if (item.get("type") == "function") or (item.get("function", {}).get("name") if isinstance(item, dict) else None)
        }
        if terminal_tool_name not in tool_names:
            raise ValueError(f"Terminal tool {terminal_tool_name} is not registered")

        trace: List[Dict[str, Any]] = []
        seen_signatures: Dict[str, int] = {}
        # Keep the loop portable across stateless/proxied Responses backends.
        # Some compatible gateways route consecutive requests to different
        # upstream resources, where previous_response_id cannot be resolved.
        conversation: List[Dict[str, Any]] = [{
            "role": "user",
            "content": [{"type": "input_text", "text": user}],
        }]

        response = self._responses_create(
            client=client,
            _trace_operation=f"tool_loop:{terminal_tool_name}:initial",
            instructions=system,
            input=conversation,
            tools=normalized_tools,
            tool_choice=initial_tool_choice or "required",
            temperature=temperature,
            max_output_tokens=token_limit,
        )

        for turn in range(max_turns):
            calls = [item for item in getattr(response, "output", []) if getattr(item, "type", None) == "function_call"]
            if len(calls) != 1:
                fallback_report = self._extract_json_object(getattr(response, "output_text", None))
                if fallback_report is not None and (
                    "execution_status" in fallback_report or "summary" in fallback_report
                ):
                    return {"report": fallback_report, "tool_trace": trace, "turns": turn + 1}
                raise LLMClientError(f"Expected exactly one tool call per executor turn, received {len(calls)}")

            call = calls[0]
            name = str(getattr(call, "name", "") or "")
            if name not in tool_names:
                raise LLMClientError(f"Model called unregistered tool: {name}")

            try:
                arguments = json.loads(call.arguments)
            except (json.JSONDecodeError, TypeError) as exc:
                raise LLMClientError(f"Tool arguments are not valid JSON: {exc}") from exc
            if not isinstance(arguments, dict):
                raise LLMClientError("Tool arguments must decode to a JSON object")

            call_id = getattr(call, "call_id", None) or getattr(call, "id", None)
            if name == terminal_tool_name:
                return {"report": arguments, "tool_trace": trace, "turns": turn + 1}

            conversation.append({
                "type": "function_call",
                "call_id": call_id,
                "name": name,
                "arguments": str(getattr(call, "arguments", "") or "{}"),
            })

            signature = json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False, sort_keys=True)
            seen_signatures[signature] = seen_signatures.get(signature, 0) + 1
            def dispatch() -> Dict[str, Any]:
                if seen_signatures[signature] > 1:
                    return {
                        "ok": False,
                        "error": "duplicate_tool_call",
                        "instruction": "Change the tool or arguments; do not repeat this call.",
                    }
                return execute_tool(name, arguments)

            result = self._tool_pipeline.execute(
                call_id=call_id,
                tool=name,
                arguments=arguments,
                handler=dispatch,
                turn=turn + 1,
                operation=f"tool_loop:{terminal_tool_name}",
            )

            trace.append({"turn": turn + 1, "tool_call_id": call_id, "tool_name": name, "arguments": arguments, "result": result})
            conversation.append({
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(result, ensure_ascii=False, default=str),
            })

            if turn >= max_turns - 1:
                fallback_report = self._extract_json_object(getattr(response, "output_text", None))
                if fallback_report is not None and (
                    "execution_status" in fallback_report or "summary" in fallback_report
                ):
                    return {"report": fallback_report, "tool_trace": trace, "turns": turn + 1}
                raise LLMClientError("Executor hit max_turns before terminal tool was called")

            next_tools = normalized_tools if turn < max_turns - 2 else [t for t in normalized_tools if (t.get("name") or t.get("function", {}).get("name")) == terminal_tool_name]
            if not next_tools:
                next_tools = [{"type": "function", "name": terminal_tool_name, "description": "Finish the task.", "parameters": {"type": "object", "properties": {}, "additionalProperties": True}, "strict": False}]

            response = self._responses_create(
                client=client,
                _trace_operation=f"tool_loop:{terminal_tool_name}:turn_{turn + 1}",
                instructions=system,
                input=conversation,
                tools=next_tools,
                tool_choice="required",
                temperature=temperature,
                max_output_tokens=token_limit,
            )

        raise LLMClientError("Executor did not produce terminal report within max_turns")
