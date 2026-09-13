from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional


class ToolExecutionPipeline:
    """One model-facing tool call in, exactly one normalized result out.

    Domain operations may emit any number of ``tool/stage`` events while the
    handler is active. The pipeline owns the durable call/result boundary so
    individual tools cannot accidentally omit the terminal event.
    """

    def __init__(self, event_sink: Callable[[str, Dict[str, Any]], None]) -> None:
        self._event_sink = event_sink
        self._active_call_id: Optional[str] = None

    @property
    def active_call_id(self) -> Optional[str]:
        return self._active_call_id

    def stage(
        self,
        stage: str,
        data: Dict[str, Any],
        *,
        duration_ms: Optional[float] = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "call_id": self._active_call_id,
            "stage": str(stage),
            **data,
        }
        if duration_ms is not None:
            payload["duration_ms"] = round(float(duration_ms), 3)
        self._event_sink("tool/stage", payload)

    def execute(
        self,
        *,
        call_id: Optional[str],
        tool: str,
        arguments: Dict[str, Any],
        handler: Callable[[], Any],
        turn: int,
        operation: str,
    ) -> Dict[str, Any]:
        normalized_call_id = str(call_id or "")
        self._event_sink("tool/call", {
            "call_id": normalized_call_id,
            "tool": tool,
            "arguments": arguments,
            "turn": turn,
            "operation": operation,
        })
        started = time.perf_counter()
        self._active_call_id = normalized_call_id or None
        try:
            try:
                result = handler()
                if not isinstance(result, dict):
                    result = {"ok": True, "result": result}
            except Exception as exc:
                result = {
                    "ok": False,
                    "error": type(exc).__name__,
                    "detail": str(exc)[:500],
                }
        finally:
            self._active_call_id = None

        status = "success" if result.get("ok", True) else "error"
        if result.get("error") == "duplicate_tool_call":
            status = "rejected"
        self._event_sink("tool/result", {
            "call_id": normalized_call_id,
            "tool": tool,
            "status": status,
            "result": result,
            "duration_ms": round((time.perf_counter() - started) * 1000, 3),
            "turn": turn,
            "operation": operation,
        })
        return result
