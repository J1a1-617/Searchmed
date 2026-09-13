import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from searchagent_retrieval.llm_client import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL_NAME,
    LLMClient,
    LLMClientError,
)


class LLMClientConfigurationTests(unittest.TestCase):
    def test_defaults_to_requested_base_url_and_gpt5(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            client = LLMClient()
        self.assertEqual(client.base_url, DEFAULT_BASE_URL)
        self.assertEqual(client.model_name, DEFAULT_MODEL_NAME)

    def test_accepts_default_alias_environment_names(self) -> None:
        with patch.dict(os.environ, {
            "DEFAULT_OPENAI_API_KEY": "alias-key",
            "DEFAULT_BASE_URL": "https://example.test/v1",
            "MODEL_NAME": "gpt-5",
        }, clear=True):
            client = LLMClient()
        self.assertEqual(client.api_key, "alias-key")
        self.assertEqual(client.base_url, "https://example.test/v1")

    def test_explicit_fast_fail_configuration_accepts_zero_retries(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            client = LLMClient(timeout=30, max_retries=0, structured_attempts=2)
        self.assertEqual(client.timeout, 30)
        self.assertEqual(client.max_retries, 0)
        self.assertEqual(client.structured_attempts, 2)

    def test_connect_timeout_is_shorter_than_long_generation_timeout(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            client = LLMClient(timeout=900)
        self.assertEqual(client.timeout, 900)
        self.assertEqual(client.connect_timeout, 20)

    def test_connection_failure_is_not_retried_inside_one_logical_call(self) -> None:
        class ConnectTimeout(RuntimeError):
            pass

        class Responses:
            def __init__(self):
                self.calls = 0

            def create(self, **request):
                self.calls += 1
                raise ConnectTimeout("connection timeout")

        responses = Responses()
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            client = LLMClient(max_retries=5)
        with self.assertRaises(ConnectTimeout):
            client._responses_create(client=SimpleNamespace(responses=responses), input="x")
        self.assertEqual(responses.calls, 1)

    def test_trace_call_budget_reports_remaining_calls(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            client = LLMClient(max_calls_per_trace=40)
        client._trace_events = [{}, {}, {}]
        self.assertEqual(client.calls_used(), 3)
        self.assertEqual(client.calls_remaining(), 37)

    def test_stage_call_budget_is_a_hard_limit(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            client = LLMClient(
                max_calls_per_trace=40,
                stage_call_budgets={
                    "planning": 2,
                    "retrieval": 3,
                    "finalization": 2,
                },
            )
        client._trace_events = [
            {"stage": "planning"},
            {"stage": "planning"},
            {"stage": "retrieval"},
        ]

        with self.assertRaisesRegex(LLMClientError, "planning"):
            client._responses_create(
                client=SimpleNamespace(),
                _trace_operation="function:submit_retrieval_plan",
                input="x",
            )

        self.assertEqual(client.stage_calls_used("planning"), 2)
        self.assertEqual(client.stage_calls_used("retrieval"), 1)

    def test_function_tool_loop_executes_and_returns_tool_result_to_model(self) -> None:
        requests = []
        emitted = []

        calls = [
            SimpleNamespace(type="function_call", id="item-1", call_id="call-1", name="echo_tool", arguments='{"value":"x"}'),
            SimpleNamespace(type="function_call", id="item-2", call_id="call-2", name="submit_report", arguments='{"status":"done"}'),
        ]

        class Responses:
            def create(self, **request):
                requests.append(request)
                return SimpleNamespace(
                    id=f"resp-{len(requests)}",
                    status="completed",
                    output=[calls[len(requests) - 1]],
                    output_text="",
                    usage=None,
                )

        fake_openai = SimpleNamespace(responses=Responses())
        tools = [
            {"type": "function", "function": {"name": "echo_tool", "strict": True, "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "submit_report", "strict": True, "parameters": {"type": "object"}}},
        ]
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            client = LLMClient()
        client.begin_trace("case-tools", event_sink=lambda event_type, data: emitted.append((event_type, data)))
        with patch("openai.OpenAI", return_value=fake_openai):
            result = client.run_function_tool_loop(
                system="s", user="u", tools=tools,
                execute_tool=lambda name, arguments: {"ok": True, "echo": arguments["value"]},
                terminal_tool_name="submit_report", max_turns=3,
            )

        self.assertEqual(result["report"], {"status": "done"})
        self.assertEqual(result["tool_trace"][0]["tool_name"], "echo_tool")
        conversation = requests[1]["input"]
        self.assertNotIn("previous_response_id", requests[1])
        self.assertEqual(conversation[0]["role"], "user")
        self.assertEqual(conversation[1]["type"], "function_call")
        self.assertEqual(conversation[1]["call_id"], "call-1")
        self.assertEqual(conversation[2]["type"], "function_call_output")
        self.assertIn('"echo": "x"', conversation[2]["output"])
        tool_events = [(event_type, data) for event_type, data in emitted if event_type.startswith("tool/")]
        self.assertEqual([event_type for event_type, _ in tool_events], ["tool/call", "tool/result"])
        self.assertEqual(tool_events[0][1]["call_id"], "call-1")
        self.assertEqual(tool_events[1][1]["call_id"], "call-1")
        self.assertEqual(tool_events[1][1]["status"], "success")
        self.assertEqual(tool_events[1][1]["result"], {"ok": True, "echo": "x"})

    def test_call_function_falls_back_to_json_content_when_tool_calls_are_missing(self) -> None:
        class Responses:
            def create(self, **request):
                return SimpleNamespace(
                    id="resp-1",
                    status="completed",
                    output=[],
                    output_text='{"status":"ok","value":3}',
                    usage=None,
                )

        fake_openai = SimpleNamespace(responses=Responses())
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            client = LLMClient()
        with patch("openai.OpenAI", return_value=fake_openai):
            result = client.call_function(
                system="s",
                user="u",
                function_name="submit_report",
                description="desc",
                parameters={"type": "object"},
            )

        self.assertEqual(result, {"status": "ok", "value": 3})

    def test_invalid_function_schema_disguised_as_429_is_not_retried(self) -> None:
        class Responses:
            def __init__(self):
                self.calls = 0

            def create(self, **request):
                self.calls += 1
                raise RuntimeError(
                    "429 RateLimitError: Invalid schema for function 'submit_reranking'; "
                    "code: invalid_function_parameters"
                )

        responses = Responses()
        fake_openai = SimpleNamespace(responses=responses)
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            client = LLMClient(max_retries=5)

        with self.assertRaisesRegex(RuntimeError, "invalid_function_parameters"):
            client._responses_create(client=fake_openai, input="x")

        self.assertEqual(responses.calls, 1)

    def test_transient_error_still_uses_bounded_retry(self) -> None:
        class Responses:
            def __init__(self):
                self.calls = 0

            def create(self, **request):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("temporary 429 capacity limit")
                return SimpleNamespace(id="r", status="completed", output=[], usage=None)

        responses = Responses()
        fake_openai = SimpleNamespace(responses=responses)
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            client = LLMClient(max_retries=1)
        with patch("searchagent_retrieval.llm_client.time.sleep"):
            response = client._responses_create(client=fake_openai, input="x")
        self.assertEqual(response.id, "r")
        self.assertEqual(responses.calls, 2)

    def test_event_sink_receives_request_retry_and_response(self) -> None:
        class Responses:
            def __init__(self):
                self.calls = 0

            def create(self, **request):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("temporary 429 capacity limit")
                return SimpleNamespace(
                    id="resp-1", status="completed", output=[], output_text="ok", usage=None,
                )

        emitted = []
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            client = LLMClient(max_retries=1)
        client.begin_trace("case-1", event_sink=lambda event_type, data: emitted.append((event_type, data)))
        with patch.dict(os.environ, {"LLM_TRACE_FULL": "1"}), patch("searchagent_retrieval.llm_client.time.sleep"):
            client._responses_create(client=SimpleNamespace(responses=Responses()), input="full input")

        self.assertEqual([event_type for event_type, _ in emitted], ["llm/request", "llm/retry", "llm/response"])
        self.assertEqual(emitted[0][1]["request"]["input"], "full input")
        self.assertEqual(emitted[2][1]["response_output_text"], "ok")
        self.assertEqual(emitted[0][1]["call_id"], emitted[2][1]["call_id"])


if __name__ == "__main__":
    unittest.main()
