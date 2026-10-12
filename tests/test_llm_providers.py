"""Ollama and OpenAI-compatible wiring, end to end, against a tiny fake HTTP server.

The fake server speaks just enough of each API to return one tool call and then
a final answer. No model, no internet: it runs on 127.0.0.1.
"""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

pytest.importorskip("langchain_ollama")
pytest.importorskip("langchain_openai")

from peakwhale_helm.assistant import Assistant  # noqa: E402
from peakwhale_helm.config import LLMSettings  # noqa: E402
from peakwhale_helm.llm import build_chat_model  # noqa: E402
from peakwhale_helm.toolbox import Toolbox  # noqa: E402
from peakwhale_helm.toolset import LocalToolSet  # noqa: E402

SQL = "SELECT COUNT(*) AS n FROM customers"
FINAL = "There are 100 customers [S1]."


def _has_tool_result(messages) -> bool:
    return any(m.get("role") == "tool" for m in messages)


class FakeLLMHandler(BaseHTTPRequestHandler):
    requests: list = []

    def log_message(self, *args):  # keep test output clean
        pass

    def _send(self, body: bytes, content_type: str = "application/json"):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/tags":
            self._send(json.dumps({"models": [{"name": "fake-model:latest"}]}).encode())
        else:
            self.send_error(404)

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeLLMHandler.requests.append((self.path, payload))
        done = _has_tool_result(payload.get("messages", []))
        tool_names = [t["function"]["name"] for t in payload.get("tools", [])]
        assert "query" in tool_names and "search_documents" in tool_names

        if self.path == "/api/chat":  # Ollama, streamed as JSON lines
            message = {"role": "assistant", "content": FINAL if done else ""}
            if not done:
                message["tool_calls"] = [{"function": {"name": "query", "arguments": {"sql": SQL}}}]
            line = {
                "model": "fake-model",
                "created_at": "2026-10-12T00:00:00Z",
                "message": message,
                "done": True,
                "done_reason": "stop",
            }
            self._send((json.dumps(line) + "\n").encode(), "application/x-ndjson")
        elif self.path == "/v1/chat/completions":  # OpenAI-compatible
            message = {"role": "assistant", "content": FINAL if done else None}
            if not done:
                message["tool_calls"] = [
                    {"id": "call_1", "type": "function", "function": {"name": "query", "arguments": json.dumps({"sql": SQL})}}
                ]
            body = {
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 0,
                "model": "fake-model",
                "choices": [{"index": 0, "message": message, "finish_reason": "stop" if done else "tool_calls"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
            self._send(json.dumps(body).encode())
        else:
            self.send_error(404)


@pytest.fixture()
def fake_server(monkeypatch):
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    FakeLLMHandler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeLLMHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


@pytest.mark.parametrize("provider", ["ollama", "openai"])
def test_provider_round_trip(provider, fake_server, demo_settings):
    base_url = fake_server if provider == "ollama" else fake_server + "/v1"
    chat = build_chat_model(LLMSettings(provider=provider, model="fake-model", base_url=base_url, api_key="test"))
    assert chat.model is not None, chat.problem

    assistant = Assistant(LocalToolSet(Toolbox(demo_settings)), demo_settings, llm=chat.model, llm_label=chat.label)
    answer, sources = asyncio.run(assistant.answer("How many customers do we have?"))

    assert [s.tool for s in sources] == ["query"]
    assert sources[0].args == {"sql": SQL} and "| 100 |" in sources[0].output
    assert answer.startswith(FINAL)
    assert "- [S1] query `SELECT COUNT(*) AS n FROM customers` -> 1 row" in answer
    # the second request carried the tagged tool result back to the model
    second = FakeLLMHandler.requests[-1][1]["messages"]
    assert any(m.get("role") == "tool" and str(m.get("content", "")).startswith("[S1]") for m in second)
