"""A stand-in for the person's own AI endpoint, used by evals/endpoint_check.py. Stdlib only.

Run: python3 evals/mock_endpoint.py --port 8123 --mode tools
It answers the OpenAI API (GET /v1/models, POST /v1/chat/completions) the way llama.cpp, LM Studio or vLLM
do: streaming SSE deltas, and in tools mode a library tool call before it answers.

Modes:
  tools     calls search_library, then read_article with an id from the results, then answers (the usual case)
  no-tools  answers 400 to any request that asks for tools, like a server that cannot do tool calling
  flaky     answers the app's tool probe but turns down tool calling when it really has to do it
  no-calls  accepts tools but never uses them, and answers from whatever is in the question
  deny-key  401 for both /v1/models and /chat/completions unless the key is exactly right
  broken    answers 200 with something that is not an answer at all
  short     cuts the answer off at the length limit (finish_reason "length")
  slow      dribbles the answer out, so a test can cut it short with Stop

GET /_log shows every request it took (path, Authorization header, body), so the tests can check what the
app asked for and where the key went.
"""
import argparse
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODELS = [{"id": "mock-model", "object": "model"}, {"id": "other/mock:2", "object": "model"}]
ANSWER = ("Keep the fridge closed: the food in it stays safe for about 4 hours, longer if it was full. "
          "A closed full freezer holds about 48 hours. Discard perishable food above 40 F.")


def plan(messages):
    """What this mock does next: look something up, open what it found, or answer from what it read."""
    results = [m for m in messages if m.get("role") == "tool"]
    if not results:
        return {"name": "search_library", "arguments": {"query": "food fridge blackout how long safe"}}
    found = re.search(r"\[id (\d+)\]", results[-1].get("content") or "")
    if found and len(results) < 3:
        return {"name": "read_article", "arguments": {"id": int(found.group(1))}}
    return None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    mode, key, log = "tools", "", []

    def log_message(self, *args):
        pass

    def _json(self, code, payload):
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _refused(self, why):
        return self._json(401, {"error": {"message": why, "type": "invalid_request_error"}})

    def _key_ok(self):
        want = f"Bearer {self.key}" if self.key else ""
        return self.mode != "deny-key" or self.headers.get("Authorization", "") == want

    def do_GET(self):
        if self.path.split("?")[0] == "/v1/models":
            if not self._key_ok():
                return self._refused("invalid api key")
            return self._json(200, {"object": "list", "data": MODELS})
        if self.path.split("?")[0] == "/_log":  # for the tests: what the app actually asked for
            return self._json(200, {"requests": list(self.log)})
        return self._json(404, {"error": {"message": "no such path", "type": "invalid_request_error"}})

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        try:
            body = json.loads(raw or b"{}")
        except ValueError:
            return self._json(400, {"error": {"message": "not json", "type": "invalid_request_error"}})
        self.log.append({"path": self.path, "authorization": self.headers.get("Authorization", ""), "body": body})
        path = self.path.split("?")[0]
        if path != "/v1/chat/completions":
            return self._json(404, {"error": {"message": "no such path", "type": "invalid_request_error"}})
        if not self._key_ok():
            return self._refused("invalid api key")
        messages = body.get("messages") or []
        if body.get("tools") and self.mode == "no-tools":
            return self._json(400, {"error": {"message": "this server does not support tools",
                                              "type": "invalid_request_error"}})
        if body.get("tools") and self.mode == "flaky" and not body.get("tool_choice"):
            return self._json(400, {"error": {"message": "tools are not supported here",
                                              "type": "invalid_request_error"}})
        probe = bool(body.get("tool_choice"))
        asked_for_tools = bool(body.get("tools")) or probe
        next_call = plan(messages) if asked_for_tools and self.mode in ("tools", "flaky") else None
        if not body.get("stream"):  # the app's one non-streaming request: does it understand tools?
            if next_call:
                return self._json(200, {"choices": [{"message": {"role": "assistant", "tool_calls": [
                    {"id": "call_probe", "type": "function", "function": {"name": next_call["name"],
                     "arguments": json.dumps(next_call["arguments"])}}]}, "finish_reason": "tool_calls"}]})
            return self._json(200, {"choices": [{"message": {"role": "assistant", "content": "I can't."},
                                                 "finish_reason": "stop"}]})
        if self.mode == "broken":
            data = b"hello, this is not an answer\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            return self.wfile.write(data)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        for chunk in self._deltas(body, next_call):
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")

    def _deltas(self, body, next_call):
        if self.mode == "slow":  # dribble it out so a test can cut it short
            for part in re.findall(r".{1,28}", ANSWER):
                yield {"choices": [{"delta": {"content": part}, "finish_reason": None}]}
                time.sleep(0.4)
            yield {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"completion_tokens": 40}}
            return
        if next_call:  # a tool call, with its arguments split up the way a real server sends them
            args = json.dumps(next_call["arguments"])
            half = len(args) // 2
            yield {"choices": [{"delta": {"role": "assistant", "tool_calls": [
                {"index": 0, "id": "call_1", "type": "function", "function": {"name": next_call["name"],
                 "arguments": args[:half]}}]}, "finish_reason": None}]}
            yield {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": args[half:]}}]},
                               "finish_reason": None}]}
            yield {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}
            return
        first, rest = ANSWER[:60], ANSWER[60:]
        yield {"choices": [{"delta": {"role": "assistant", "content": first}, "finish_reason": None}]}
        if rest and self.mode != "short":
            yield {"choices": [{"delta": {"content": rest}, "finish_reason": None}]}
        reason = "length" if self.mode == "short" else "stop"
        yield {"choices": [{"delta": {}, "finish_reason": reason}], "usage": {"completion_tokens": 40}}


def serve(port, mode, key):
    """One mock in its own mode: each gets its own handler class, since they share the base one."""
    handler = type(f"Handler_{mode}", (Handler,), {"mode": mode, "key": key, "log": []})
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--mode", default="tools", choices=["tools", "no-tools", "flaky", "no-calls", "deny-key",
                                                            "broken", "short", "slow"])
    parser.add_argument("--key", default="")
    args = parser.parse_args()
    serve(args.port, args.mode, args.key)
    print(f"mock endpoint ({args.mode}) on http://127.0.0.1:{args.port}/v1", flush=True)
    threading.Event().wait()
