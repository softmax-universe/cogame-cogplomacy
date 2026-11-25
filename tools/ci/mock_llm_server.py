#!/usr/bin/env python3
"""Deterministic local Anthropic /v1/messages fixture for Cogplomacy CI & local certification.

Speaks the exact Cogplomacy prompt/response contract without any external network or paid LLM call.
Writes the bound port to --port-file and logs every served call to --log-file if provided.
"""

import argparse
import json
import sys
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def choose_reply(user: str) -> str:
    if "— PRESS." in user:
        return json.dumps({"broadcast": "Steady growth in Europe.", "notes": "press-turn"})
    if "— ORDERS." in user:
        orders = []
        lines = user.splitlines()
        idx = 0
        while idx < len(lines):
            line = lines[idx]
            if line.startswith(("A ", "F ")) and " in " in line and line.endswith(":"):
                # Pick first non-hold move if available, else hold.
                chosen = None
                j = idx + 1
                while j < len(lines) and lines[j].startswith("  "):
                    candidate = lines[j].strip()
                    if chosen is None:
                        chosen = candidate
                    elif " - " in candidate and " S " not in candidate and " C " not in candidate:
                        chosen = candidate
                        break
                    j += 1
                if chosen:
                    orders.append(chosen)
            idx += 1
        if not orders:
            raise ValueError(f"no legal orders found in prompt: {user[:300]}")
        # Exercise standard outer Markdown JSON fence on orders.
        return "```json\n" + json.dumps({"orders": orders, "notes": "orders-turn"}) + "\n```"
    if "— RETREATS." in user:
        retreats = []
        for line in user.splitlines():
            if " was dislodged from " in line and "Legal: " in line:
                legal = line.split("Legal: ", 1)[1].split(" (disband).", 1)[0]
                retreats.append(legal.split(", ")[0].strip())
        return json.dumps({"retreats": retreats})
    if "— BUILDS." in user:
        if "disband " in user and "Legal disbands: " in user:
            count = int(user.split("disband ", 1)[1].split(".", 1)[0])
            disbands = [x.strip() for x in user.split("Legal disbands: ", 1)[1].splitlines()[0].split(", ")]
            return json.dumps({"adjustments": disbands[:count]})
        if "Legal builds: " in user:
            builds_text = user.split("Legal builds: ", 1)[1].splitlines()[0].strip()
            if builds_text.startswith("("):
                return json.dumps({"adjustments": []})
            count = int(user.split("build ", 1)[1].split(".", 1)[0])
            builds = [x.strip() for x in builds_text.split(", ")]
            picked = []
            seen_provs = set()
            for b in builds:
                parts = b.split()
                if len(parts) == 3:
                    prov = parts[2].split("/")[0]
                    if prov not in seen_provs:
                        seen_provs.add(prov)
                        picked.append(b)
                        if len(picked) >= count:
                            break
            return json.dumps({"adjustments": picked})
        return json.dumps({"adjustments": []})
    raise ValueError(f"unrecognized phase in prompt: {user[:200]}")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path.startswith("/healthz"):
            body = b'{"ok":true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        body = json.loads(raw)
        slot = self.headers.get("X-Coworld-Player-Slot", "")
        user = body["messages"][0]["content"]
        reply = choose_reply(user)
        if self.server.log_file:
            with open(self.server.log_file, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"slot": slot, "model": body.get("model"), "reply": reply}) + "\n")
        resp = json.dumps(
            {
                "model": body.get("model", "local-deterministic-fixture"),
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": reply}],
                "usage": {"input_tokens": 100, "output_tokens": 20},
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.send_header("X-Softmax-Llm-Call-Id", str(uuid.uuid5(uuid.NAMESPACE_URL, raw.decode("utf-8"))))
        self.end_headers()
        self.wfile.write(resp)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--port-file", default="")
    parser.add_argument("--log-file", default="")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.log_file = args.log_file
    if args.port_file:
        Path(args.port_file).write_text(str(server.server_port), encoding="utf-8")
    print(f"mock_llm_server listening on {args.host}:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
