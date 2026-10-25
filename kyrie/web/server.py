"""Local web interface for Kyrie: playground, chat preview, training dashboard, tokenizer.

    python -m kyrie.web                 # then open http://127.0.0.1:7860
    python -m kyrie.web --port 8000

Run it from the repo folder; it reads runs/, data/ and tokenizer/ from there.
It listens on 127.0.0.1 only, because it can start training jobs.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import torch

from ..checkpoint import load_model
from ..config import PRESETS
from ..device import pick_device
from ..tokenizer import EOT, load as load_tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


def _read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as fh:
        for line in fh:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # a line still being written
    return out


def _tail(path: str, n: int = 40) -> list[str]:
    if not os.path.exists(path):
        return []
    with open(path, "rb") as fh:
        fh.seek(0, os.SEEK_END)
        fh.seek(max(0, fh.tell() - 16_000))
        return fh.read().decode("utf-8", "replace").splitlines()[-n:]


class Models:
    """Loads checkpoints on demand and reloads them when the file changes."""

    def __init__(self, root: str):
        self.root = root
        self.cache: dict[str, tuple[float, object, object, torch.device]] = {}
        self.lock = threading.Lock()

    def get(self, run_dir: str):
        best = os.path.join(run_dir, "best.pt")
        path = best if os.path.exists(best) else os.path.join(run_dir, "ckpt.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"no checkpoint in {run_dir}")
        mtime = os.path.getmtime(path)
        hit = self.cache.get(path)
        if hit and hit[0] == mtime:
            return hit[1], hit[2], hit[3]
        model, tok, dev = load_model(path, tokenizer=self._tokenizer(run_dir))
        self.cache[path] = (mtime, model, tok, dev)
        return model, tok, dev

    def _tokenizer(self, run_dir: str) -> str:
        """The run's tokenizer, resolving a relative data_dir against the repo root."""
        try:
            with open(os.path.join(run_dir, "config.json")) as fh:
                data_dir = json.load(fh)["train"]["data_dir"]
            with open(os.path.join(self.root, data_dir, "meta.json")) as fh:
                tok = json.load(fh)["tokenizer"]
        except (OSError, KeyError, ValueError):
            tok = "tokenizer/kyrie.json"
        return os.path.join(self.root, tok)  # no-op when tok is already absolute


class TrainingJob:
    """At most one training subprocess, started and stopped from the UI."""

    def __init__(self, root: str):
        self.root = root
        self.proc: subprocess.Popen | None = None
        self.run_name = ""
        self.log_path = ""
        self.started = 0.0

    def status(self) -> dict:
        running = self.proc is not None and self.proc.poll() is None
        return {
            "running": running,
            "run_name": self.run_name,
            "pid": self.proc.pid if self.proc else None,
            "returncode": None if running or not self.proc else self.proc.returncode,
            "started": self.started,
            "log_tail": _tail(self.log_path) if self.log_path else [],
        }

    def start(self, preset: str, run_name: str, max_steps: int, data_dir: str) -> dict:
        if self.proc is not None and self.proc.poll() is None:
            raise ValueError(f"a training run ({self.run_name}) is already going; stop it first")
        if preset not in PRESETS:
            raise ValueError(f"unknown preset {preset!r}")
        if not RUN_NAME.match(run_name):
            raise ValueError("run name: letters, digits, '.', '_' or '-' only")
        if not 1 <= max_steps <= 10_000_000:
            raise ValueError("max_steps out of range")
        data_abs = os.path.abspath(os.path.join(self.root, data_dir))
        if not data_abs.startswith(os.path.abspath(self.root) + os.sep) or not os.path.isdir(data_abs):
            raise ValueError(f"data dir {data_dir!r} not found inside the repo")

        os.makedirs(os.path.join(self.root, "runs"), exist_ok=True)
        self.log_path = os.path.join(self.root, "runs", f"{run_name}.log")
        cmd = [sys.executable, "-m", "kyrie.train", "--preset", preset, "--data_dir", data_dir,
               "--out_dir", f"runs/{run_name}", "--max_steps", str(max_steps)]
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        with open(self.log_path, "a") as log:
            self.proc = subprocess.Popen(cmd, cwd=self.root, stdout=log, stderr=subprocess.STDOUT, env=env)
        self.run_name, self.started = run_name, time.time()
        return self.status()

    def stop(self) -> dict:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()  # the last checkpoint is kept; the run resumes from it
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        return self.status()


class App:
    def __init__(self, root: str):
        self.root = root
        self.models = Models(root)
        self.job = TrainingJob(root)
        self.gen_lock = threading.Lock()
        self.tokenizers: dict[str, object] = {}

    # ---- read-only views -------------------------------------------------
    def info(self) -> dict:
        datasets = []
        for meta in sorted(glob.glob(os.path.join(self.root, "data", "*", "meta.json"))):
            m = json.load(open(meta))
            datasets.append({"dir": os.path.relpath(os.path.dirname(meta), self.root),
                             "train_tokens": m.get("train_tokens"), "vocab_size": m.get("vocab_size")})
        return {
            "device": str(pick_device()),
            "torch": torch.__version__,
            "presets": {k: {"d_model": v.model.d_model, "n_layers": v.model.n_layers,
                            "seq_len": v.model.seq_len, "max_steps": v.train.max_steps}
                        for k, v in PRESETS.items()},
            "datasets": datasets,
            "tokenizers": [os.path.relpath(p, self.root)
                           for p in sorted(glob.glob(os.path.join(self.root, "tokenizer", "*.json")))],
        }

    def runs(self) -> list[dict]:
        out = []
        for d in sorted(glob.glob(os.path.join(self.root, "runs", "*"))):
            if not os.path.isdir(d):
                continue
            cfg_path = os.path.join(d, "config.json")
            has_ckpt = any(os.path.exists(os.path.join(d, f)) for f in ("best.pt", "ckpt.pt"))
            if not (has_ckpt or os.path.exists(cfg_path)):
                continue
            cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
            log = _read_jsonl(os.path.join(d, "log.jsonl"))
            train = [r for r in log if "loss" in r]
            val = [r for r in log if "val_loss" in r]
            out.append({
                "name": os.path.basename(d),
                "has_checkpoint": has_ckpt,
                "model": cfg.get("model", {}),
                "max_steps": cfg.get("train", {}).get("max_steps"),
                "step": train[-1]["step"] if train else 0,
                "loss": train[-1]["loss"] if train else None,
                "best_val": min((r["val_loss"] for r in val), default=None),
                "updated": os.path.getmtime(d),
            })
        return sorted(out, key=lambda r: -r["updated"])

    def log(self, name: str) -> list[dict]:
        if not RUN_NAME.match(name):
            raise ValueError("bad run name")
        return _read_jsonl(os.path.join(self.root, "runs", name, "log.jsonl"))

    def tokenize(self, text: str, tokenizer: str) -> dict:
        path = os.path.abspath(os.path.join(self.root, tokenizer))
        if not path.startswith(os.path.abspath(self.root) + os.sep) or not os.path.exists(path):
            raise ValueError(f"tokenizer {tokenizer!r} not found")
        tok = self.tokenizers.get(path) or load_tokenizer(path)
        self.tokenizers[path] = tok
        enc = tok.encode(text)
        # Show each token as the slice of text it covers. A Ge'ez character is 3 bytes
        # and can span several byte-level tokens; the extra ones are marked "cont".
        pieces, prev = [], None
        for i, span in zip(enc.ids, enc.offsets):
            cont = span == prev and span[1] > span[0]
            pieces.append({"id": i, "text": "" if cont else text[span[0]:span[1]], "cont": cont})
            prev = span
        return {"n_chars": len(text), "n_tokens": len(enc.ids), "pieces": pieces}

    # ---- generation --------------------------------------------------------
    def stream(self, req: dict):
        """Check the request and load the model now, so errors become a 400 before
        streaming starts; return a generator of text pieces."""
        name = str(req.get("run", ""))
        if not RUN_NAME.match(name):
            raise ValueError("pick a run")
        model, tok, dev = self.models.get(os.path.join(self.root, "runs", name))
        max_new = max(1, min(int(req.get("max_new_tokens", 200)), model.cfg.seq_len - 1))
        temperature = float(req.get("temperature", 0.8))
        top_k = int(req.get("top_k", 50))
        top_p = float(req.get("top_p", 0.95))
        stops = {tok.token_to_id(EOT)}

        if req.get("mode") == "chat":
            parts = []
            for m in req.get("messages", [])[-8:]:
                role = "<|assistant|>" if m.get("role") == "assistant" else "<|user|>"
                parts.append(f"{role}{m.get('content', '')}<|end|>")
            prompt = "".join(parts) + "<|assistant|>"
            end = tok.token_to_id("<|end|>")
            if end is not None:
                stops.add(end)
        else:
            prompt = str(req.get("prompt", ""))
        ids = tok.encode(prompt).ids or [tok.token_to_id(EOT)]
        idx = torch.tensor([ids], dtype=torch.long, device=dev)
        return self._pieces(model, tok, idx, max_new, temperature, top_k, top_p, stops)

    def _pieces(self, model, tok, idx, max_new, temperature, top_k, top_p, stops):
        out: list[int] = []
        sent = ""
        with self.gen_lock:
            for t in model.generate_stream(idx, max_new, temperature, top_k, top_p, stop_ids=stops):
                if t in stops:
                    break
                out.append(t)
                text = tok.decode(out)
                # Hold back a half-finished multi-byte character (Ge'ez is 3 bytes).
                if text.endswith("�"):
                    continue
                if len(text) > len(sent):
                    yield text[len(sent):]
                    sent = text


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        server_version = "Kyrie"

        def log_message(self, fmt, *args):  # keep the terminal quiet
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}") if n else {}

        def do_GET(self):
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if url.path in ("/", "/index.html"):
                    body = open(os.path.join(HERE, "index.html"), "rb").read()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif url.path == "/api/info":
                    self._json(app.info())
                elif url.path == "/api/runs":
                    self._json(app.runs())
                elif url.path.startswith("/api/runs/") and url.path.endswith("/log"):
                    self._json(app.log(url.path.split("/")[3]))
                elif url.path == "/api/tokenize":
                    self._json(app.tokenize(q.get("text", ""), q.get("tokenizer", "tokenizer/kyrie.json")))
                elif url.path == "/api/train":
                    self._json(app.job.status())
                else:
                    self._json({"error": "not found"}, 404)
            except (ValueError, FileNotFoundError) as e:
                self._json({"error": str(e)}, 400)

        def do_POST(self):
            url = urlparse(self.path)
            try:
                req = self._body()
                if url.path == "/api/generate":
                    pieces = app.stream(req)
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; charset=utf-8")
                    # HTTP/1.0 with no Content-Length: the body streams until we close.
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    try:
                        for piece in pieces:
                            self.wfile.write(piece.encode())
                            self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        pass  # the browser pressed Stop
                elif url.path == "/api/train":
                    self._json(app.job.start(str(req.get("preset", "tiny")), str(req.get("run_name", "")),
                                             int(req.get("max_steps", 1000)),
                                             str(req.get("data_dir", "data/shards"))))
                elif url.path == "/api/train/stop":
                    self._json(app.job.stop())
                else:
                    self._json({"error": "not found"}, 404)
            except (ValueError, FileNotFoundError, json.JSONDecodeError) as e:
                self._json({"error": str(e)}, 400)

    return Handler


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--root", default=".", help="repo folder with runs/, data/, tokenizer/")
    args = p.parse_args(argv)
    app = App(os.path.abspath(args.root))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(app))
    print(f"Kyrie UI on http://127.0.0.1:{args.port}  (Ctrl+C to quit)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.job.stop()


if __name__ == "__main__":
    main()
