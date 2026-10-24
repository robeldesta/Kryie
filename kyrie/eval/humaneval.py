"""HumanEval-style functional correctness (pass@k).

    python -m kyrie.eval.humaneval runs/base --k 1 --n 1 --temperature 0     # greedy pass@1
    python -m kyrie.eval.humaneval runs/base --k 1,10 --n 20 --temperature 0.8

Problems come from the Hugging Face `openai/openai_humaneval` dataset, or any
JSONL file with the same fields (task_id, prompt, test, entry_point).

Generated code is executed in a separate Python process with a timeout. That
is enough isolation for our own model's output on a dev machine, but run it
in a container if you evaluate untrusted checkpoints.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

STOP = ["\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\n#", "\nassert "]


def load_problems(path: str = "") -> list[dict]:
    if path:
        with open(path) as fh:
            return [json.loads(l) for l in fh if l.strip()]
    from datasets import load_dataset

    return list(load_dataset("openai/openai_humaneval", split="test"))


def truncate(completion: str) -> str:
    cut = len(completion)
    for s in STOP:
        i = completion.find(s)
        if i != -1:
            cut = min(cut, i)
    return completion[:cut]


def check(problem: dict, completion: str, timeout: float = 5.0) -> bool:
    program = (problem["prompt"] + completion + "\n\n" + problem["test"] + "\n\n"
               + f"check({problem['entry_point']})\n")
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "prog.py")
        with open(f, "w") as fh:
            fh.write(program)
        try:
            r = subprocess.run([sys.executable, "-I", f], cwd=d, timeout=timeout,
                               capture_output=True, stdin=subprocess.DEVNULL)
            return r.returncode == 0
        except subprocess.TimeoutExpired:
            return False


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator from the Codex paper."""
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def run(model, tok, device, problems, n=1, ks=(1,), temperature=0.0, max_new_tokens=384,
        out_path: str = "", limit: int = 0):
    from ..sample import complete

    if limit:
        problems = problems[:limit]
    results = []
    for i, prob in enumerate(problems):
        correct = 0
        samples = []
        for _ in range(n):
            comp = truncate(complete(model, tok, device, prob["prompt"], max_new_tokens,
                                     temperature=temperature, top_k=0, top_p=0.95))
            ok = check(prob, comp)
            correct += ok
            samples.append({"completion": comp, "passed": ok})
        results.append({"task_id": prob["task_id"], "n": n, "correct": correct, "samples": samples})
        print(f"[{i + 1}/{len(problems)}] {prob['task_id']}: {correct}/{n}")

    scores = {f"pass@{k}": float(np.mean([pass_at_k(r["n"], r["correct"], k) for r in results]))
              for k in ks if k <= n}
    if out_path:
        with open(out_path, "w") as fh:
            json.dump({"scores": scores, "results": results}, fh, indent=2)
    return scores


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", help="run directory or checkpoint .pt")
    p.add_argument("--problems", default="", help="JSONL file; default downloads HumanEval")
    p.add_argument("--n", type=int, default=1, help="samples per problem")
    p.add_argument("--k", default="1")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_new_tokens", type=int, default=384)
    p.add_argument("--limit", type=int, default=0, help="only the first N problems")
    p.add_argument("--out", default="", help="write per-sample results JSON here")
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    from ..checkpoint import load_model

    model, tok, dev = load_model(args.run, args.device)
    ks = tuple(int(k) for k in args.k.split(","))
    scores = run(model, tok, dev, load_problems(args.problems), args.n, ks, args.temperature,
                 args.max_new_tokens, args.out, args.limit)
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
