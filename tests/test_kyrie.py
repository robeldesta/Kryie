import json
import os
import textwrap

import numpy as np
import torch

from kyrie.config import ModelConfig, PRESETS, parse_config
from kyrie.eval.humaneval import check, load_problems, pass_at_k, truncate
from kyrie.model import GPT
from kyrie.sources import ExactDedup, keep_code

HERE = os.path.dirname(__file__)


def tiny_cfg(**kw):
    base = dict(vocab_size=97, seq_len=32, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2)
    base.update(kw)
    return ModelConfig(**base)


def test_forward_shapes_and_loss():
    m = GPT(tiny_cfg())
    x = torch.randint(0, 97, (3, 16))
    y = torch.randint(0, 97, (3, 16))
    logits, loss = m(x, y)
    assert logits.shape == (3, 16, 97)
    # Untrained loss should be close to log(vocab).
    assert abs(loss.item() - np.log(97)) < 0.5


def test_tied_embeddings():
    m = GPT(tiny_cfg())
    assert m.lm_head.weight is m.embed.weight


def test_kv_cache_matches_full_forward():
    torch.manual_seed(0)
    m = GPT(tiny_cfg()).eval()
    x = torch.randint(0, 97, (1, 10))
    full, _ = m(x, x)  # logits for every position
    caches = [{} for _ in m.blocks]
    step_logits, _ = m(x[:, :6], kv_caches=caches, start_pos=0)
    assert torch.allclose(step_logits[0, -1], full[0, 5], atol=1e-5)
    for t in range(6, 10):
        step_logits, _ = m(x[:, t : t + 1], kv_caches=caches, start_pos=t)
        assert torch.allclose(step_logits[0, -1], full[0, t], atol=1e-5)


def test_overfits_one_batch():
    torch.manual_seed(0)
    m = GPT(tiny_cfg())
    opt = m.configure_optimizer(0.0, 3e-3, (0.9, 0.95), "cpu")
    x = torch.randint(0, 97, (4, 32))
    for _ in range(150):
        _, loss = m(x[:, :-1], x[:, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.5


def test_generate_respects_length():
    m = GPT(tiny_cfg())
    out = m.generate(torch.zeros(1, 5, dtype=torch.long), max_new_tokens=8, temperature=1.0)
    assert len(out) == 8


def test_presets_and_overrides():
    cfg = parse_config(["--preset", "small", "--lr", "1e-3", "--compile", "true"])
    assert cfg.train.lr == 1e-3 and cfg.train.compile is True
    assert PRESETS["small"].train.lr != 1e-3  # presets are not mutated
    base = GPT(PRESETS["base"].model)
    assert 70e6 < base.num_params() < 80e6  # ~75M non-embedding, ~101M total


def test_code_filter_and_dedup():
    good = textwrap.dedent("""
        def area(r):
            return 3.14159 * r * r
    """)
    assert keep_code(good)
    assert not keep_code("x" * 5000)  # one huge line
    assert not keep_code("# auto-generated, do not edit\n" + good)
    d = ExactDedup()
    assert d(good) and not d(good.replace("    ", "  "))


def test_humaneval_harness():
    probs = load_problems(os.path.join(HERE, "fixtures_humaneval.jsonl"))
    assert check(probs[0], "    return a + b\n")
    assert not check(probs[0], "    return a - b\n")
    assert not check(probs[1], "    while True: pass\n")  # times out
    assert truncate("    return 1\ndef other():\n    pass") == "    return 1"
    assert pass_at_k(10, 0, 1) == 0.0 and pass_at_k(10, 10, 1) == 1.0
    assert abs(pass_at_k(10, 5, 1) - 0.5) < 1e-9
