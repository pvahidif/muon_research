"""Verifies muon_research.scripts.run_curv_profile keeps pool A (whatever
batch the caller's forward+backward populates ``p.grad`` from, hence what
``profile_source="signal"``/``"grad"`` read ``D_i``/``sigma_i`` off -- see
``muon_research/curv.py``'s own pool A/B docstring) genuinely independent
of pool B (what gamma/phi are estimated from, ``pool_b_batches``) whenever
``ProfileConfig.profile_batch_size`` is set.

This is a real, previously-shipped bug: the pre-fix code drew ONE batch
(the resumed train cursor by default, or -- once ``profile_batch_size`` was
set -- a held-out val batch instead) and used that SAME batch both to
populate ``p.grad`` (pool A, for "signal"/"grad") AND as ``pool_b_batches``.
So setting ``profile_batch_size``, whose entire documented point is to give
pool B "a genuinely independent sample from pool A", instead silently
redirected pool A onto that same batch too -- for "grad" pool A and pool B
ended up IDENTICAL; for "signal" pool A ended up partly (via the Nesterov/
momentum blend's own current-gradient term) built from pool B's own data.
``experiments/exp003_curvature`` (behind ``article.ipynb``'s section 4) hit
this directly: it uses ``profile_source="signal"`` with ``profile_batch_size``
set specifically for pool B independence.

Two layers of coverage:

1. ``_load_pool_batches`` in isolation (no model needed) -- tiny synthetic
   data shards with disjoint, individually-identifiable train/val token
   ranges, so pool A/B (in)dependence can be checked by exact token-value
   (dis)equality rather than indirectly.
2. ``run_checkpoint_profile`` end to end, with a real (tiny) checkpoint --
   confirms the actual wiring (not just the extracted helper) keeps
   ``sigma`` (pool A) identical regardless of ``profile_batch_size``, for
   both "grad" (pool A IS ``p.grad``) and "signal" (pool A blends real
   prior momentum with the current gradient) -- and, as the complementary
   check, that ``gamma_diag`` (pool B) DOES change when ``profile_batch_size``
   gives it genuinely different data, proving pool B is actually wired
   through rather than silently defaulted somewhere.
"""

import json
import os
from dataclasses import asdict

import numpy as np
import torch

from muon_research import fork
from muon_research.constants import FILENAME_CONFIGS
from muon_research.curv import ProfileConfig
from muon_research.data import DistributedDataCursor
from muon_research.rules import Rule, RuleSet, TrainConfig
from muon_research.scripts.run_curv_profile import (
    _load_pool_batches,
    run_checkpoint_profile,
)

MAGIC = 20240520
HEADER_INT32S = 256

# Disjoint, individually-identifiable ranges: train tokens are small
# integers, val tokens start well above them -- both stay within uint16
# (the on-disk shard dtype's own range, 0..65535: a token id outside that
# would silently wrap around when written, defeating the whole point of
# these being individually identifiable). vocab_size (below) is far larger
# than either range's top, so `% vocab_size` (next_batch's own wraparound,
# a SEPARATE step from the on-disk dtype) is a no-op and every token's own
# identity survives intact, letting tests assert exact (dis)equality of
# raw token values.
TRAIN_TOKENS = np.arange(0, 20_000, dtype=np.int64)
VAL_TOKENS = np.arange(40_000, 60_000, dtype=np.int64)


def _write_shard(path: str, tokens: np.ndarray) -> None:
    header = np.zeros(HEADER_INT32S, dtype=np.int32)
    header[0] = MAGIC
    header[1] = 1
    header[2] = tokens.size
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens.astype("<u2").tobytes())


def _make_data_source(base_dir: str) -> str:
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    _write_shard(os.path.join(data_dir, "train_000.bin"), TRAIN_TOKENS)
    _write_shard(os.path.join(data_dir, "val_000.bin"), VAL_TOKENS)
    return data_dir


def tiny_train_config(data_source: str, **overrides) -> TrainConfig:
    kwargs = dict(
        data_source=data_source,
        train_steps=1000,
        report_steps=1000,
        seq_len=4,
        val_size=16,
        batch_size=8,
        mbs=1,
        vocab_size=1_000_000,
        num_layers=1,
        model_dim=16,
        head_dim=8,
        num_heads=2,
        train_data_pattern="train_*.bin",
        val_data_pattern="val_*.bin",
    )
    kwargs.update(overrides)
    return TrainConfig(**kwargs)


def tiny_profile_config(**overrides) -> ProfileConfig:
    kwargs = dict(profile_source="weight")
    kwargs.update(overrides)
    return ProfileConfig(**kwargs)


def _fake_ckpt(train_config: TrainConfig, *, advance_tokens: int = 0) -> dict:
    """A resumed train cursor's own ``state_dict``, as ``fork.load_checkpoint``
    would return it inside ``ckpt["train_loader"]`` -- optionally advanced
    first, to check pool A really does resume from a non-zero position, not
    just always the file start."""
    cursor = DistributedDataCursor(
        os.path.join(train_config.data_source, train_config.train_data_pattern),
        train_config.batch_size,
        vocab_size=train_config.vocab_size,
        seq_len=train_config.seq_len,
    )
    if advance_tokens:
        cursor.advance_tokens(advance_tokens)
    return {"train_loader": cursor.state_dict()}


def _flat_tokens(batches) -> torch.Tensor:
    """Every input token across every microbatch, flattened -- for
    equality checks that don't care about mbs chunking."""
    return torch.cat([x.reshape(-1) for x, _y in batches]).cpu()


########################################
#     _load_pool_batches, in isolation  #
########################################


def test_profile_batch_size_none_pool_b_is_pool_a_exactly(tmp_path):
    data_source = _make_data_source(str(tmp_path))
    tc = tiny_train_config(data_source)
    pc = tiny_profile_config(profile_batch_size=None)
    ckpt = _fake_ckpt(tc)

    pool_a, pool_b = _load_pool_batches(tc, pc, ckpt, checkpoint_step=0)

    assert pool_b is pool_a, "profile_batch_size unset must reuse pool A as-is"
    assert torch.equal(_flat_tokens(pool_a), _flat_tokens(pool_b))


def test_profile_batch_size_set_pool_b_disjoint_from_pool_a(tmp_path):
    data_source = _make_data_source(str(tmp_path))
    tc = tiny_train_config(data_source)
    pc = tiny_profile_config(profile_batch_size=8)
    ckpt = _fake_ckpt(tc)

    pool_a, pool_b = _load_pool_batches(tc, pc, ckpt, checkpoint_step=0)

    a_tokens = set(_flat_tokens(pool_a).tolist())
    b_tokens = set(_flat_tokens(pool_b).tolist())
    assert a_tokens.issubset(set(TRAIN_TOKENS.tolist())), "pool A must be train data"
    assert b_tokens.issubset(set(VAL_TOKENS.tolist())), "pool B must be held-out val"
    assert a_tokens.isdisjoint(b_tokens), (
        "pool A and pool B must not share any token -- this is exactly the "
        "bug: pre-fix, pool B (and hence pool A too, for profile_source in "
        "{'grad', 'signal'}) would have come from this same val batch"
    )


def test_pool_a_unaffected_by_profile_batch_size(tmp_path):
    """The core bug being guarded against: pool A must be identical
    whether or not profile_batch_size is set -- it's always the resumed
    train cursor, never redirected onto held-out val data."""
    data_source = _make_data_source(str(tmp_path))
    tc = tiny_train_config(data_source)
    ckpt = _fake_ckpt(tc)

    pool_a_unset, _ = _load_pool_batches(
        tc, tiny_profile_config(profile_batch_size=None), ckpt, checkpoint_step=0
    )
    pool_a_set, _ = _load_pool_batches(
        tc, tiny_profile_config(profile_batch_size=8), ckpt, checkpoint_step=0
    )
    assert torch.equal(_flat_tokens(pool_a_unset), _flat_tokens(pool_a_set))


def test_pool_a_resumes_checkpoints_own_cursor_position(tmp_path):
    """Pool A isn't just "the first batch of the file" -- it must resume
    from wherever the checkpoint's own train cursor had gotten to."""
    data_source = _make_data_source(str(tmp_path))
    tc = tiny_train_config(data_source)
    pc = tiny_profile_config(profile_batch_size=None)

    ckpt_fresh = _fake_ckpt(tc, advance_tokens=0)
    ckpt_advanced = _fake_ckpt(tc, advance_tokens=800)

    pool_a_fresh, _ = _load_pool_batches(tc, pc, ckpt_fresh, checkpoint_step=0)
    pool_a_advanced, _ = _load_pool_batches(tc, pc, ckpt_advanced, checkpoint_step=0)

    assert not torch.equal(_flat_tokens(pool_a_fresh), _flat_tokens(pool_a_advanced))
    assert int(_flat_tokens(pool_a_fresh)[0].item()) == int(TRAIN_TOKENS[0])
    assert int(_flat_tokens(pool_a_advanced)[0].item()) == int(TRAIN_TOKENS[0]) + 800


def test_profile_batch_resample_false_pool_b_fixed_across_steps(tmp_path):
    data_source = _make_data_source(str(tmp_path))
    tc = tiny_train_config(data_source)
    pc = tiny_profile_config(profile_batch_size=8, profile_batch_resample=False)
    ckpt = _fake_ckpt(tc)

    _, pool_b_step0 = _load_pool_batches(tc, pc, ckpt, checkpoint_step=0)
    _, pool_b_step5 = _load_pool_batches(tc, pc, ckpt, checkpoint_step=5)

    assert torch.equal(_flat_tokens(pool_b_step0), _flat_tokens(pool_b_step5))


def test_profile_batch_resample_true_pool_b_varies_across_steps(tmp_path):
    data_source = _make_data_source(str(tmp_path))
    tc = tiny_train_config(data_source)
    pc = tiny_profile_config(profile_batch_size=8, profile_batch_resample=True)
    ckpt = _fake_ckpt(tc)

    _, pool_b_step0 = _load_pool_batches(tc, pc, ckpt, checkpoint_step=0)
    _, pool_b_step5 = _load_pool_batches(tc, pc, ckpt, checkpoint_step=5)

    assert not torch.equal(_flat_tokens(pool_b_step0), _flat_tokens(pool_b_step5))
    expected_start = int(VAL_TOKENS[0]) + tc.val_size + 5 * pc.profile_batch_size
    assert int(_flat_tokens(pool_b_step5)[0].item()) == expected_start


def test_profile_batch_resample_true_step0_matches_fixed_batch(tmp_path):
    """checkpoint_step=0 advances 0 extra tokens either way, so the
    resampled and fixed variants must coincide exactly at step 0."""
    data_source = _make_data_source(str(tmp_path))
    tc = tiny_train_config(data_source)
    ckpt = _fake_ckpt(tc)

    _, pool_b_fixed = _load_pool_batches(
        tc,
        tiny_profile_config(profile_batch_size=8, profile_batch_resample=False),
        ckpt,
        checkpoint_step=0,
    )
    _, pool_b_resampled = _load_pool_batches(
        tc,
        tiny_profile_config(profile_batch_size=8, profile_batch_resample=True),
        ckpt,
        checkpoint_step=0,
    )
    assert torch.equal(_flat_tokens(pool_b_fixed), _flat_tokens(pool_b_resampled))


def test_pool_b_never_overlaps_val_size_prefix(tmp_path):
    """Pool B must start strictly after val_size tokens into the val
    stream, so it never overlaps whatever eval reads from that same
    pattern."""
    data_source = _make_data_source(str(tmp_path))
    tc = tiny_train_config(data_source, val_size=100)
    pc = tiny_profile_config(profile_batch_size=8)
    ckpt = _fake_ckpt(tc)

    _, pool_b = _load_pool_batches(tc, pc, ckpt, checkpoint_step=0)
    assert int(_flat_tokens(pool_b)[0].item()) == int(VAL_TOKENS[0]) + 100


def test_microbatch_splitting_preserved_for_both_pools(tmp_path):
    """mbs=1 with batch_size=8/seq_len=4 (2 sequences) must yield 2
    microbatches of 1 sequence each, for both pools."""
    data_source = _make_data_source(str(tmp_path))
    tc = tiny_train_config(data_source, mbs=1)
    pc = tiny_profile_config(profile_batch_size=8)
    ckpt = _fake_ckpt(tc)

    pool_a, pool_b = _load_pool_batches(tc, pc, ckpt, checkpoint_step=0)
    assert len(pool_a) == 2
    assert len(pool_b) == 2
    for x, y in pool_a + pool_b:
        assert x.shape == (1, tc.seq_len)
        assert y.shape == (1, tc.seq_len)


########################################
#   run_checkpoint_profile, end to end  #
########################################


def _gpl_train_config(data_source: str, **overrides) -> TrainConfig:
    kwargs = dict(
        data_source=data_source,
        train_steps=100,
        report_steps=100,
        seq_len=8,
        val_size=64,
        batch_size=16,
        mbs=2,
        vocab_size=1000,
        num_layers=1,
        model_dim=16,
        model_type="gpl",
        embed_dim=8,
        num_tokens=4,
        expansion_ratio=2.0,
        train_data_pattern="train_*.bin",
        val_data_pattern="val_*.bin",
    )
    kwargs.update(overrides)
    return TrainConfig(**kwargs)


def _all_rule(train_steps: int, **overrides) -> Rule:
    kwargs = dict(
        name="all",
        patterns=["*"],
        start=0,
        end=train_steps,
        update="adamw",
        sizing="learning_rate",
        lr=0.01,
        betas=(0.9, 0.95),
        nesterov=True,
        wd_raw=0.0,
    )
    kwargs.update(overrides)
    return Rule(**kwargs)


def _fake_batch(tc: TrainConfig):
    x = torch.randint(0, tc.vocab_size, (tc.mbs, tc.seq_len), device="cuda")
    y = torch.randint(0, tc.vocab_size, (tc.mbs, tc.seq_len), device="cuda")
    return x, y


def _build_checkpoint(
    source_path: str, tc: TrainConfig, rules: list[Rule], *, warmup_steps: int
) -> int:
    """A real (tiny) GPL model+Geon, optionally given a few real training
    steps first (random batches -- just to give its momentum non-trivial
    state, so profile_source="signal" actually exercises the historical-
    momentum blend rather than degenerating to "grad") -- then saved as a
    real checkpoint at ``<source_path>/checkpoints/step_<n>.pt`` plus
    ``<source_path>/config.json``, exactly what run_checkpoint_profile
    reads. Returns the saved step number.
    """
    built = fork.build_model_and_geon(tc, RuleSet(rules))
    model, optimizer, rule_set = built["model"], built["optimizer"], built["rule_set"]
    named_params = list(model.named_parameters())
    for step in range(warmup_steps):
        x, y = _fake_batch(tc)
        loss = model(x, y)
        loss.backward()
        updates, sizings = rule_set.apply_for_step(step, 1.0, named_params, optimizer)
        optimizer.step(updates, sizings, model=model, batches=[(x, y)])
        model.zero_grad(set_to_none=True)

    # The checkpoint's own train-cursor position -- a fresh cursor at the
    # start of the (separate, file-backed) train shard, independent of the
    # random warmup batches above. What pool A resumes from.
    cursor = DistributedDataCursor(
        os.path.join(tc.data_source, tc.train_data_pattern),
        tc.batch_size,
        vocab_size=tc.vocab_size,
        seq_len=tc.seq_len,
    )

    os.makedirs(source_path, exist_ok=True)
    payload = {**asdict(tc), "rules": [asdict(r) for r in rules]}
    with open(os.path.join(source_path, FILENAME_CONFIGS), "w", encoding="utf-8") as f:
        json.dump(payload, f)

    fork.save_checkpoint(
        os.path.join(source_path, "checkpoints"),
        warmup_steps,
        model,
        optimizer,
        cursor.state_dict(),
    )
    return warmup_steps


def _profiled_matrix_payload(run_path: str, step: int, matrix_name: str) -> dict:
    profile_path = os.path.join(
        run_path, "profiles", f"step_{step:06d}", "svd_curv.pt"
    )
    payload = torch.load(profile_path, map_location="cpu", weights_only=False)
    return payload["matrices"][matrix_name]


def test_end_to_end_grad_pool_a_unaffected_by_profile_batch_size(
    tmp_path, monkeypatch
):
    """profile_source="grad" is the most exposed case: pool A IS p.grad
    directly. sigma (pool A's own SVD) must come out identical whether or
    not profile_batch_size is set -- pre-fix, setting it would have
    redirected the forward+backward (hence p.grad, hence sigma) onto the
    held-out val batch instead of the resumed train batch. gamma_diag
    (pool B), by contrast, MUST differ -- proving pool B is actually being
    read from wherever profile_batch_size points it, not silently ignored.
    """
    monkeypatch.setenv("LOCAL_RANK", "0")
    source_path = os.path.join(tmp_path, "source")
    data_source = _make_data_source(source_path)
    tc = _gpl_train_config(data_source)
    rules = [_all_rule(tc.train_steps)]
    step = _build_checkpoint(source_path, tc, rules, warmup_steps=3)

    matrix_name = "blocks.0.fc.weight"
    run_path_none = os.path.join(tmp_path, "run_none")
    run_path_set = os.path.join(tmp_path, "run_set")

    run_checkpoint_profile(
        "none",
        source_path,
        step,
        ProfileConfig(
            profile_source="grad", profile_batch_size=None,
            compute_gamma=True, compute_phi=False, max_modes=4,
        ),
        run_path_none,
    )
    run_checkpoint_profile(
        "set",
        source_path,
        step,
        ProfileConfig(
            profile_source="grad", profile_batch_size=16,
            compute_gamma=True, compute_phi=False, max_modes=4,
        ),
        run_path_set,
    )

    mat_none = _profiled_matrix_payload(run_path_none, step, matrix_name)
    mat_set = _profiled_matrix_payload(run_path_set, step, matrix_name)

    assert torch.allclose(mat_none["sigma"], mat_set["sigma"], atol=1e-5, rtol=1e-5), (
        "sigma (pool A) must not depend on profile_batch_size for "
        "profile_source='grad'"
    )
    # Exact (not allclose) on purpose: gamma_diag's own natural scale here
    # is tiny (~1e-5, a lightly-trained toy model), so any fixed allclose
    # tolerance either trivially passes (too loose) or is fragile (too
    # tight) -- two independent computations over genuinely different real
    # data essentially never coincide bit-for-bit, so exact inequality is
    # the robust way to confirm pool B's data actually changed.
    assert not torch.equal(mat_none["gamma_diag"], mat_set["gamma_diag"]), (
        "gamma_diag (pool B) should differ once profile_batch_size points "
        "it at genuinely different (held-out val) data -- if this fails, "
        "pool B likely isn't wired through at all"
    )


def test_end_to_end_signal_pool_a_unaffected_by_profile_batch_size(
    tmp_path, monkeypatch
):
    """Same check as the "grad" test, for profile_source="signal" -- the
    exact configuration experiments/exp003_curvature (article.ipynb
    section 4) actually used. Real prior training steps first, so the
    optimizer's own momentum is non-trivial and "signal" genuinely blends
    it with the current gradient (a fresh, never-stepped optimizer would
    make _updated_signal degenerate to plain "grad", too weak a check).
    """
    monkeypatch.setenv("LOCAL_RANK", "0")
    source_path = os.path.join(tmp_path, "source")
    data_source = _make_data_source(source_path)
    tc = _gpl_train_config(data_source)
    rules = [_all_rule(tc.train_steps)]
    step = _build_checkpoint(source_path, tc, rules, warmup_steps=3)

    matrix_name = "blocks.0.fc.weight"
    run_path_none = os.path.join(tmp_path, "run_none")
    run_path_set = os.path.join(tmp_path, "run_set")

    run_checkpoint_profile(
        "none",
        source_path,
        step,
        ProfileConfig(
            profile_source="signal", profile_batch_size=None,
            compute_gamma=False, compute_phi=False,
        ),
        run_path_none,
    )
    run_checkpoint_profile(
        "set",
        source_path,
        step,
        ProfileConfig(
            profile_source="signal", profile_batch_size=16,
            compute_gamma=False, compute_phi=False,
        ),
        run_path_set,
    )

    mat_none = _profiled_matrix_payload(run_path_none, step, matrix_name)
    mat_set = _profiled_matrix_payload(run_path_set, step, matrix_name)

    assert torch.allclose(mat_none["sigma"], mat_set["sigma"], atol=1e-5, rtol=1e-5), (
        "sigma (pool A, via profile_source='signal') must not depend on "
        "profile_batch_size"
    )
