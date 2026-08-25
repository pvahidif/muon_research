"""Executable verification that muon_research.fork produces branches that
are genuinely independent copies of a trunk -- no shared storage with the
trunk or with each other, in either mutation direction -- and that a
branch's own hyperparameters (not the trunk's) are what its optimizer
actually reads.

This exists because reading the source was not enough. Two real bugs were
only caught by running real training steps and comparing tensor
identity/values/behavior:

1. The original code (independently, in more than one run_branch_*.py
   script) assumed
   `optimizer.load_state_dict(other_optimizer.state_dict())` clones
   tensors, the way `Module.load_state_dict` does. It doesn't, in this
   PyTorch version -- `Optimizer.state_dict()` returns live tensors by
   reference and `load_state_dict` installs them as-is. That left every
   branch's Adam momentum (m/v) aliasing the trunk's own tensors, and
   every sibling branch forked from the same trunk snapshot. See
   `test_mutating_one_branch_leaves_trunk_and_siblings_untouched`, which
   fails against the pre-fix code and passes against fork.fork_branch.

2. `optimizer.load_state_dict(...)` -- whether from another live
   optimizer or from a checkpoint -- unconditionally replaces
   `optimizer.param_groups` with new dict objects. The original code
   (independently, again) cached a `group_of` (param -> its param_group
   dict) mapping once and reused it, which meant every write through that
   mapping after a `load_state_dict` call landed in an orphaned dict
   nobody read, so a branch whose rule overrides betas/nesterov/lr/wd_raw
   would silently keep using the trunk's original ones. Now structurally
   impossible: `Geon.group_of(p)` is a plain lookup, called fresh every
   time (see `RuleSet.apply_for_step`), never cached by any caller. See
   `test_fork_branch_uses_its_own_hyperparameters_not_trunks`.
"""

import os
import tempfile

import torch

from muon_research import fork
from muon_research.rules import Rule, RuleSet, TrainConfig


def tiny_train_config(**overrides) -> TrainConfig:
    kwargs = dict(
        data_source="dummy",
        train_steps=1000,
        report_steps=100,
        seq_len=16,
        val_size=16,
        batch_size=4,
        mbs=4,
        vocab_size=32,
        num_layers=1,
        model_dim=16,
        head_dim=8,
        num_heads=2,
    )
    kwargs.update(overrides)
    return TrainConfig(**kwargs)


def all_rule(train_steps: int, **overrides) -> Rule:
    kwargs = dict(
        name="all",
        patterns=["*"],
        start=0,
        end=train_steps,
        update="adamw",
        sizing="learning_rate",
        lr=0.01,
        betas=(0.9, 0.95),
        nesterov=False,
        wd_raw=0.0,
    )
    kwargs.update(overrides)
    return Rule(**kwargs)


def fake_batch(tc: TrainConfig):
    x = torch.randint(0, tc.vocab_size, (tc.batch_size, tc.seq_len), device="cuda")
    y = torch.randint(0, tc.vocab_size, (tc.batch_size, tc.seq_len), device="cuda")
    return x, y


def train_step_with_batch(
    built: dict, x: torch.Tensor, y: torch.Tensor, step: int
) -> None:
    """Same real forward/backward/optimizer.step as train_step, but with
    an explicit (x, y) batch instead of a freshly-sampled one -- for tests
    that need two independently-built objects to train on IDENTICAL data
    (e.g. checkpoint resume-equivalence, where the whole point is to
    compare final states, so any RNG-driven difference in batch content
    between the two runs would be a confound)."""
    model, optimizer = built["model"], built["optimizer"]
    loss = model(x, y)
    loss.backward()
    named_params = list(model.named_parameters())
    updates, sizings = built["rule_set"].apply_for_step(
        step, 1.0, named_params, optimizer
    )
    optimizer.step(updates, sizings, model=model, batches=[(x, y)])
    model.zero_grad(set_to_none=True)


def train_step(built: dict, tc: TrainConfig, step: int) -> None:
    """One real forward/backward/optimizer.step, same call shape
    run_optim_rules.run_geon_rules's own training loop uses."""
    x, y = fake_batch(tc)
    train_step_with_batch(built, x, y, step)


def build_trained_trunk(
    train_steps: int = 5, seed: int = 0
) -> tuple[dict, TrainConfig]:
    """A trunk that's had a few *real* training steps, so its optimizer
    momentum is non-zero -- forking a freshly-initialized (all-zero)
    optimizer state can't tell a shared tensor from an independent one."""
    torch.manual_seed(seed)
    tc = tiny_train_config()
    rule_set = RuleSet([all_rule(tc.train_steps)])
    built = fork.build_model_and_geon(tc, rule_set)
    optimizer = built["optimizer"]
    for step in range(train_steps):
        train_step(built, tc, step)
    any_nonzero = any(float(s["m"].abs().sum()) > 0 for s in optimizer.state.values())
    assert any_nonzero, "test setup bug: trunk momentum is all zero"
    return built, tc


def snapshot(built: dict) -> dict:
    """Clone of every param + optimizer (m, v) tensor, keyed by param
    name, plus every buffer (e.g. Rotary's angular_freq) -- for
    before/after comparison after some other object trains."""
    model, optimizer = built["model"], built["optimizer"]
    params = {}
    for name, p in model.named_parameters():
        st = optimizer.state[p]
        params[name] = (
            p.detach().clone(),
            st["m"].clone(),
            st["v"].clone(),
            st["step"],
        )
    buffers = {name: b.detach().clone() for name, b in model.named_buffers()}
    return {"params": params, "buffers": buffers}


def assert_unchanged(built: dict, before: dict, who: str) -> None:
    model, optimizer = built["model"], built["optimizer"]
    for name, p in model.named_parameters():
        p0, m0, v0, step0 = before["params"][name]
        assert torch.equal(p, p0), f"{who}: param {name} changed -- ENTANGLED"
        st = optimizer.state[p]
        assert torch.equal(
            st["m"], m0
        ), f"{who}: optimizer m[{name}] changed -- ENTANGLED"
        assert torch.equal(
            st["v"], v0
        ), f"{who}: optimizer v[{name}] changed -- ENTANGLED"
        assert (
            st["step"] == step0
        ), f"{who}: optimizer step[{name}] changed -- ENTANGLED"
    for name, b in model.named_buffers():
        assert torch.equal(
            b, before["buffers"][name]
        ), f"{who}: buffer {name} changed -- ENTANGLED"


def assert_no_shared_storage(built_a: dict, built_b: dict, who: str) -> None:
    a_params = dict(built_a["model"].named_parameters())
    b_params = dict(built_b["model"].named_parameters())
    for name in a_params:
        pa, pb = a_params[name], b_params[name]
        assert pa.data_ptr() != pb.data_ptr(), f"{who}: param {name} shares storage"
        sa, sb = built_a["optimizer"].state[pa], built_b["optimizer"].state[pb]
        assert (
            sa["m"].data_ptr() != sb["m"].data_ptr()
        ), f"{who}: state[m][{name}] shares storage"
        assert (
            sa["v"].data_ptr() != sb["v"].data_ptr()
        ), f"{who}: state[v][{name}] shares storage"
    a_buffers = dict(built_a["model"].named_buffers())
    b_buffers = dict(built_b["model"].named_buffers())
    assert a_buffers.keys() == b_buffers.keys(), f"{who}: buffer name sets differ"
    for name, ba in a_buffers.items():
        bb = b_buffers[name]
        if ba.numel() == 0:
            continue
        assert ba.data_ptr() != bb.data_ptr(), f"{who}: buffer {name} shares storage"


def assert_values_equal(built_a: dict, built_b: dict, who: str) -> None:
    a_params = dict(built_a["model"].named_parameters())
    b_params = dict(built_b["model"].named_parameters())
    for name in a_params:
        pa, pb = a_params[name], b_params[name]
        assert torch.equal(pa, pb), f"{who}: param {name} value differs"
        sa, sb = built_a["optimizer"].state[pa], built_b["optimizer"].state[pb]
        assert torch.equal(sa["m"], sb["m"]), f"{who}: state[m][{name}] differs"
        assert torch.equal(sa["v"], sb["v"]), f"{who}: state[v][{name}] differs"
        assert sa["step"] == sb["step"], f"{who}: state[step][{name}] differs"
    a_buffers = dict(built_a["model"].named_buffers())
    b_buffers = dict(built_b["model"].named_buffers())
    for name, ba in a_buffers.items():
        assert torch.equal(ba, b_buffers[name]), f"{who}: buffer {name} differs"


def all_live_tensors(built: dict) -> list[torch.Tensor]:
    """Every tensor reachable from a built (model, optimizer) pair that
    could, in principle, alias another one: params, buffers, and Adam
    (m, v) state -- used by the fill-probe tests to mutate every byte at
    once rather than checking identity/values piecemeal."""
    model, optimizer = built["model"], built["optimizer"]
    tensors = [p for _name, p in model.named_parameters()]
    tensors += [b for _name, b in model.named_buffers() if b.numel() > 0]
    for p in model.parameters():
        st = optimizer.state.get(p)
        if st:
            tensors += [st["m"], st["v"]]
    return tensors


def fake_print0(s, **kwargs):
    pass


def do_fork(trunk_built: dict, tc: TrainConfig, label: str, **kwargs) -> dict:
    rule_set = RuleSet([all_rule(tc.train_steps)])
    return fork.fork_branch(
        trunk_built["model"],
        trunk_built["optimizer"],
        tc,
        rule_set,
        compile_models=False,
        label=label,
        print0=fake_print0,
        **kwargs,
    )


# --------------------------------------------------------------------------


def test_build_model_and_geon_has_expected_custom_attributes():
    """A fresh build (not a fork) sets Geon's own custom attributes from
    config -- the whole reason fork_branch avoids copy.deepcopy(optimizer)
    (which would silently drop them, per its own docstring)."""
    tc = tiny_train_config(geon_s_min=1e-4, geon_s_max=1e4)
    rule_set = RuleSet([all_rule(tc.train_steps)])
    built = fork.build_model_and_geon(tc, rule_set)
    model, optimizer = built["model"], built["optimizer"]
    assert optimizer.s_min == 1e-4
    assert optimizer.s_max == 1e4
    assert optimizer._step_count == 0
    assert len(optimizer.state) == 0  # nothing trained yet
    assert len(list(model.named_parameters())) == len(optimizer.param_groups)


def test_fork_branch_matches_trunk_immediately_after_fork():
    """Right after forking, a branch's weights and optimizer state must be
    bit-identical to the trunk's -- not just independently-stored, but
    correctly copied."""
    trunk, tc = build_trained_trunk()
    branch = do_fork(trunk, tc, "b1")
    assert_values_equal(trunk, branch, "branch vs trunk right after fork")


def test_fork_branch_no_shared_storage_with_trunk():
    trunk, tc = build_trained_trunk()
    branch = do_fork(trunk, tc, "b1")
    assert_no_shared_storage(trunk, branch, "branch vs trunk")


def test_three_branches_pairwise_independent_storage_and_values():
    """Not just trunk-vs-one-branch: fork three branches from the same
    trunk snapshot and check every pair, including branch-vs-branch."""
    trunk, tc = build_trained_trunk()
    b1 = do_fork(trunk, tc, "b1")
    b2 = do_fork(trunk, tc, "b2")
    b3 = do_fork(trunk, tc, "b3")
    for name, a, b in [
        ("trunk-b1", trunk, b1),
        ("trunk-b2", trunk, b2),
        ("trunk-b3", trunk, b3),
        ("b1-b2", b1, b2),
        ("b1-b3", b1, b3),
        ("b2-b3", b2, b3),
    ]:
        assert_no_shared_storage(a, b, name)
        assert_values_equal(a, b, name)


def test_mutating_one_branch_leaves_trunk_and_siblings_untouched():
    """The core regression test: really train one branch for several real
    steps and confirm nothing else moved. This is the check that fails
    against the pre-fix code (aliased optimizer state) and passes against
    fork.fork_branch."""
    trunk, tc = build_trained_trunk()
    b1 = do_fork(trunk, tc, "b1")
    b2 = do_fork(trunk, tc, "b2")

    trunk_before = snapshot(trunk)
    b2_before = snapshot(b2)

    for step in range(5, 10):
        train_step(b1, tc, step)

    # sanity: b1 actually changed
    b1_params_after = dict(b1["model"].named_parameters())
    trunk_params = dict(trunk["model"].named_parameters())
    changed = any(
        not torch.equal(b1_params_after[n], trunk_params[n]) for n in trunk_params
    )
    assert changed, "test setup bug: branch1 didn't change after training"

    assert_unchanged(trunk, trunk_before, "trunk (after training branch1)")
    assert_unchanged(b2, b2_before, "branch2 (after training branch1)")


def test_mutating_trunk_leaves_existing_branches_untouched():
    """Reverse direction -- relevant to any caller whose trunk keeps
    training after forking a branch off it."""
    trunk, tc = build_trained_trunk()
    b1 = do_fork(trunk, tc, "b1")
    b2 = do_fork(trunk, tc, "b2")

    b1_before = snapshot(b1)
    b2_before = snapshot(b2)

    for step in range(5, 10):
        train_step(trunk, tc, step)

    assert_unchanged(b1, b1_before, "branch1 (after continuing to train trunk)")
    assert_unchanged(b2, b2_before, "branch2 (after continuing to train trunk)")


def test_klmatch_schedule_unset_does_not_call_setter(monkeypatch):
    """When fork_branch's klmatch_schedule kwarg isn't passed at all (a
    caller happy with Geon's own constructed default), set_kl_match_cache_schedule
    must not be called at all. This matters even though the *end state*
    (Schedule(None)) looks the same as calling it with None explicitly --
    a real caller-configured Geon default would be silently overwritten by
    a naive 'always call it' implementation, and that bug wouldn't show up
    by inspecting cache_schedule afterward. Spy on the actual call instead
    of the end state."""
    from muon_research.optim.geon import Geon

    calls = []
    orig = Geon.set_kl_match_cache_schedule

    def spy(self, *a, **k):
        calls.append((a, k))
        return orig(self, *a, **k)

    monkeypatch.setattr(Geon, "set_kl_match_cache_schedule", spy)

    trunk, tc = build_trained_trunk()
    do_fork(trunk, tc, "b1")  # no klmatch_schedule kwarg
    assert calls == [], (
        f"set_kl_match_cache_schedule was called even though klmatch_schedule "
        f"was never passed to fork_branch: {calls}"
    )


def test_klmatch_schedule_explicit_none_calls_setter(monkeypatch):
    """klmatch_schedule=None (explicitly passed, unlike the previous test)
    must still route through set_kl_match_cache_schedule exactly once --
    the _UNSET sentinel must distinguish this from 'not given', not treat
    them the same."""
    from muon_research.optim.geon import Geon

    calls = []
    orig = Geon.set_kl_match_cache_schedule

    def spy(self, *a, **k):
        calls.append((a, k))
        return orig(self, *a, **k)

    monkeypatch.setattr(Geon, "set_kl_match_cache_schedule", spy)

    trunk, tc = build_trained_trunk()
    branch = do_fork(trunk, tc, "b1", klmatch_schedule=None)
    assert len(calls) == 1 and calls[0][0] == (
        None,
    ), f"expected exactly one set_kl_match_cache_schedule(None) call, got {calls}"
    assert branch["optimizer"].kl_match_cache_schedule.cache_schedule is None


def test_klmatch_schedule_real_value_is_applied():
    trunk, tc = build_trained_trunk()
    spec = {"_type": "ap", "k": 4}
    branch = do_fork(trunk, tc, "b1", klmatch_schedule=spec)
    assert branch["optimizer"].kl_match_cache_schedule.cache_schedule == spec


def test_fork_branch_uses_its_own_hyperparameters_not_trunks():
    """Fork a branch whose rule overrides betas/nesterov/lr/wd_raw to
    values very different from the trunk's, apply_for_step it (exactly as
    the real training loop does every iteration), and confirm what Geon
    ACTUALLY reads internally -- optimizer.group_of(p), a fresh lookup,
    not any cached mapping -- reflects the branch's own values."""
    trunk, tc = build_trained_trunk()
    branch_rule = all_rule(
        tc.train_steps,
        update="adamw",
        sizing="learning_rate",
        lr=0.02,
        betas=(0.5, 0.5),
        nesterov=True,
        wd_raw=0.0003,
    )
    branch = fork.fork_branch(
        trunk["model"],
        trunk["optimizer"],
        tc,
        RuleSet([branch_rule]),
        compile_models=False,
        label="b1",
        print0=fake_print0,
    )
    named_params = list(branch["model"].named_parameters())
    branch["rule_set"].apply_for_step(0, 1.0, named_params, branch["optimizer"])
    p0 = named_params[0][1]
    internal_group = branch["optimizer"].group_of(p0)
    assert internal_group["betas"] == (0.5, 0.5), (
        f"Geon's real internal betas are {internal_group['betas']}, not the "
        f"branch's own (0.5, 0.5) rule"
    )
    assert internal_group["nesterov"] is True
    assert internal_group["lr"] == 0.02
    assert internal_group["wd_raw"] == 0.0003
    # And a real optimizer.step must actually use these, not the trunk's
    # (0.9, 0.95)/False -- run one step and confirm momentum matches a
    # manual Adam-style update with the branch's own betas.
    x, y = fake_batch(tc)
    loss = branch["model"](x, y)
    loss.backward()
    updates, sizings = branch["rule_set"].apply_for_step(
        1, 1.0, named_params, branch["optimizer"]
    )
    m_before = branch["optimizer"].state[p0]["m"].clone()
    grad = p0.grad.clone()
    branch["optimizer"].step(updates, sizings, model=branch["model"], batches=[(x, y)])
    m_after = branch["optimizer"].state[p0]["m"]
    expected_m = 0.5 * m_before + 0.5 * grad
    assert torch.allclose(m_after, expected_m, atol=1e-5), (
        "branch optimizer.step did not use the branch's own beta1=0.5 -- "
        f"got m update inconsistent with beta1=0.5 (max diff "
        f"{(m_after - expected_m).abs().max().item()})"
    )


def test_load_checkpoint_restores_hyperparameters_from_checkpoint_not_fresh_build():
    """load_checkpoint's optimizer.load_state_dict(...) call must actually
    install the checkpoint's own saved hyperparameters -- confirmed via
    optimizer.group_of(p) (a fresh lookup, not any cached mapping) -- not
    leave the fresh build's own initial-construction values in place."""
    trunk, tc = build_trained_trunk()
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = os.path.join(tmpdir, "checkpoints")
        fork.save_checkpoint(
            ckpt_dir,
            5,
            trunk["model"],
            trunk["optimizer"],
            trunk["model"].state_dict(),  # stand-in train_loader_state
        )
        ckpt_path = os.path.join(ckpt_dir, "step_5.pt")

        # Different rule than the trunk was built/trained with -- proves
        # what's asserted below comes from load_state_dict actually
        # restoring the checkpoint's own (0.9, 0.95)/False, not this fresh
        # build's initial (0.3, 0.7)/True surviving untouched.
        rule_set = RuleSet([all_rule(tc.train_steps, betas=(0.3, 0.7), nesterov=True)])
        built = fork.build_model_and_geon(tc, rule_set)
        model, optimizer = built["model"], built["optimizer"]
        named_params = list(model.named_parameters())
        ckpt = fork.load_checkpoint(ckpt_path, model, optimizer, device="cuda")
        assert ckpt["step"] == 5
        for _name, p in named_params:
            # The checkpoint's own stored hyperparameters (from the
            # trunk's all_rule defaults), not this fresh build's (0.3, 0.7).
            group = optimizer.group_of(p)
            assert group["betas"] == (0.9, 0.95)
            assert group["nesterov"] is False


def test_compile_and_warmup_does_not_change_weights():
    """compile_and_warmup runs a throwaway forward+backward on the branch
    before load_state_dict overlays the trunk's real weights -- confirm it
    only populates/clears .grad, never touches the fresh model's .data, so
    running it doesn't leak into what load_state_dict then installs."""
    tc = tiny_train_config()
    rule_set = RuleSet([all_rule(tc.train_steps)])
    model = fork.build_model_and_geon(tc, rule_set)["model"]
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    fork.compile_and_warmup(model, tc, fake_print0, "test")
    for n, p in model.named_parameters():
        assert torch.equal(p, before[n]), f"compile_and_warmup changed {n}'s weights"
        assert p.grad is None, f"compile_and_warmup left a stale .grad on {n}"


# ---------------------------------------------------------------------------
# Buffers: params and optimizer (m, v) are covered above (and by every test
# using assert_no_shared_storage/assert_values_equal/assert_unchanged, all
# of which now check buffers too) -- this makes that coverage explicit and
# documents why buffers were never at risk the same way optimizer state was.
# ---------------------------------------------------------------------------


def test_fork_branch_buffers_are_independent_objects_not_shared_with_trunk():
    """Model buffers (Rotary's persistent angular_freq; Linear's
    non-persistent _act_sink/_err_sink covariance-capture scratch space)
    go through Module.load_state_dict (copies into the target's own
    pre-allocated storage, like params) or are simply constructed fresh by
    the branch's own __init__ (for persistent=False buffers, which
    load_state_dict skips entirely) -- neither path is the
    Optimizer.load_state_dict-returns-live-tensors pattern that caused
    Bug 1, so buffers were never at risk of aliasing. Verified directly
    anyway, the same way as params, rather than just trusting that
    argument."""
    trunk, tc = build_trained_trunk()
    branch = do_fork(trunk, tc, "b1")
    trunk_buffers = dict(trunk["model"].named_buffers())
    branch_buffers = dict(branch["model"].named_buffers())
    assert trunk_buffers.keys() == branch_buffers.keys()
    assert len(trunk_buffers) > 0, "test setup bug: model has no buffers to check"
    for name, tb in trunk_buffers.items():
        bb = branch_buffers[name]
        if tb.numel() == 0:
            continue
        assert tb.data_ptr() != bb.data_ptr(), f"buffer {name} shares storage"
        assert torch.equal(tb, bb), f"buffer {name} value differs right after fork"


# ---------------------------------------------------------------------------
# Direct memory-level ("fill probe") verification: instead of inferring
# independence from data_ptr() or from values after normal training, write
# a distinctive sentinel directly into every live tensor of one object and
# confirm zero bytes of it are visible from any other object.
# ---------------------------------------------------------------------------


def test_fill_probe_no_storage_overlap_between_trunk_and_branches():
    """The most literal 'check the underlying memory bits' test: overwrite
    every element of every trunk tensor (params, buffers, optimizer m/v)
    in place with a sentinel value, then confirm neither branch's
    corresponding tensors contain it anywhere -- and the reverse, filling
    one branch and checking the trunk and its sibling. data_ptr()
    inequality proves the storages are different allocations; this proves
    writing through one allocation is genuinely invisible from the
    other -- the failure mode data_ptr() alone can't rule out is a
    same-storage view with a nonzero offset (data_ptr() differs, contents
    still alias part of the same buffer), which this test would catch."""
    trunk, tc = build_trained_trunk()
    b1 = do_fork(trunk, tc, "b1")
    b2 = do_fork(trunk, tc, "b2")

    sentinel = 64.0  # exact in bf16 and fp32 alike (a small power of two)

    for t in all_live_tensors(trunk):
        t.data.fill_(sentinel)
    for who, built in [("b1", b1), ("b2", b2)]:
        for t in all_live_tensors(built):
            assert not torch.any(t == sentinel), (
                f"{who}: contains the trunk's fill-probe sentinel -- shares "
                f"underlying storage with the trunk"
            )

    trunk2, tc2 = build_trained_trunk()
    c1 = do_fork(trunk2, tc2, "c1")
    c2 = do_fork(trunk2, tc2, "c2")
    for t in all_live_tensors(c1):
        t.data.fill_(sentinel)
    for who, built in [("trunk", trunk2), ("c2 (sibling branch)", c2)]:
        for t in all_live_tensors(built):
            assert not torch.any(t == sentinel), (
                f"{who}: contains branch c1's fill-probe sentinel -- shares "
                f"underlying storage with c1"
            )


# ---------------------------------------------------------------------------
# Chained forking and adversarial (interleaved) training order.
# ---------------------------------------------------------------------------


def test_grandchild_fork_is_independent_of_parent_branch_and_trunk():
    """fork_branch takes any live (model, optimizer) as its source -- not
    specially 'the trunk' -- so a branch can itself be forked from. Fork
    b1 from the trunk, train b1 so it diverges, fork
    grandchild g1 from b1, and confirm independence holds at every
    generation and in every direction."""
    trunk, tc = build_trained_trunk()
    b1 = do_fork(trunk, tc, "b1")
    for step in range(5, 8):
        train_step(b1, tc, step)

    g1 = do_fork(b1, tc, "g1")
    assert_values_equal(b1, g1, "grandchild vs parent branch right after fork")
    assert_no_shared_storage(trunk, g1, "trunk vs grandchild")
    assert_no_shared_storage(b1, g1, "parent branch vs grandchild")

    trunk_before = snapshot(trunk)
    b1_before = snapshot(b1)
    for step in range(8, 11):
        train_step(g1, tc, step)
    assert_unchanged(trunk, trunk_before, "trunk (after training grandchild)")
    assert_unchanged(b1, b1_before, "parent branch (after training grandchild)")

    g1_before = snapshot(g1)
    trunk_before = snapshot(trunk)
    for step in range(8, 11):
        train_step(b1, tc, step)
    assert_unchanged(
        trunk, trunk_before, "trunk (after further training parent branch)"
    )
    assert_unchanged(g1, g1_before, "grandchild (after further training parent branch)")


def test_interleaved_training_many_branches_no_cross_contamination():
    """Adversarial ordering: every other test trains one object at a time
    to completion before touching the next. Here, round-robin real
    training steps across the trunk and three branches instead, so
    optimizer.step() calls on different Geon instances interleave in time
    -- guards against any bug that only shows up under interleaved (not
    sequential) mutation, e.g. shared mutable state accidentally hanging
    off a class attribute instead of self."""
    trunk, tc = build_trained_trunk()
    b1 = do_fork(trunk, tc, "b1")
    b2 = do_fork(trunk, tc, "b2")
    b3 = do_fork(trunk, tc, "b3")
    objs = [trunk, b1, b2, b3]
    step = 5
    for _round in range(6):
        for obj in objs:
            train_step(obj, tc, step)
            step += 1

    pairs = [
        ("trunk-b1", trunk, b1),
        ("trunk-b2", trunk, b2),
        ("trunk-b3", trunk, b3),
        ("b1-b2", b1, b2),
        ("b1-b3", b1, b3),
        ("b2-b3", b2, b3),
    ]
    for name, a, b in pairs:
        assert_no_shared_storage(a, b, name)
    # Storage independence alone wouldn't catch every branch secretly
    # reading the SAME hyperparameter dict (distinct storage per param,
    # but every branch computing an identical update because they all
    # read one shared lr/betas) -- with independent random batches, that
    # would still show up as suspiciously-identical params after
    # training. Rule it out directly.
    for name, a, b in pairs:
        a_params = dict(a["model"].named_parameters())
        b_params = dict(b["model"].named_parameters())
        assert any(
            not torch.equal(a_params[n], b_params[n]) for n in a_params
        ), f"{name}: identical values after independent interleaved training"


# ---------------------------------------------------------------------------
# save_checkpoint / load_checkpoint: roundtrip correctness, independence
# from the source, and file-level edge cases.
# ---------------------------------------------------------------------------


def test_checkpoint_roundtrip_into_fresh_model_matches_trunk_and_has_no_shared_storage():
    """save_checkpoint + load_checkpoint into a completely fresh
    (model, optimizer) -- the shape of every script's resume path -- must
    reproduce the trunk's values exactly. Independence is expected almost
    for free here (torch.save/load always materializes brand-new tensors,
    unlike fork_branch's live-optimizer-to-live-optimizer path, which
    needed the explicit copy.deepcopy fix) but is checked directly anyway,
    the same way as a fork_branch branch."""
    trunk, tc = build_trained_trunk()
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = os.path.join(tmpdir, "checkpoints")
        fork.save_checkpoint(
            ckpt_dir, 5, trunk["model"], trunk["optimizer"], {"dummy": True}
        )
        ckpt_path = os.path.join(ckpt_dir, "step_5.pt")

        rule_set = RuleSet([all_rule(tc.train_steps)])
        loaded = fork.build_model_and_geon(tc, rule_set)
        ckpt = fork.load_checkpoint(
            ckpt_path, loaded["model"], loaded["optimizer"], device="cuda"
        )
        assert ckpt["step"] == 5
        assert_values_equal(trunk, loaded, "checkpoint-loaded vs trunk")
        assert_no_shared_storage(trunk, loaded, "checkpoint-loaded vs trunk")


def test_checkpoint_resume_matches_uninterrupted_training_bit_for_bit():
    """The strongest correctness proof for save/load together: continuing
    an uninterrupted trunk for 5 more steps must produce EXACTLY the same
    final weights/optimizer state as saving a checkpoint at that point,
    loading it into a completely fresh (model, optimizer), and replaying
    the identical 5 batches there. Any silent state loss or corruption in
    save_checkpoint/load_checkpoint -- a dropped optimizer key, a buffer
    that didn't round-trip -- would make the two final states diverge.
    The two runs are given the exact same (x, y) batches (not just the
    same RNG seed) so this isolates save/load correctness from any
    incidental RNG-determinism assumption.

    Deliberately uses a TWO-PHASE rule (one betas/nesterov/lr/wd_raw for
    steps [0, 5), a different one for [5, train_steps)) rather than one
    constant rule for the whole run: a fresh build's initial hyperparameters
    always reflect step 0's rule, and the checkpoint itself (saved right at
    the phase boundary) also still reflects phase 0 -- so with a single
    constant rule, a resumed run's replay would apply the correct
    hyperparameters purely by coincidence, without proving that
    load_checkpoint's optimizer.load_state_dict() call, combined with
    RuleSet.apply_for_step's fresh optimizer.group_of(p) lookups, actually
    picks up the checkpoint's own restored state each step (see
    test_load_checkpoint_restores_hyperparameters_from_checkpoint_not_fresh_build
    for that same trap, avoided there by giving the fresh build a different
    rule up front). The phase change here forces the replay to actually
    apply new hyperparameters partway through, so this test only passes if
    resuming is doing real work, not just because the checkpoint already
    happened to have the right values."""
    torch.manual_seed(0)
    tc = tiny_train_config()

    def two_phase_rules():
        # A fresh RuleSet per build_model_and_geon call: it resolves
        # (mutates) whichever instance it's given, so reusing one object
        # across the trunk and the resumed-from-checkpoint model below
        # would have the second build's resolve() overwrite the first's.
        return RuleSet(
            [
                all_rule(
                    tc.train_steps,
                    name="phase0",
                    end=5,
                    betas=(0.9, 0.95),
                    nesterov=False,
                    lr=0.01,
                    wd_raw=0.0,
                ),
                all_rule(
                    tc.train_steps,
                    name="phase1",
                    start=5,
                    betas=(0.3, 0.6),
                    nesterov=True,
                    lr=0.03,
                    wd_raw=0.002,
                ),
            ]
        )

    trunk = fork.build_model_and_geon(tc, two_phase_rules())
    for step in range(5):
        train_step(trunk, tc, step)  # phase0

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = os.path.join(tmpdir, "checkpoints")
        fork.save_checkpoint(
            ckpt_dir, 5, trunk["model"], trunk["optimizer"], {"dummy": True}
        )
        ckpt_path = os.path.join(ckpt_dir, "step_5.pt")

        batches = [fake_batch(tc) for _ in range(5)]

        for i, (x, y) in enumerate(batches):
            train_step_with_batch(trunk, x, y, step=5 + i)  # phase1

        resumed = fork.build_model_and_geon(tc, two_phase_rules())
        fork.load_checkpoint(
            ckpt_path, resumed["model"], resumed["optimizer"], device="cuda"
        )
        for i, (x, y) in enumerate(batches):  # phase1
            train_step_with_batch(resumed, x, y, step=5 + i)

        assert_values_equal(
            trunk, resumed, "resumed-from-checkpoint vs uninterrupted training"
        )


def test_save_checkpoint_noops_and_writes_nothing_on_nonzero_rank(monkeypatch):
    """save_checkpoint is self-guarding: only rank 0 ever writes (every
    rank computes bit-identical state off the same all-reduced gradients,
    see its own docstring), so on any other rank it must return
    immediately without creating the checkpoint directory or any file in
    it. This test harness only ever runs single-process rank 0, so faking
    dist.get_rank() is the only way to exercise that branch at all."""
    monkeypatch.setattr(fork.dist, "get_rank", lambda: 1)
    trunk, tc = build_trained_trunk(train_steps=1)
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = os.path.join(tmpdir, "checkpoints")
        fork.save_checkpoint(ckpt_dir, 1, trunk["model"], trunk["optimizer"], {})
        assert not os.path.exists(
            ckpt_dir
        ), "save_checkpoint created a directory/file on a non-zero rank"


def test_save_checkpoint_leaves_no_tmp_file_behind():
    """Written to a .tmp path and atomically os.replace()'d into place --
    confirm the directory holds exactly the final step_<n>.pt and nothing
    else once save_checkpoint returns."""
    trunk, tc = build_trained_trunk(train_steps=1)
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = os.path.join(tmpdir, "checkpoints")
        fork.save_checkpoint(ckpt_dir, 3, trunk["model"], trunk["optimizer"], {})
        entries = os.listdir(ckpt_dir)
        assert entries == [
            "step_3.pt"
        ], f"expected exactly one file step_3.pt, found {entries}"


def test_save_checkpoint_extra_kwargs_are_saved_verbatim():
    """**extra (training_time/rolling_loss/rolling_loss_step for
    run_optim_rules.py's own trunk checkpoints) must round-trip through
    the saved payload untouched, since save_checkpoint doesn't know ahead
    of time which caller passed what."""
    trunk, tc = build_trained_trunk(train_steps=1)
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = os.path.join(tmpdir, "checkpoints")
        fork.save_checkpoint(
            ckpt_dir,
            2,
            trunk["model"],
            trunk["optimizer"],
            {"cursor": 7},
            training_time=12.5,
            rolling_loss=3.25,
            rolling_loss_step=2,
        )
        raw = torch.load(os.path.join(ckpt_dir, "step_2.pt"), weights_only=False)
        assert raw["train_loader"] == {"cursor": 7}
        assert raw["training_time"] == 12.5
        assert raw["rolling_loss"] == 3.25
        assert raw["rolling_loss_step"] == 2


def test_find_latest_checkpoint_returns_none_when_dir_missing_or_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        missing_dir = os.path.join(tmpdir, "does_not_exist")
        assert fork.find_latest_checkpoint(missing_dir) is None
        empty_dir = os.path.join(tmpdir, "empty")
        os.makedirs(empty_dir)
        assert fork.find_latest_checkpoint(empty_dir) is None


def test_find_latest_checkpoint_picks_highest_step_and_ignores_non_matching_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        for name in [
            "step_1.pt",
            "step_20.pt",
            "step_3.pt",
            "step_20.pt.tmp",
            "notes.txt",
            "step_abc.pt",
        ]:
            open(os.path.join(tmpdir, name), "w").close()
        assert fork.find_latest_checkpoint(tmpdir) == os.path.join(tmpdir, "step_20.pt")


# ---------------------------------------------------------------------------
# Foundational invariants and documented (non-bug) asymmetries the rest of
# this file's tests implicitly rely on.
# ---------------------------------------------------------------------------


def test_named_parameters_order_is_deterministic_across_independent_builds():
    """torch.optim.Optimizer.load_state_dict() (used by both
    load_checkpoint and fork_branch) remaps saved state to the current
    optimizer's params purely by positional order through param_groups --
    it has no notion of param names or identity. That's only correct if
    named_parameters() always yields the same order for two
    independently-constructed models built from the same (TrainConfig,
    rules): otherwise a param's saved momentum could silently attach to a
    different tensor after a state_dict restored from another model
    instance (another process's trunk, or a checkpoint written days ago).
    Verified directly, not just assumed."""
    tc = tiny_train_config()
    model_a = fork.build_model_and_geon(tc, RuleSet([all_rule(tc.train_steps)]))["model"]
    model_b = fork.build_model_and_geon(tc, RuleSet([all_rule(tc.train_steps)]))["model"]
    names_a = [n for n, _p in model_a.named_parameters()]
    names_b = [n for n, _p in model_b.named_parameters()]
    assert (
        names_a == names_b
    ), "named_parameters() order differs between independently-built models"


def test_fork_branch_step_count_and_kl_matched_coeffs_start_fresh_not_copied_from_trunk():
    """Geon's step_dict()/load_state_dict() (inherited from
    torch.optim.Optimizer, never overridden) only ever covers {state,
    param_groups}. Instance attributes that live outside that --
    _step_count, kl_matched_coeffs, s_min, s_max, kl_search_* -- are NEVER
    touched by fork_branch's `b_optimizer.load_state_dict(...)` call, so a
    freshly-forked branch always keeps whatever build_model_and_geon set
    them to (_step_count=0, kl_matched_coeffs={}), regardless of how far
    the trunk has trained or what it has cached. This is intentional (see
    fork_branch's own docstring for why copy.deepcopy(optimizer) is
    avoided) but is a real, non-obvious asymmetry with the
    state/param_groups fields (which DO carry over) -- frozen here as an
    explicit test so a future change to either behavior is a deliberate
    decision, not a silent side-effect."""
    trunk, tc = build_trained_trunk(train_steps=5)
    trunk["optimizer"]._step_count = 42
    trunk["optimizer"].kl_matched_coeffs[frozenset({"marker"})] = 0.99

    branch = do_fork(trunk, tc, "b1")

    assert branch["optimizer"]._step_count == 0
    assert branch["optimizer"].kl_matched_coeffs == {}
    # Not merely empty right now -- a genuinely separate dict object, not
    # the trunk's own dict aliased and about to fill up in lockstep.
    branch["optimizer"].kl_matched_coeffs[frozenset({"other"})] = 0.5
    assert frozenset({"other"}) not in trunk["optimizer"].kl_matched_coeffs


def test_fork_branch_with_compile_models_true_still_matches_and_is_independent():
    """Real script callers with branch_config.compile_models=True compile
    the branch's freshly-initialized model and run a throwaway
    forward/backward on it (compile_and_warmup) BEFORE load_state_dict
    overlays the trunk's real weights. Every other test in this file uses
    compile_models=False (compiling just costs wall-clock time); this one
    exercises the compiled path directly, since torch.compile rewrites
    the model's forward graph and it's worth confirming that doesn't
    somehow disturb parameter identity or what load_state_dict installs
    afterward."""
    trunk, tc = build_trained_trunk()
    branch = fork.fork_branch(
        trunk["model"],
        trunk["optimizer"],
        tc,
        RuleSet([all_rule(tc.train_steps)]),
        compile_models=True,
        label="b1_compiled",
        print0=fake_print0,
    )
    assert_values_equal(trunk, branch, "compiled branch vs trunk right after fork")
    assert_no_shared_storage(trunk, branch, "compiled branch vs trunk")

    for step in range(5, 8):
        train_step(branch, tc, step)
    trunk_before = snapshot(trunk)
    assert_unchanged(
        trunk, trunk_before, "trunk (after training a compile_models=True branch)"
    )
