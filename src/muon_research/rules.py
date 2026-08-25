"""The rules config schema -- ``TrainConfig`` (a run's shape/hyperparameters)
and ``Rule`` (one param-set/step-range optimizer spec) -- plus everything
that reads, validates, sweeps, or applies it: parsing both from YAML/JSON,
resolving each param's rules for a given step, applying an
``override_args``/branch-spec combo, and computing a rule's live
optimizer-group values at a training step. Shared by every script that
trains or forks a Geon-optimized model (see ``muon_research.fork``) or
sweeps this schema (``muon_research.scripts.run_optim_rules`` and the
``run_branch_*`` scripts) -- see ``run_optim_rules.py``'s module docstring
for the YAML schema itself (what a config file looks like), and this
module for what happens to it once read.

Everything that operates on a *single* ``Rule`` (parsing one YAML/JSON
entry, checking its own step range, its per-step lr-warmup factor) is a
method on ``Rule`` itself. Everything that operates on a *collection* of
rules (coverage/uniqueness across params, override_args sweeps, applying
the active rule to every param each step) is a method on ``RuleSet``.
Nothing here is a free-floating private function that exists only to be
called from exactly one place in one of those two classes.
"""

from __future__ import annotations

import itertools
import json
import os
from dataclasses import MISSING, Field, dataclass, fields, replace
from fnmatch import fnmatch

import torch
import yaml

from muon_research.constants import DEFAULT_SEED, FILENAME_CONFIGS
from muon_research.optim.geon import Geon, SizingEntry, UpdateKind

MODEL_CHOICES = ("gpt", "gpl")


@dataclass
class TrainConfig:
    # data
    # Directory containing train_data_pattern/val_data_pattern shards, kept
    # exactly as given (relative or absolute) -- never eagerly resolved
    # against the repo root, so it round-trips unchanged through config.json
    # and stays portable across checkouts/machines. Callers that actually
    # open shard files resolve it themselves, on demand, via
    # muon_research.paths.resolve_repo_path.
    data_source: str
    # run
    train_steps: int
    report_steps: int
    seq_len: int
    val_size: int
    batch_size: int
    mbs: int
    # model config
    vocab_size: int
    num_layers: int
    model_dim: int

    # data (defaults)
    train_data_pattern: str = "fineweb_train_*.bin"
    val_data_pattern: str = "fineweb_val_*.bin"
    # run (defaults)
    seed: int = DEFAULT_SEED
    # model config (defaults)
    model_type: str = "gpt"  # "gpt" (attention) or "gpl" (fixed-window MLP)
    expansion_ratio: float = 4.0
    # GPT-only
    head_dim: int | None = None
    num_heads: int | None = None
    # GPL-only
    embed_dim: int | None = None
    num_tokens: int | None = None
    # initialization
    zero_proj_init: bool = True
    # optim config (defaults) -- global schedule + Geon-wide constants only;
    # everything per-param (lr/betas/nesterov/wd_raw/update/sizing, per step
    # range) comes from 'rules', not TrainConfig.
    lr_cooldown_frac: float = 0.7
    geon_eps: float = 1e-10
    geon_s_min: float = 1e-5
    geon_s_max: float = 1e5
    # Steps completed so far at which to checkpoint model/optimizer/
    # data-loader state to <run_path>/checkpoints/step_<n>.pt -- see
    # muon_research.fork.save_checkpoint (e.g. 500 = "after the 500th
    # step"; 0 = the freshly initialized model, before any training). None
    # (default) or [] means never checkpoint. If the run is interrupted,
    # the next invocation resumes from the latest one found instead of
    # restarting from step 0 -- see muon_research.fork.find_latest_checkpoint.
    checkpoint_steps: list[int] | None = None

    def __post_init__(self):
        if self.model_type not in MODEL_CHOICES:
            raise ValueError(
                f"model_type must be one of {MODEL_CHOICES}, got {self.model_type!r}"
            )
        if self.model_type == "gpt":
            missing = [f for f in ("head_dim", "num_heads") if getattr(self, f) is None]
        else:
            missing = [
                f for f in ("embed_dim", "num_tokens") if getattr(self, f) is None
            ]
        if missing:
            raise ValueError(
                f"model_type={self.model_type!r} requires: {sorted(missing)}"
            )
        if self.checkpoint_steps is not None:
            self.checkpoint_steps = sorted({int(s) for s in self.checkpoint_steps})
            if self.checkpoint_steps and self.checkpoint_steps[0] < 0:
                raise ValueError(
                    f"checkpoint_steps must all be >= 0, got {self.checkpoint_steps!r}"
                )

    def eta_of(self, step: int) -> float:
        """Global lr/wd_raw multiplier at ``step``: 1.0 for the "stable"
        leading ``1 - lr_cooldown_frac`` fraction of training, then linear
        decay to 0 over the trailing ``lr_cooldown_frac`` fraction. Applied
        on top of every rule's own (base) lr/wd_raw -- see
        ``RuleSet.apply_for_step``."""
        progress = step / self.train_steps
        assert 0 <= progress < 1
        if progress < 1 - self.lr_cooldown_frac:
            return 1.0
        return (1 - progress) / self.lr_cooldown_frac

    @classmethod
    def load_from_config(cls, path: str) -> "TrainConfig":
        """Read the required top-level 'train' section into a
        ``TrainConfig``."""
        with open(path, encoding="utf-8") as f:
            payload = yaml.safe_load(f)
        payload = payload["train"]
        field_names = {field.name for field in fields(cls)}
        required = {
            field.name
            for field in fields(cls)
            if field.default is MISSING and field.default_factory is MISSING
        }
        missing = required - payload.keys()
        if missing:
            raise ValueError(f"config {path!r} is missing keys: {sorted(missing)}")
        unknown = payload.keys() - field_names
        if unknown:
            raise ValueError(f"config {path!r} has unknown keys: {sorted(unknown)}")
        return cls(**payload)


@dataclass
class Rule:
    """One fully-self-contained optimizer spec for a (param set, step
    range) — see run_optim_rules.py's module docstring for the config
    schema. ``start``/``end`` are 0-indexed, half-open (``[start, end)``),
    in the same ``step`` units the training loop's own loop variable uses.
    ``end=None`` means "through train_steps" -- never resolved to a
    concrete value; every consumer (``RuleSet.resolve``, ``rule_for_step``)
    treats it as such directly, against whatever ``train_steps`` applies at
    that point (a combo's own, if overridden -- not necessarily the base
    config's).
    """

    name: str
    patterns: list[str]
    start: int
    end: int | None
    update: UpdateKind
    sizing: str
    lr: float
    betas: tuple[float, float]
    nesterov: bool
    wd_raw: float
    coeff: float = 1.0
    # Steps of linear lr warmup, counted from this rule's own activation
    # (0 = no warmup): at step `step`, with `s = step - start` steps since
    # this rule became active, lr is scaled by an extra
    # min(1, (s+1)/warmup_steps) on top of the usual stable-then-cooldown
    # eta(step) -- see warmup_factor below. Relative to `start` (not
    # absolute step 0) so a rule that only activates mid-training (e.g.
    # `start=1000`) still gets its own fresh warmup when it does. wd_raw is
    # untouched by warmup (still just wd_raw * eta, as before).
    warmup_steps: int = 0

    SIZING_KINDS = ("learning_rate", "kl_match", "fro_match", "op_match")

    @classmethod
    def data_fields(cls) -> dict[str, Field]:
        """Every field except ``name`` (a rule's own identity, used to
        address it from override_args) and ``start``/``end`` (set via the
        YAML-only 'steps' convenience instead), keyed by name -- the ones
        a YAML rule entry or a ``rules.<name>.<field>`` override coerces
        generically. A field's own declared default here (e.g. ``coeff``'s
        ``1.0``, ``warmup_steps``'s ``0``) is exactly what makes it
        optional in a YAML entry -- see ``from_entry`` -- so there's no
        separate "which fields are optional" list to keep in sync by hand.
        """
        return {
            f.name: f for f in fields(cls) if f.name not in ("name", "start", "end")
        }

    @staticmethod
    def coerce_field(field_name: str, value):
        """Coerce + validate one raw (YAML- or JSON-typed) value for a
        single field -- shared by ``from_entry``, ``from_payload``, and
        ``RuleSet.apply_overrides`` (one ``rules.<name>.<field>`` override
        value), so all three go through identical validation.
        """
        if field_name == "patterns":
            if isinstance(value, str):
                value = [value]
            value = [str(v) for v in value]
            if not value:
                raise ValueError("'patterns' must be non-empty")
            return value
        if field_name == "start":
            return int(value)
        if field_name == "end":
            return None if value is None else int(value)
        if field_name == "update":
            if isinstance(value, (list, tuple)):
                if len(value) == 2:
                    return (str(value[0]), float(value[1]))
                if len(value) == 3:
                    return (str(value[0]), float(value[1]), float(value[2]))
                raise ValueError(
                    f"'update' list must be [name, power] or [name, q1, q2], "
                    f"got {value!r}"
                )
            return str(value)
        if field_name == "sizing":
            value = str(value)
            if value not in Rule.SIZING_KINDS:
                raise ValueError(
                    f"'sizing' must be one of {Rule.SIZING_KINDS}, got {value!r}"
                )
            return value
        if field_name == "lr":
            return float(value)
        if field_name == "betas":
            betas = tuple(float(b) for b in value)
            if len(betas) != 2:
                raise ValueError(f"'betas' must have 2 entries, got {value!r}")
            return betas
        if field_name == "nesterov":
            return bool(value)
        if field_name == "wd_raw":
            return float(value)
        if field_name == "coeff":
            return float(value)
        if field_name == "warmup_steps":
            v = int(value)
            if v < 0:
                raise ValueError(f"'warmup_steps' must be >= 0, got {v}")
            return v
        raise ValueError(f"unknown Rule field {field_name!r}")

    @classmethod
    def from_entry(cls, entry: dict, label: str) -> Rule:
        """Parse one raw (YAML-typed) rule dict -- ``{"steps": [start, end],
        <data fields>...}`` -- into a ``Rule``. Shared by
        ``RuleSet.load_from_config`` (the top-level 'rules' list) and
        ``RuleSet.apply_overrides`` (a 'rules' override replacing a
        branch's entire rule list), so both accept identical rule
        syntax/validation. Doesn't check the range against any
        ``train_steps`` -- callers do that themselves, via ``check_range``,
        once they know their own combo's final ``train_steps`` (which may
        itself have just been overridden).
        """
        try:
            steps = entry["steps"]
            if len(steps) != 2:
                raise ValueError(f"'steps' must be [start, end], got {steps!r}")
            start = cls.coerce_field("start", steps[0])
            end = cls.coerce_field("end", steps[1])
            kwargs = {}
            for name, field in cls.data_fields().items():
                if name in entry:
                    raw = entry[name]
                elif field.default is not MISSING:
                    raw = field.default
                elif field.default_factory is not MISSING:
                    raw = field.default_factory()
                else:
                    raise KeyError(name)
                kwargs[name] = cls.coerce_field(name, raw)
        except KeyError as e:
            raise ValueError(f"rule {label!r} is missing key {e}") from e
        return cls(name=label, start=start, end=end, **kwargs)

    @classmethod
    def from_payload(cls, payload: dict) -> Rule:
        """Inverse of ``asdict(rule)`` -- reuses ``coerce_field`` (same
        coercion ``from_entry`` uses) so ``betas``/``update`` round-trip
        their tuple types correctly from JSON's list-only representation.
        Used to read a checkpoint's already-``asdict``'d rules back off
        disk (see ``RuleSet.load_from_payload``, ``load_checkpoint_config``).
        """
        return cls(
            name=str(payload["name"]),
            start=cls.coerce_field("start", payload["start"]),
            end=cls.coerce_field("end", payload["end"]),
            **{
                name: cls.coerce_field(name, payload[name])
                for name in cls.data_fields()
            },
        )

    def check_range(self, train_steps: int) -> None:
        """Raise if this rule's ``[start, end)`` (``end=None`` meaning
        "through train_steps") doesn't satisfy
        ``0 <= start < end <= train_steps``. Never writes a concrete value
        onto ``self.end``."""
        effective_end = train_steps if self.end is None else self.end
        if not 0 <= self.start < effective_end <= train_steps:
            raise ValueError(
                f"rule {self.name!r}: steps must satisfy 0 <= start < end <= "
                f"train_steps ({train_steps}), got [{self.start}, {self.end})"
            )

    def warmup_factor(self, step: int) -> float:
        """Linear lr warmup multiplier at ``step`` -- see
        ``Rule.warmup_steps``: 1.0 (no-op) once ``warmup_steps`` steps have
        passed since ``start`` (or immediately, if ``warmup_steps <= 0``).
        """
        if self.warmup_steps <= 0:
            return 1.0
        steps_since_start = step - self.start
        return min(1.0, (steps_since_start + 1) / self.warmup_steps)


class RuleSet:
    """A list of ``Rule``s, plus (once ``resolve``'d) the per-param lookup
    derived from them -- the one object every script threads from load
    through override through the training loop, instead of separately
    carrying a ``list[Rule]`` and a ``dict[str, list[Rule]]`` everywhere.

    Lifecycle: construct (or ``load_from_config``/``load_from_payload``),
    optionally ``apply_overrides`` (returns a new, still-unresolved
    ``RuleSet``), then ``resolve`` once real param names and a final
    ``train_steps`` are known (see ``fork.build_model_and_geon``) -- after
    that, ``rule_for_step``/``apply_for_step`` answer per-step queries for
    the remainder of that model's training.
    """

    def __init__(self, rules: list[Rule]):
        self.rules = rules
        self._per_param: dict[str, list[Rule]] | None = None

    def __len__(self) -> int:
        return len(self.rules)

    def __repr__(self) -> str:
        return repr(self.rules)

    @classmethod
    def load_from_config(cls, path: str) -> RuleSet:
        """Read the required top-level 'rules' list; ``steps: [start,
        null]`` leaves ``end`` as ``None`` ("through train_steps"), which
        every consumer interprets directly -- it's never resolved to a
        concrete value. Raises ValueError on any malformed entry
        (missing/invalid key) or a duplicate ``name`` (names must be
        unique -- they're how ``override_args`` addresses a specific
        rule) -- range/coverage/uniqueness-across-params is checked
        separately, by ``resolve``, once a final ``train_steps`` (a combo
        may still override it) and the model's param names are both
        known.
        """
        with open(path, encoding="utf-8") as f:
            payload = yaml.safe_load(f)
        raw = payload.get("rules") or []
        if not raw:
            raise ValueError(f"config {path!r} has no 'rules' entries")

        rules = []
        names_seen = set()
        for idx, entry in enumerate(raw):
            label = str(entry.get("name", f"rules[{idx}]"))
            if label in names_seen:
                raise ValueError(
                    f"duplicate rule name {label!r} -- names must be unique "
                    f"(they address a rule from override_args)"
                )
            names_seen.add(label)
            rules.append(Rule.from_entry(entry, label))
        return cls(rules)

    @classmethod
    def load_from_payload(cls, payload: list[dict]) -> RuleSet:
        """Inverse of ``[asdict(rule) for rule in rule_set.rules]`` -- a
        checkpoint's already-``asdict``'d rules, read back off disk (see
        ``Rule.from_payload``, ``load_checkpoint_config``)."""
        return cls([Rule.from_payload(r) for r in payload])

    def resolve(self, param_names: list[str], train_steps: int) -> RuleSet:
        """Checks every rule's own ``[start, end)`` against this final
        ``train_steps`` (see ``Rule.check_range``), then, for every name in
        ``param_names``, computes (and caches) the sorted-by-``start`` list
        of rules whose ``patterns`` match it -- validated to exactly
        partition ``[0, train_steps)`` for that param, with no gaps and no
        overlaps (two rules both matching the same (param, step) would
        make "the thing to pass to Geon" ambiguous, which is exactly what
        this rules out). Also requires every rule to match at least one
        param (typo protection, same spirit as curv.py's
        ``select_matrix_params``).

        Raises one combined ``ValueError`` covering every problem
        param/rule, rather than failing on the first one, so a broken rule
        set can be fixed in one pass.

        Mutates and returns ``self`` (chainable), rather than a new
        ``RuleSet``: unlike ``apply_overrides``, this doesn't change
        *which* rules apply, only caches how to look them up for this
        specific model.
        """
        for rule in self.rules:
            rule.check_range(train_steps)

        errors = []
        rule_matched = [False] * len(self.rules)
        per_param: dict[str, list[Rule]] = {}

        for name in param_names:
            matches = [
                rule
                for rule in self.rules
                if any(fnmatch(name, pat) for pat in rule.patterns)
            ]
            for rule in matches:
                rule_matched[self.rules.index(rule)] = True
            matches.sort(key=lambda rule: rule.start)

            cursor = 0
            covered = True
            for rule in matches:
                if rule.start != cursor:
                    covered = False
                    break
                cursor = train_steps if rule.end is None else rule.end
            covered = covered and cursor == train_steps

            if not covered:
                spans = [(rule.start, rule.end, rule.name) for rule in matches]
                errors.append(
                    f"  {name!r}: rules cover {spans} -- must exactly partition "
                    f"[0, {train_steps}) with no gaps or overlaps"
                )
            else:
                per_param[name] = matches

        unmatched_rules = [
            self.rules[i].name for i, matched in enumerate(rule_matched) if not matched
        ]
        if unmatched_rules:
            errors.append(f"  rules matched no param at all: {unmatched_rules}")

        if errors:
            raise ValueError(
                f"rules do not uniquely cover every (param, step) in "
                f"[0, {train_steps}):\n" + "\n".join(errors)
            )
        self._per_param = per_param
        return self

    def rule_for_step(self, param_name: str, step: int) -> Rule:
        """The one rule active for ``param_name`` at ``step``. Requires
        ``resolve`` to have been called first."""
        if self._per_param is None:
            raise RuntimeError(
                "RuleSet.resolve() must be called before rule_for_step()"
            )
        for rule in self._per_param[param_name]:
            if rule.start <= step and (rule.end is None or step < rule.end):
                return rule
        raise AssertionError(
            f"step {step} not covered for {param_name!r} -- resolve() should "
            f"have caught this"
        )

    def validate_override(self, override: dict) -> None:
        """Validate every key of one override dict -- a bare ``TrainConfig``
        field name, ``"rules.<rule_name>.<field>"`` targeting one specific
        rule's field (any ``Rule`` field except ``name``, which is a
        rule's own fixed identity, not something to sweep), or the literal
        key ``"rules"`` (replacing the entire rule list, see
        ``apply_overrides``) -- the one override-validation entry point
        every script uses: ``apply_overrides`` calls this on every combo
        it's given, and run_branch_compare.py's own ``_validate_branch_specs``
        additionally calls it upfront, on every branch spec at once, so a
        config typo is caught before any GPU work starts rather than only
        once that branch's own fork step arrives. A change to what counts
        as a valid override key (like adding ``"rules"`` support) takes
        effect everywhere at once either way.

        ``"rules"`` (replacing the entire rule list) can't be combined
        with any ``"rules.<name>.<field>"`` key in the *same* override
        dict -- which rule set the per-field key should apply to (the base
        rules, or the ones ``"rules"`` is about to replace them with)
        would be ambiguous. Put the per-field values directly in the
        ``"rules"`` list's entries instead.
        """
        train_config_fields = {f.name for f in fields(TrainConfig)}
        rule_names = {rule.name for rule in self.rules}

        full_rules_override = False
        partial_rule_override = False
        for key in override:
            if key == "rules":
                full_rules_override = True
                continue
            if key in train_config_fields:
                continue
            parts = key.split(".")
            if len(parts) == 3 and parts[0] == "rules":
                partial_rule_override = True
                _prefix, rule_name, field_name = parts
                if rule_name not in rule_names:
                    raise ValueError(
                        f"override_args key {key!r}: unknown rule {rule_name!r}"
                    )
                if field_name not in Rule.data_fields():
                    raise ValueError(
                        f"override_args key {key!r}: unknown Rule field "
                        f"{field_name!r}"
                    )
                continue
            raise ValueError(
                f"override_args key {key!r} must be a TrainConfig field name, "
                f"'rules.<rule_name>.<field>' targeting one rule's field, or the "
                f"literal key 'rules' (replacing the whole rule list)"
            )
        if full_rules_override and partial_rule_override:
            raise ValueError(
                "override can't combine a 'rules' (whole rule list replacement) "
                "key with any 'rules.<name>.<field>' key in the same override -- "
                "put the per-field values directly in the 'rules' list instead"
            )

    def apply_for_step(
        self,
        step: int,
        eta: float,
        named_params: list[tuple[str, torch.nn.Parameter]],
        optimizer: Geon,
    ) -> tuple[dict[torch.nn.Parameter, UpdateKind], list[SizingEntry]]:
        """Per-step: for every param, find its active rule (via
        ``rule_for_step``), overwrite that param's own Geon group's lr
        (``eta``, the caller's own stable-then-cooldown schedule value,
        further scaled by that rule's own lr warmup -- see
        ``Rule.warmup_factor``)/wd_raw (``eta``-scaled only, no warmup)/
        betas/nesterov in place (via ``optimizer.group_of(p)``, so no
        stale mapping to keep in sync -- see ``Geon.group_of``), and build
        ``updates``/``sizings`` for ``optimizer.step()``. Requires
        ``resolve`` to have been called first.

        Params governed by the *same* rule object at this step are grouped
        into one shared sizing entry -- rules are exactly the granularity a
        joint ``kl_match``/``fro_match``/``op_match`` probe (see
        ``Geon._resolve_sizes``) should apply at.
        """
        if self._per_param is None:
            raise RuntimeError(
                "RuleSet.resolve() must be called before apply_for_step()"
            )

        updates: dict[torch.nn.Parameter, UpdateKind] = {}
        by_rule: dict[int, list[torch.nn.Parameter]] = {}
        rule_of_id: dict[int, Rule] = {}

        for name, p in named_params:
            rule = self.rule_for_step(name, step)
            group = optimizer.group_of(p)
            group["lr"] = rule.lr * rule.warmup_factor(step) * eta
            group["wd_raw"] = rule.wd_raw * eta
            group["betas"] = rule.betas
            group["nesterov"] = rule.nesterov
            updates[p] = rule.update
            by_rule.setdefault(id(rule), []).append(p)
            rule_of_id[id(rule)] = rule

        sizings = [
            (
                rule_of_id[rule_id].sizing,
                params,
                rule_of_id[rule_id].lr
                * rule_of_id[rule_id].warmup_factor(step)
                * eta
                * rule_of_id[rule_id].coeff,
            )
            for rule_id, params in by_rule.items()
        ]
        return updates, sizings


def apply_overrides(
    train_config: TrainConfig, rule_set: RuleSet, overrides: dict
) -> tuple[TrainConfig, RuleSet]:
    """Apply one ``override_args``/branch-spec combo to ``(train_config,
    rule_set.rules)``: bare keys replace a ``TrainConfig`` field;
    ``"rules.<name>.<field>"`` keys replace one field on a *copy* of that
    one named rule (every other rule, and every other field of that rule,
    is untouched); a ``"rules"`` key replaces the entire rule list outright
    (same YAML shape as the top-level 'rules:' config section) -- for a
    branch whose rules don't just tweak the trunk's own (different rule
    names/count/structure entirely, e.g. splitting one rule covering
    several params into one rule per param) -- see ``validate_override``
    for why it can't be combined with a ``"rules.<name>.<field>"`` key in
    the same ``overrides`` dict; validated via exactly that call, as this
    function's first action, so every combo goes through it regardless of
    caller. Neither ``rule_set`` nor ``train_config`` is mutated; the
    returned ``RuleSet`` is fresh and unresolved, even if the combo touched
    no rule (``rule_set``'s own ``_per_param`` cache, if any, is never
    inherited -- resolve the new one again against the new combo's own
    param names/train_steps).

    A rule's ``end`` left ``None`` (``steps: [start, null]``, meaning
    "through train_steps") is never resolved to a concrete value --
    ``resolve``'s own ``check_range`` pass interprets it directly, against
    *this* combo's own (possibly just-overridden) ``train_steps``, not the
    base config's.
    """
    rule_set.validate_override(overrides)

    train_overrides = {}
    rule_field_overrides: dict[str, dict] = {}
    rules_override: list[Rule] | None = None
    for key, value in overrides.items():
        if key == "rules":
            if not isinstance(value, list) or not value:
                raise ValueError(
                    f"'rules' override must be a non-empty list of rule "
                    f"entries, got {value!r}"
                )
            rules_override = []
            names_seen = set()
            for idx, entry in enumerate(value):
                if not isinstance(entry, dict):
                    raise ValueError(
                        f"'rules' override entry {idx} must be a dict, got "
                        f"{type(entry).__name__}"
                    )
                label = str(entry.get("name", f"rules[{idx}]"))
                if label in names_seen:
                    raise ValueError(f"'rules' override: duplicate rule name {label!r}")
                names_seen.add(label)
                rules_override.append(Rule.from_entry(entry, label))
            continue
        parts = key.split(".")
        if len(parts) == 3 and parts[0] == "rules":
            _prefix, rule_name, field_name = parts
            rule_field_overrides.setdefault(rule_name, {})[field_name] = (
                Rule.coerce_field(field_name, value)
            )
        else:
            train_overrides[key] = value

    new_train_config = (
        replace(train_config, **train_overrides) if train_overrides else train_config
    )
    base_rules = rules_override if rules_override is not None else rule_set.rules
    new_rules = (
        [
            (
                replace(rule, **rule_field_overrides[rule.name])
                if rule.name in rule_field_overrides
                else rule
            )
            for rule in base_rules
        ]
        if rule_field_overrides
        else base_rules
    )
    return new_train_config, RuleSet(new_rules)


def load_checkpoint_config(path: str) -> tuple[TrainConfig, RuleSet]:
    """Read a checkpoint-producing script's own ``config.json`` back into
    ``(TrainConfig, RuleSet)`` -- the ``source_path``/``path`` every
    run_branch_compare.py ``runs`` entry forks from. Accepts either shape:

    - run_optim_rules.py's flat one -- ``{**train_fields, "rules": [...]}``.
    - a run_branch_compare.py branch's own nested one -- ``{"train": {...},
      "rules": [...], "branch_name": ..., "override": ..., ...}`` (see
      ``_make_branch_dir``) -- so a branch's own checkpoint (e.g.
      ``.../branches/svdp_p025/``) can itself be used as a further
      run_branch_compare.py job's ``source_path``, re-forked from exactly
      the rules that branch ended up running (e.g. one short KL-matched
      divergent window feeding straight into another, each with its own
      ``branch_specs``, rather than needing run_branch_continue.py's
      single-shared-override re-fork instead).

    Not ``load_compare_job_config`` (a run_branch_compare.py *job* dir's
    own config.json -- a third, different shape again, giving that job's
    *original pre-override* rules; used by run_branch_continue.py, which
    needs the baseline every branch diverged from, not any one branch's
    own already-overridden rules).
    """
    with open(os.path.join(path, FILENAME_CONFIGS), encoding="utf-8") as f:
        payload = json.load(f)
    train_fields = {f.name for f in fields(TrainConfig)}
    train_payload = payload["train"] if "train" in payload else payload
    train_config = TrainConfig(
        **{k: v for k, v in train_payload.items() if k in train_fields}
    )
    return train_config, RuleSet.load_from_payload(payload["rules"])


def load_override_args(path: str) -> list[dict]:
    """Read the optional 'override_args' section and return every override
    combo to run, as a flat, deduped list of ``{key: value}`` dicts (a
    script applies each one via ``apply_overrides``, which does its own
    ``validate_override`` -- this function doesn't need a ``RuleSet`` at
    all, since it never checks a key against one itself).

    'override_args' is either a single dict -- ``{<key>: [values, ...]}``
    -- or a list of such dicts (a bare dict is treated as the singleton
    list ``[override_args]``). Each dict's combos are its own cartesian
    product across its listed keys' values; when 'override_args' is a
    list, every dict's combos are computed separately and then
    concatenated (+deduped) instead of cross-producted together.

    Each ``<key>`` is either a bare ``TrainConfig`` field name, or
    ``"rules.<rule_name>.<field>"`` to sweep one specific rule's field --
    e.g. ``{"seed": [0, 1], "rules.blocks_early.lr": [0.01, 0.03]}`` runs
    all 4 combinations, each with ``seed`` replaced in ``TrainConfig`` and
    the rule named ``"blocks_early"``'s ``lr`` replaced, every other
    rule/field untouched.
    """

    def hashable(value):
        if isinstance(value, dict):
            return tuple(sorted((k, hashable(v)) for k, v in value.items()))
        if isinstance(value, (list, tuple)):
            return tuple(hashable(v) for v in value)
        return value

    with open(path, encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    raw = payload.get("override_args") or {}
    if not isinstance(raw, (dict, list)):
        raise ValueError(
            f"config {path!r} override_args must be a dict or a list of "
            f"dicts, got {type(raw).__name__}"
        )
    specs = [raw] if isinstance(raw, dict) else list(raw)

    combos = []
    seen = set()
    for spec in specs:
        if not isinstance(spec, dict):
            raise ValueError(
                f"config {path!r} override_args list entries must be "
                f"dicts, got {type(spec).__name__}"
            )
        sweep_keys = list(spec.keys())
        value_lists = [spec[key] for key in sweep_keys]
        spec_combos = itertools.product(*value_lists) if sweep_keys else [()]
        for combo in spec_combos:
            overrides = dict(zip(sweep_keys, combo))
            dedup_key = tuple(sorted((k, hashable(v)) for k, v in overrides.items()))
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            combos.append(overrides)
    return combos
