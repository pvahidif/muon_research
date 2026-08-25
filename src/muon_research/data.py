"""Distributed FineWeb dataloader.

Token streams are deterministic: shard files are ``sorted(glob(...))`` and read
front-to-back with no shuffle by default. Same pattern + same files ⇒ same
batch order. Pass ``shard_order_seed`` to permute the shard file list
deterministically (e.g. one permutation per experiment seed).

Use :class:`DistributedDataCursor` when the shard index / byte position must be
checkpointed and restored (branching experiments). The generator wrapper keeps
the older call sites working.
"""

from __future__ import annotations

import copy
import warnings
from pathlib import Path

import torch
import torch.distributed as dist


def _shard_num_tokens(file: Path) -> int:
    header = torch.from_file(str(file), False, 256, dtype=torch.int32)
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    return int(header[2])


def _glob_shard_files(filename_pattern: str) -> list[Path]:
    """Expand a shard glob; supports relative or absolute directory prefixes."""
    pat = Path(filename_pattern)
    root = pat.parent
    if str(root) == ".":
        root = Path.cwd()
    elif not root.is_absolute():
        root = Path.cwd() / root
    files = sorted(root.glob(pat.name))
    assert files, f"no shards matched {filename_pattern!r} under {root}"
    return files


def _load_data_shard(file: Path):
    num_tokens = _shard_num_tokens(file)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())  # avoid bytes->array copy
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens


def _ordered_shard_files(
    filename_pattern: str, *, shard_order_seed: int | None
) -> list[Path]:
    """Sorted shard paths, optionally permuted by ``shard_order_seed``."""
    files = _glob_shard_files(filename_pattern)
    if shard_order_seed is None:
        return files
    g = torch.Generator()
    g.manual_seed(int(shard_order_seed))
    perm = torch.randperm(len(files), generator=g).tolist()
    return [files[i] for i in perm]


class DistributedDataCursor:
    """Deterministic distributed token cursor with ``state_dict`` / ``load_state_dict``.

    ``shard_order_seed=None`` keeps lexicographic shard order. A non-None seed
    applies a deterministic permutation of the shard file list (same seed ⇒
    same order across ranks / resumes).
    """

    def __init__(
        self,
        filename_pattern: str,
        batch_size: int,
        vocab_size: int,
        seq_len: int = 1024,
        shard_order_seed: int | None = None,
    ):
        self.filename_pattern = filename_pattern
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.shard_order_seed = (
            None if shard_order_seed is None else int(shard_order_seed)
        )

        world_size = dist.get_world_size()
        assert batch_size % world_size == 0
        self.world_size = world_size
        self.rank = dist.get_rank()
        self.local_batch_size = batch_size // world_size

        self.files = _ordered_shard_files(
            filename_pattern, shard_order_seed=self.shard_order_seed
        )

        self.file_idx = 0
        self.pos = 0
        self.tokens = _load_data_shard(self.files[self.file_idx])

    def state_dict(self) -> dict:
        return {
            "filename_pattern": self.filename_pattern,
            "batch_size": self.batch_size,
            "vocab_size": self.vocab_size,
            "seq_len": self.seq_len,
            "shard_order_seed": self.shard_order_seed,
            "file_idx": self.file_idx,
            "pos": self.pos,
            # Identity check so resumes fail loudly if the shard list changed.
            "num_files": len(self.files),
            "file_name": self.files[self.file_idx].name,
            "file_names": [f.name for f in self.files],
        }

    def load_state_dict(self, state: dict) -> None:
        if (
            state.get("filename_pattern", self.filename_pattern)
            != self.filename_pattern
        ):
            raise ValueError(
                f"data pattern mismatch: ckpt={state.get('filename_pattern')!r} "
                f"vs live={self.filename_pattern!r}"
            )
        # batch_size/vocab_size/seq_len mismatches only warn, not raise:
        # file_idx/pos are plain token-stream positions, independent of all
        # three, so resuming with a different one is mechanically safe --
        # it just means next_batch() chunks/remaps/reshapes differently
        # from here on, which may or may not be what was intended.
        if int(state["batch_size"]) != self.batch_size:
            warnings.warn(
                f"DistributedDataCursor.load_state_dict: batch_size mismatch "
                f"(ckpt={state['batch_size']} vs live={self.batch_size})"
            )
        if int(state["vocab_size"]) != self.vocab_size:
            warnings.warn(
                f"DistributedDataCursor.load_state_dict: vocab_size mismatch "
                f"(ckpt={state['vocab_size']} vs live={self.vocab_size})"
            )
        if int(state["seq_len"]) != self.seq_len:
            warnings.warn(
                f"DistributedDataCursor.load_state_dict: seq_len mismatch "
                f"(ckpt={state['seq_len']} vs live={self.seq_len})"
            )
        ckpt_order_seed = state.get("shard_order_seed", None)
        if ckpt_order_seed is not None:
            ckpt_order_seed = int(ckpt_order_seed)
        if ckpt_order_seed != self.shard_order_seed:
            raise ValueError(
                f"shard_order_seed mismatch: ckpt={ckpt_order_seed!r} "
                f"vs live={self.shard_order_seed!r}"
            )
        if int(state["num_files"]) != len(self.files):
            raise ValueError(
                f"shard count mismatch: ckpt={state['num_files']} vs live={len(self.files)}"
            )
        ckpt_names = state.get("file_names")
        if ckpt_names is not None:
            live_names = [f.name for f in self.files]
            if list(ckpt_names) != live_names:
                raise ValueError(
                    "shard file order mismatch between checkpoint and live cursor "
                    f"(shard_order_seed={self.shard_order_seed!r})"
                )
        file_idx = int(state["file_idx"])
        pos = int(state["pos"])
        if not 0 <= file_idx < len(self.files):
            raise ValueError(f"file_idx out of range: {file_idx}")
        if self.files[file_idx].name != state["file_name"]:
            raise ValueError(
                f"shard identity mismatch at idx={file_idx}: "
                f"ckpt={state['file_name']!r} vs live={self.files[file_idx].name!r}"
            )
        self.file_idx = file_idx
        self.pos = pos
        self.tokens = _load_data_shard(self.files[self.file_idx])
        if not 0 <= self.pos < len(self.tokens):
            raise ValueError(
                f"pos out of range for shard {self.files[self.file_idx]}: "
                f"pos={self.pos}, len={len(self.tokens)}"
            )

    def clone(self) -> "DistributedDataCursor":
        """An independent cursor at the same position, for exploration that
        must not advance ``self`` (e.g. forking into branches). Cheap: a
        shallow copy sharing ``self.tokens``/``self.files`` (no disk
        re-read) -- safe because ``next_batch``/``_advance_shard`` only
        ever reassign those attributes, never mutate them in place, so the
        clone and ``self`` can't step on each other once either advances.
        """
        return copy.copy(self)

    def _advance_shard(self) -> None:
        self.file_idx += 1
        if self.file_idx >= len(self.files):
            # Match the old generator's failure mode (finite shard list).
            raise StopIteration(
                f"exhausted {len(self.files)} shards for pattern {self.filename_pattern!r}"
            )
        self.tokens = _load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the next ``(inputs, targets)`` and advance the cursor."""
        while self.pos + self.batch_size + 1 >= len(self.tokens):
            self._advance_shard()
        buf = self.tokens[self.pos + self.rank * self.local_batch_size :][
            : self.local_batch_size + 1
        ]
        inputs = (
            buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True)
            % self.vocab_size
        )
        targets = (
            buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True)
            % self.vocab_size
        )
        self.pos += self.batch_size
        return inputs.view(-1, self.seq_len), targets.view(-1, self.seq_len)

    def advance_tokens(self, n_tokens: int) -> None:
        """Skip ahead by ``n_tokens`` without materializing batches on GPU."""
        remaining = max(0, int(n_tokens))
        while remaining > 0:
            while self.pos + self.batch_size + 1 >= len(self.tokens):
                self._advance_shard()
            # Largest stride that keeps ``pos + batch_size + 1 < len(tokens)``.
            room = len(self.tokens) - self.pos - 1
            step = min(remaining, room, self.batch_size)
            if step <= 0:
                self._advance_shard()
                continue
            self.pos += step
            remaining -= step

    def __iter__(self):
        return self

    def __next__(self):
        return self.next_batch()


def distributed_data_generator(
    filename_pattern: str,
    batch_size: int,
    vocab_size: int,
    seq_len: int = 1024,
):
    """Yield ``(inputs, targets)`` forever in a fixed, sequential shard order."""
    cursor = DistributedDataCursor(
        filename_pattern, batch_size, vocab_size, seq_len=seq_len
    )
    while True:
        yield cursor.next_batch()
