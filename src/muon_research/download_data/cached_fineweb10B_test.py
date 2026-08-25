"""
Verify fineweb10B_v{vocab} against the GPT-2 fineweb10B cache.

Reads the first val and first train shard from each directory, decodes documents
with the corresponding tokenizer, and checks that the custom-vocab document
prefix matches the GPT-2-decoded Choice-B document stream.

  GPT-2 side:  data/fineweb10B/          + tiktoken gpt2
  Custom side: data/fineweb10B_v{V}/     + tokenizer.json

Only documents closed by an EOT *inside* the files we read are compared (the
trailing possibly-partial document at each file end is ignored).

Usage:
  python src/muon_research/download_data/cached_fineweb10B_test.py
  python src/muon_research/download_data/cached_fineweb10B_test.py --vocab-size 5000
  python src/muon_research/download_data/cached_fineweb10B_test.py --vocab-size 5000 --max-docs 100

Original code, but reuses the .bin shard header format from
cached_fineweb10B.py (adapted from modded-nanogpt); see
THIRD_PARTY_NOTICES.md at the repo root.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tiktoken
from tokenizers import Tokenizer
from tqdm import tqdm

from muon_research.download_data.cached_fineweb10B_vocab import (
    DATA_ROOT,
    GPT2_DIR,
    GPT2_EOT,
    HEADER_INTS,
    MAGIC,
)

EOT_TOKEN = "<|endoftext|>"


def read_shard_tokens(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        header = np.frombuffer(f.read(HEADER_INTS * 4), dtype=np.int32)
        assert header[0] == MAGIC, f"magic mismatch in {path}"
        assert header[1] == 1, f"unsupported version in {path}"
        ntok = int(header[2])
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, f"token count mismatch in {path}"
    return tokens


def iter_eot_closed_docs_gpt2(shard_paths: list[Path]):
    """
    Yield (split, body_token_ids) for documents closed by a following EOT.
    Choice B split relative to the first shard (val) length.
    Does not emit a trailing open document at EOF.
    """
    val_end = int(read_shard_tokens(shard_paths[0]).shape[0])
    body_chunks: list[np.ndarray] = []
    doc_start_abs: int | None = None
    abs_pos = 0

    def flush_body() -> list[int]:
        nonlocal body_chunks
        if not body_chunks:
            out: list[int] = []
        elif len(body_chunks) == 1:
            out = body_chunks[0].tolist()
        else:
            out = np.concatenate(body_chunks).tolist()
        body_chunks = []
        return out

    for path in shard_paths:
        tokens = read_shard_tokens(path)
        n = len(tokens)
        eot_rel = np.flatnonzero(tokens == GPT2_EOT)
        cursor = 0
        for er in eot_rel:
            er = int(er)
            if er > cursor:
                body_chunks.append(tokens[cursor:er])
            eot_abs = abs_pos + er
            if doc_start_abs is not None:
                body = flush_body()
                split = "val" if eot_abs <= val_end else "train"
                yield split, body
            else:
                body_chunks = []
            doc_start_abs = eot_abs
            cursor = er + 1
        if cursor < n:
            if doc_start_abs is None:
                doc_start_abs = abs_pos + cursor
            body_chunks.append(tokens[cursor:n])
        abs_pos += n


def iter_eot_closed_docs_custom(
    shard_paths: list[Path], tokenizer: Tokenizer, eot_id: int
):
    """Yield document texts closed by a following EOT; no EOF emit."""
    body_chunks: list[np.ndarray] = []
    open_doc = False

    def flush_body() -> list[int]:
        nonlocal body_chunks
        if not body_chunks:
            out: list[int] = []
        elif len(body_chunks) == 1:
            out = body_chunks[0].tolist()
        else:
            out = np.concatenate(body_chunks).tolist()
        body_chunks = []
        return out

    for path in shard_paths:
        tokens = read_shard_tokens(path)
        n = len(tokens)
        eot_rel = np.flatnonzero(tokens == np.uint16(eot_id))
        cursor = 0
        for er in eot_rel:
            er = int(er)
            if er > cursor:
                body_chunks.append(tokens[cursor:er])
            if open_doc:
                body = flush_body()
                yield tokenizer.decode(body)
            open_doc = True
            body_chunks = []
            cursor = er + 1
        if cursor < n:
            if not open_doc:
                open_doc = True
            body_chunks.append(tokens[cursor:n])


def collect_custom_docs(
    path: Path, tokenizer: Tokenizer, eot_id: int, max_docs: int | None
) -> list[str]:
    out: list[str] = []
    for text in tqdm(
        iter_eot_closed_docs_custom([path], tokenizer, eot_id),
        desc=f"custom {path.name}",
        unit="doc",
    ):
        out.append(text)
        if max_docs is not None and len(out) >= max_docs:
            break
    return out


def collect_gpt2_prefix(
    shard_paths: list[Path],
    enc: tiktoken.Encoding,
    n_val_needed: int,
    n_train_needed: int,
) -> tuple[list[str], list[str]]:
    """Decode GPT-2 stream only until we have enough closed val/train docs."""
    val_docs: list[str] = []
    train_docs: list[str] = []
    for split, body in tqdm(
        iter_eot_closed_docs_gpt2(shard_paths),
        desc="gpt2 val+train",
        unit="doc",
    ):
        if split == "val":
            if len(val_docs) < n_val_needed:
                text = enc.decode_bytes(body).decode("utf-8", errors="replace")
                val_docs.append(text)
        elif len(train_docs) < n_train_needed:
            text = enc.decode_bytes(body).decode("utf-8", errors="replace")
            train_docs.append(text)
        if len(val_docs) >= n_val_needed and len(train_docs) >= n_train_needed:
            break
    return val_docs, train_docs


def first_mismatch(a: list[str], b: list[str]) -> tuple[int | None, str]:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            ca, cb = a[i], b[i]
            return i, (
                f"doc[{i}] differs "
                f"(len {len(ca)} vs {len(cb)}; "
                f"gpt2[:80]={ca[:80]!r} custom[:80]={cb[:80]!r})"
            )
    return None, f"match over {n} docs"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--vocab-size", type=int, default=5000)
    p.add_argument("--gpt2-dir", type=str, default=str(GPT2_DIR))
    p.add_argument(
        "--custom-dir",
        type=str,
        default=None,
        help="custom vocab directory (default: data/fineweb10B_v{vocab_size})",
    )
    p.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="only compare the first N closed documents per split",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gpt2_dir = Path(args.gpt2_dir)
    custom_dir = (
        Path(args.custom_dir)
        if args.custom_dir
        else DATA_ROOT / f"fineweb10B_v{args.vocab_size}"
    )

    gpt2_val = gpt2_dir / "fineweb_val_000000.bin"
    gpt2_train = gpt2_dir / "fineweb_train_000001.bin"
    custom_val = custom_dir / "fineweb_val_000000.bin"
    custom_train = custom_dir / "fineweb_train_000001.bin"
    tok_path = custom_dir / "tokenizer.json"

    for p in (gpt2_val, gpt2_train, custom_val, custom_train, tok_path):
        if not p.exists():
            print(f"ERROR: missing {p}", file=sys.stderr)
            return 1

    print(f"gpt2 dir:   {gpt2_dir}")
    print(f"custom dir: {custom_dir}")

    enc = tiktoken.get_encoding("gpt2")
    tokenizer = Tokenizer.from_file(str(tok_path))
    eot_id = tokenizer.token_to_id(EOT_TOKEN)
    if eot_id is None:
        meta_path = custom_dir / "meta.json"
        if meta_path.exists():
            eot_id = int(json.loads(meta_path.read_text())["eot_id"])
        else:
            print(
                f"ERROR: {EOT_TOKEN} not in tokenizer and no meta.json", file=sys.stderr
            )
            return 1
    print(f"custom vocab_size={tokenizer.get_vocab_size()} eot_id={eot_id}")

    # Decode custom first (smaller work if --max-docs; tells us how many GPT-2 docs we need)
    custom_val_docs = collect_custom_docs(custom_val, tokenizer, eot_id, args.max_docs)
    custom_train_docs = collect_custom_docs(
        custom_train, tokenizer, eot_id, args.max_docs
    )
    print(
        f"custom closed docs: val={len(custom_val_docs)} train={len(custom_train_docs)}"
    )
    if not custom_val_docs or not custom_train_docs:
        print("ERROR: no closed documents found in a custom shard", file=sys.stderr)
        return 1

    gpt2_val_docs, gpt2_train_docs = collect_gpt2_prefix(
        [gpt2_val, gpt2_train],
        enc,
        n_val_needed=len(custom_val_docs),
        n_train_needed=len(custom_train_docs),
    )
    print(
        f"gpt2 closed docs collected: val={len(gpt2_val_docs)} train={len(gpt2_train_docs)}"
    )

    ok = True
    if len(gpt2_val_docs) < len(custom_val_docs):
        print(
            f"ERROR: need {len(custom_val_docs)} GPT-2 val docs, got {len(gpt2_val_docs)}",
            file=sys.stderr,
        )
        ok = False
    else:
        idx, msg = first_mismatch(gpt2_val_docs, custom_val_docs)
        if idx is not None:
            print(f"VAL FAIL: {msg}")
            ok = False
        else:
            print(f"VAL OK: {msg}")

    if len(gpt2_train_docs) < len(custom_train_docs):
        print(
            f"ERROR: need {len(custom_train_docs)} GPT-2 train docs from first "
            f"shards, got {len(gpt2_train_docs)}. Custom first train shard may "
            f"contain more documents than GPT-2 train_000001 can close.",
            file=sys.stderr,
        )
        ok = False
    else:
        idx, msg = first_mismatch(gpt2_train_docs, custom_train_docs)
        if idx is not None:
            print(f"TRAIN FAIL: {msg}")
            ok = False
        else:
            print(f"TRAIN OK: {msg}")

    if ok:
        print("ALL CHECKS PASSED")
        return 0
    print("CHECKS FAILED", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
