"""
Build FineWeb10B shards with a custom-vocab BPE tokenizer.

Mirrors cached_fineweb10B.py's output layout (val + train .bin shards), and
also saves the trained tokenizer. Uses the existing GPT-2-tokenized FineWeb
cache as the source of documents:

  1. Download GPT-2 FineWeb .bin shards (same as cached_fineweb10B.py)
  2. Reconstruct complete documents across shard boundaries (EOT=50256)
  3. Choice-B document-level train/val split (no document in both sets)
  4. Train a byte-level BPE on a random sample of *training* documents only
  5. Retokenize val + train and write new .bin shards + tokenizer.json

Usage:
  python src/muon_research/download_data/cached_fineweb10B_vocab.py            # full 10B, vocab 5000
  python src/muon_research/download_data/cached_fineweb10B_vocab.py 20         # ~2B GPT-2 tokens of source
  python src/muon_research/download_data/cached_fineweb10B_vocab.py --num-chunks 20
  python src/muon_research/download_data/cached_fineweb10B_vocab.py 20 --vocab-size 5000

Original code, but reuses the .bin shard header format and download
conventions from cached_fineweb10B.py (adapted from modded-nanogpt); see
THIRD_PARTY_NOTICES.md at the repo root.
"""

from __future__ import annotations

import argparse
import json
import random
import tempfile
from pathlib import Path

import numpy as np
import tiktoken
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
from tqdm import tqdm

from muon_research.paths import REPO_ROOT

MAGIC = 20240520
HEADER_INTS = 256
GPT2_EOT = 50256  # <|endoftext|>
SHARD_SIZE = 10**8
GPT2_REPO = "kjj0/fineweb10B-gpt2"
EOT_TOKEN = "<|endoftext|>"

DATA_ROOT = REPO_ROOT / "data"
GPT2_DIR = DATA_ROOT / "fineweb10B"


# ---------------------------------------------------------------------------
# Download (same behavior as cached_fineweb10B.py)
# ---------------------------------------------------------------------------
def download_gpt2_shard(fname: str) -> Path:
    GPT2_DIR.mkdir(parents=True, exist_ok=True)
    dest = GPT2_DIR / fname
    if not dest.exists():
        hf_hub_download(
            repo_id=GPT2_REPO,
            filename=fname,
            repo_type="dataset",
            local_dir=str(GPT2_DIR),
        )
    return dest


def gpt2_shard_paths(num_train_chunks: int, download: bool) -> list[Path]:
    names = ["fineweb_val_%06d.bin" % 0]
    names += ["fineweb_train_%06d.bin" % i for i in range(1, num_train_chunks + 1)]
    paths = []
    for name in names:
        if download:
            paths.append(download_gpt2_shard(name))
        else:
            path = GPT2_DIR / name
            if not path.exists():
                raise SystemExit(f"missing GPT-2 shard: {path}")
            paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# GPT-2 shard I/O
# ---------------------------------------------------------------------------
def read_shard_tokens(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        header = np.frombuffer(f.read(HEADER_INTS * 4), dtype=np.int32)
        assert header[0] == MAGIC, f"magic mismatch in {path}"
        assert header[1] == 1, f"unsupported version in {path}"
        ntok = int(header[2])
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert (
        len(tokens) == ntok
    ), f"token count mismatch in {path}: {len(tokens)} vs {ntok}"
    return tokens


def write_datafile(filename: Path, toks: np.ndarray) -> None:
    """Same layout as fineweb.py write_datafile."""
    assert toks.dtype == np.uint16
    assert len(toks) < 2**31
    header = np.zeros(HEADER_INTS, dtype=np.int32)
    header[0] = MAGIC
    header[1] = 1
    header[2] = len(toks)
    print(f"writing {len(toks):,} tokens to {filename}")
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks.tobytes())


# ---------------------------------------------------------------------------
# Document reconstruction (streaming, Choice B)
# ---------------------------------------------------------------------------
def iter_documents_fast(shard_paths: list[Path], enc: tiktoken.Encoding):
    """
    Yield (split, text) for each complete document in the GPT-2 token stream.

    Stream format (from fineweb.py):
        EOT, doc1..., EOT, doc2..., EOT, doc3..., ...

    Choice B: a document is validation iff it finishes entirely inside the GPT-2
    val shard (absolute end position <= len(val_shard)). The document that
    straddles the val/train boundary, and all later documents, are training.
    No document is assigned to both splits.
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

    def emit(end_abs: int):
        nonlocal doc_start_abs
        assert doc_start_abs is not None
        body = flush_body()
        text = enc.decode_bytes(body).decode("utf-8", errors="replace")
        split = "val" if end_abs <= val_end else "train"
        return split, text

    for path in shard_paths:
        tokens = read_shard_tokens(path)
        n = len(tokens)
        # Relative indices of EOTs within this shard
        eot_rel = np.flatnonzero(tokens == GPT2_EOT)
        cursor = 0
        for er in eot_rel:
            er = int(er)
            # body between cursor and this EOT
            if er > cursor:
                body_chunks.append(tokens[cursor:er])
            eot_abs = abs_pos + er
            if doc_start_abs is not None:
                yield emit(eot_abs)
            doc_start_abs = eot_abs
            body_chunks = []
            cursor = er + 1
        # remainder of shard after last EOT (or whole shard if no EOT)
        if cursor < n:
            if doc_start_abs is None:
                # shard begins mid-document without a leading EOT in this view;
                # mark start at current abs_pos + cursor
                doc_start_abs = abs_pos + cursor
            body_chunks.append(tokens[cursor:n])
        abs_pos += n

    if doc_start_abs is not None:
        yield emit(abs_pos)


# ---------------------------------------------------------------------------
# Tokenizer training + retokenization
# ---------------------------------------------------------------------------
def collect_tokenizer_sample(
    shard_paths: list[Path],
    sample_docs: int,
    sample_chars: int,
    seed: int,
    sample_path: Path,
) -> dict:
    """
    Stream documents once: reservoir-sample training docs into sample_path
    (one document per line, JSON-encoded), and return counts.
    """
    enc = tiktoken.get_encoding("gpt2")
    rng = random.Random(seed)

    n_val = 0
    n_train = 0
    # Reservoir sample of training document texts
    reservoir: list[str] = []

    for split, text in tqdm(
        iter_documents_fast(shard_paths, enc),
        desc="pass1: scan+sample",
        unit="doc",
    ):
        if split == "val":
            n_val += 1
            continue
        n_train += 1
        # Reservoir sampling over training docs
        if len(reservoir) < sample_docs:
            reservoir.append(text)
        else:
            j = rng.randint(0, n_train - 1)
            if j < sample_docs:
                reservoir[j] = text

    # Cap by characters while preserving randomness of order
    rng.shuffle(reservoir)
    chosen: list[str] = []
    total_chars = 0
    for t in reservoir:
        chosen.append(t)
        total_chars += len(t)
        if total_chars >= sample_chars:
            break

    with open(sample_path, "w", encoding="utf-8") as f:
        for t in chosen:
            # JSON string per line preserves newlines inside documents
            f.write(json.dumps(t, ensure_ascii=False))
            f.write("\n")

    stats = {
        "val_docs": n_val,
        "train_docs": n_train,
        "sample_docs": len(chosen),
        "sample_chars": total_chars,
    }
    print(
        f"documents: val={n_val:,} train={n_train:,}; "
        f"tokenizer sample: {len(chosen):,} docs / {total_chars:,} chars"
    )
    return stats


def train_tokenizer(sample_path: Path, vocab_size: int) -> Tokenizer:
    def iterator():
        with open(sample_path, "r", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)

    n_lines = sum(1 for _ in open(sample_path, "r", encoding="utf-8"))

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[EOT_TOKEN],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(iterator(), trainer=trainer, length=n_lines)

    eot_id = tokenizer.token_to_id(EOT_TOKEN)
    assert eot_id is not None, f"{EOT_TOKEN} missing from vocabulary"
    got = tokenizer.get_vocab_size()
    print(f"trained tokenizer: vocab_size={got} eot_id={eot_id}")
    if got != vocab_size:
        print(
            f"warning: requested vocab_size={vocab_size} but trainer produced {got} "
            "(corpus may be too small for the requested size)"
        )
    return tokenizer


def encode_doc(tokenizer: Tokenizer, text: str, eot_id: int) -> np.ndarray:
    """Prefix document with EOT (same convention as fineweb.py)."""
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    out = np.empty(1 + len(ids), dtype=np.uint16)
    out[0] = np.uint16(eot_id)
    if ids:
        arr = np.asarray(ids, dtype=np.int64)
        assert (arr >= 0).all() and (arr < 2**16).all(), "token id exceeds uint16"
        out[1:] = arr.astype(np.uint16)
    return out


def retokenize_and_write(
    shard_paths: list[Path],
    tokenizer: Tokenizer,
    out_dir: Path,
    shard_size: int,
) -> dict:
    """
    Stream documents a second time, encode with the custom tokenizer, and write
    val_000000.bin plus train_000001.bin, train_000002.bin, ...
    """
    enc = tiktoken.get_encoding("gpt2")
    eot_id = tokenizer.token_to_id(EOT_TOKEN)
    assert eot_id is not None

    # Per-split buffers. Val is always a single shard (index 0), possibly shorter
    # than shard_size — matching the GPT-2 cache convention when content is shorter,
    # but we still pack up to shard_size and spill to additional val shards if needed.
    state = {
        "val": {
            "buf": np.empty((shard_size,), dtype=np.uint16),
            "count": 0,
            "index": 0,
            "tokens": 0,
            "docs": 0,
            "shards": 0,
        },
        "train": {
            "buf": np.empty((shard_size,), dtype=np.uint16),
            "count": 0,
            "index": 1,
            "tokens": 0,
            "docs": 0,
            "shards": 0,
        },
    }

    def flush(split: str) -> None:
        st = state[split]
        if st["count"] == 0:
            return
        if split == "val":
            fname = out_dir / f"fineweb_val_{st['index']:06d}.bin"
        else:
            fname = out_dir / f"fineweb_train_{st['index']:06d}.bin"
        write_datafile(fname, st["buf"][: st["count"]].copy())
        st["shards"] += 1
        st["index"] += 1
        st["count"] = 0

    for split, text in tqdm(
        iter_documents_fast(shard_paths, enc),
        desc="pass2: retokenize",
        unit="doc",
    ):
        st = state[split]
        toks = encode_doc(tokenizer, text, eot_id)
        st["docs"] += 1
        st["tokens"] += len(toks)
        offset = 0
        while offset < len(toks):
            space = shard_size - st["count"]
            take = min(space, len(toks) - offset)
            st["buf"][st["count"] : st["count"] + take] = toks[offset : offset + take]
            st["count"] += take
            offset += take
            if st["count"] == shard_size:
                flush(split)

    flush("val")
    flush("train")
    return {
        "val": {k: state["val"][k] for k in ("docs", "tokens", "shards")},
        "train": {k: state["train"][k] for k in ("docs", "tokens", "shards")},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "num_chunks_positional",
        nargs="?",
        type=int,
        default=None,
        metavar="num_chunks",
        help="number of GPT-2 train shards to download/use (default 103 = full 10B)",
    )
    p.add_argument(
        "--num-chunks",
        type=int,
        default=None,
        help="same as positional num_chunks (default 103 = full 10B)",
    )
    p.add_argument(
        "--vocab-size",
        type=int,
        default=5000,
        help="BPE vocab size including specials",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="output directory (default: data/fineweb10B_v{vocab_size})",
    )
    p.add_argument(
        "--shard-size",
        type=int,
        default=SHARD_SIZE,
        help="tokens per output shard",
    )
    p.add_argument(
        "--tokenizer-sample-docs",
        type=int,
        default=200_000,
        help="max training documents to reservoir-sample for tokenizer training",
    )
    p.add_argument(
        "--tokenizer-sample-chars",
        type=int,
        default=200_000_000,
        help="approx max characters used for tokenizer training",
    )
    p.add_argument(
        "--seed", type=int, default=0, help="RNG seed for tokenizer document sample"
    )
    p.add_argument(
        "--skip-download",
        action="store_true",
        help="require GPT-2 shards already present under data/fineweb10B",
    )
    args = p.parse_args(argv)
    if args.num_chunks is not None and args.num_chunks_positional is not None:
        if args.num_chunks != args.num_chunks_positional:
            p.error("conflicting values for num_chunks and --num-chunks")
    args.num_chunks = (
        args.num_chunks
        if args.num_chunks is not None
        else (
            args.num_chunks_positional
            if args.num_chunks_positional is not None
            else 103
        )
    )
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else DATA_ROOT / f"fineweb10B_v{args.vocab_size}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"source GPT-2 shards: {GPT2_DIR}")
    print(f"output directory:    {out_dir}")
    print(f"vocab_size:          {args.vocab_size}")
    print(f"num_train_chunks:    {args.num_chunks}")

    paths = gpt2_shard_paths(args.num_chunks, download=not args.skip_download)

    with tempfile.TemporaryDirectory(prefix="fw_tok_sample_") as tmp:
        sample_path = Path(tmp) / "sample.jsonl"
        scan_stats = collect_tokenizer_sample(
            paths,
            sample_docs=args.tokenizer_sample_docs,
            sample_chars=args.tokenizer_sample_chars,
            seed=args.seed,
            sample_path=sample_path,
        )
        if scan_stats["train_docs"] == 0:
            raise SystemExit(
                "no training documents reconstructed; need at least one train shard"
            )
        if scan_stats["val_docs"] == 0:
            raise SystemExit("no validation documents reconstructed")

        tokenizer = train_tokenizer(sample_path, vocab_size=args.vocab_size)

    tok_path = out_dir / "tokenizer.json"
    tokenizer.save(str(tok_path))
    print(f"saved tokenizer → {tok_path}")

    stats = retokenize_and_write(paths, tokenizer, out_dir, args.shard_size)

    eot_id = tokenizer.token_to_id(EOT_TOKEN)
    meta = {
        "vocab_size": tokenizer.get_vocab_size(),
        "requested_vocab_size": args.vocab_size,
        "eot_token": EOT_TOKEN,
        "eot_id": eot_id,
        "source_repo": GPT2_REPO,
        "source_train_chunks": args.num_chunks,
        "split": "choice_B_document_level",
        "split_note": (
            "Documents that finish inside the GPT-2 val shard are validation; "
            "the document straddling the val/train boundary and all later documents "
            "are training. Tokenizer trained on a reservoir sample of training documents only."
        ),
        "tokenizer_sample_docs": args.tokenizer_sample_docs,
        "tokenizer_sample_chars": args.tokenizer_sample_chars,
        "seed": args.seed,
        "shard_size": args.shard_size,
        "scan": scan_stats,
        "val": stats["val"],
        "train": stats["train"],
        "tokenizer_file": "tokenizer.json",
    }
    meta_path = out_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    print(f"saved meta → {meta_path}")
    print(
        f"done: val_tokens={stats['val']['tokens']:,} "
        f"train_tokens={stats['train']['tokens']:,} "
        f"vocab={meta['vocab_size']} eot_id={eot_id}"
    )


if __name__ == "__main__":
    main()
