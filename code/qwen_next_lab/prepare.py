"""Bounded data preparation. Writes uint16 byte IDs; never downloads all OpenWebText."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np

SHAKESPEARE = "https://raw.githubusercontent.com/karpathy/char-rnn/6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e/data/tinyshakespeare/input.txt"
OWT_REVISION = "433fe0f44ed7894fea29c08b3202aa348ccc6369"
OWT = f"https://huggingface.co/datasets/Skylion007/openwebtext/resolve/{OWT_REVISION}/plain_text/train/0000.parquet"


def encode(text):
    return np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype("<u2")


def document_split(text):
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return digest, "val" if int(digest[:8], 16) % 20 == 0 else "train"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["shakespeare", "openwebtext"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-bytes", type=int, default=20_000_000)
    parser.add_argument("--val-bytes", type=int, default=1_000_000)
    args = parser.parse_args()
    if min(args.train_bytes, args.val_bytes) < 2048:
        parser.error("Data budgets must be >= 2048 bytes")
    args.out.mkdir(parents=True, exist_ok=False)
    metadata = {"source": args.source, "tokenizer": "utf8-bytes+eod256", "dtype": "<u2"}
    if args.source == "shakespeare":
        with urlopen(SHAKESPEARE, timeout=60) as response:
            raw = response.read(2_000_001)
        if len(raw) > 2_000_000:
            raise ValueError("Unexpected Shakespeare file size")
        text = raw.decode("utf-8")
        split = int(0.9 * len(text))
        for name, part in (("train", text[:split]), ("val", text[split:])):
            np.append(encode(part), 256).astype("<u2").tofile(args.out / f"{name}.bin")
        metadata.update(url=SHAKESPEARE, raw_sha256=hashlib.sha256(raw).hexdigest(),
                        split="contiguous first 90% / last 10%; no crossing windows")
    else:
        from datasets import load_dataset
        stream = load_dataset("parquet", data_files=OWT, split="train", streaming=True)
        limits = {"train": args.train_bytes, "val": args.val_bytes}
        counts, docs, seen = dict(train=0, val=0), dict(train=0, val=0), set()
        with (args.out / "train.bin").open("wb") as train, (args.out / "val.bin").open("wb") as val:
            files = {"train": train, "val": val}
            for row in stream:
                # Bound single-document processing; split and deduplicate after truncation.
                text = row["text"][:32_000].strip()
                digest, split = document_split(text)
                if not text or digest in seen or counts[split] >= limits[split]:
                    continue
                seen.add(digest)
                ids = encode(text)[:limits[split] - counts[split]]
                ids.tofile(files[split])
                np.array([256], dtype="<u2").tofile(files[split])
                counts[split] += len(ids)
                docs[split] += 1
                if sum(docs.values()) % 500 == 0:
                    print({"bytes": counts, "documents": docs}, flush=True)
                if all(counts[s] >= limits[s] for s in limits):
                    break
        if any(counts[s] < limits[s] for s in limits):
            raise RuntimeError("Pinned first shard exhausted. Use smaller budgets and a NEW output directory.")
        metadata.update(url=OWT, revision=OWT_REVISION, bytes=counts, documents=docs,
                        split="sha256(document) modulo 20; 0=val; exact dedup; first shard only")
    metadata["files"] = {}
    for split in ("train", "val"):
        raw = (args.out / f"{split}.bin").read_bytes()
        metadata["files"][split] = {"tokens": len(raw) // 2, "sha256": hashlib.sha256(raw).hexdigest()}
    (args.out / "meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
