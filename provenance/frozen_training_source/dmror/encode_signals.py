"""Frozen local LLM embeddings, risk-segment pooling, auditable offline cache.

No data is sent to a service. Download/model access is a separate user action.
Projection reduces hidden states to portable 64-dimensional cached features;
the projection is frozen, label-independent, and shared across all splits.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .data import audit_dataset, sha256, write_json


def encode(paths, model_path, output, device="cuda:0", batch_size=8, max_length=256, dim=64):
    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.manual_seed(731)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModel.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32)
    model.to(device).eval()
    hidden = model.config.hidden_size
    rng = np.random.default_rng(731)
    projection = rng.normal(0, 1 / np.sqrt(hidden), (hidden, dim)).astype(np.float32)
    proj = torch.tensor(projection, device=device)
    all_texts, text_index, dataset_records = [], {}, {}
    for dataset in paths:
        dataset = Path(dataset)
        records = [json.loads(line) for line in (dataset / "signals.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        dataset_records[dataset] = records
        for record in records:
            if record["text"] not in text_index:
                text_index[record["text"]] = len(all_texts)
                all_texts.append(record["text"])
    cache_path = output / "embedding_cache.npz"
    embeddings = np.zeros((len(all_texts), dim), np.float32)
    prefix = "Encode this industrial risk evidence for future supply-chain risk prediction. Evidence: "
    with torch.inference_mode():
        for start in range(0, len(all_texts), batch_size):
            chunk = all_texts[start:start + batch_size]
            tokens = tokenizer([prefix + text for text in chunk], return_tensors="pt", padding=True, truncation=True, max_length=max_length, return_offsets_mapping=True)
            offsets = tokens.pop("offset_mapping").to(device)
            tokens = {key: value.to(device) for key, value in tokens.items()}
            states = model(**tokens, use_cache=False).last_hidden_state.float()
            mask = tokens["attention_mask"].clone()
            mask *= (offsets[..., 0] >= len(prefix)).to(mask.dtype)
            # In pathological tokenization/truncation cases use valid tokens.
            empty = mask.sum(1) == 0
            mask[empty] = tokens["attention_mask"][empty]
            pooled = (states * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
            vectors = torch.nn.functional.normalize(pooled @ proj, dim=-1)
            embeddings[start:start + len(chunk)] = vectors.cpu().numpy()
            if start % (batch_size * 10) == 0:
                print(f"Encoded {start + len(chunk)}/{len(all_texts)} evidence segments", flush=True)
    np.savez_compressed(cache_path, embeddings=embeddings, projection=projection)
    (output / "texts.jsonl").write_text("".join(json.dumps({"cache_id": i, "text": text}, ensure_ascii=False) + "\n" for i, text in enumerate(all_texts)), encoding="utf-8")
    config = Path(model_path) / "config.json"
    weight_files = list(Path(model_path).glob("*.safetensors")) or list(Path(model_path).glob("*.bin"))
    provenance = {"encoder": "frozen_local_LLM", "model_identifier": Path(model_path).name, "model_config_sha256": sha256(config), "weight_files": [{"name": file.name, "bytes": file.stat().st_size, "sha256": sha256(file)} for file in weight_files], "pooling": "masked mean of evidence segment last hidden states, character offsets identify evidence tokens", "projection_seed": 731, "projection_dim": dim, "original_hidden_dim": hidden, "max_length": max_length, "prompt_prefix": prefix, "cache_sha256": sha256(cache_path), "torch_version": torch.__version__, "device": device, "unique_evidence_segments": len(all_texts), "limitations": ["Frozen feature extraction rather than end-to-end LLM fine-tuning.", "Signal fields in bundled datasets are rule-extracted or simulated; the separate extractor supports local LLM JSON extraction.", "One evidence unit is encoded per signal; full entity-neighborhood prompts are supported by extract_signals but not encoded in these cached runs."]}
    write_json(output / "provenance.json", provenance)
    for dataset, records in dataset_records.items():
        lookup = {int(record["signal_id"]): text_index[record["text"]] for record in records}
        with np.load(dataset / "dataset.npz", allow_pickle=False) as source:
            data = {key: source[key].copy() for key in source.files}
        indices = data["memory_indices"]
        result = np.zeros((*indices.shape, dim), np.float32)
        for signal_id, cache_id in lookup.items():
            result[indices == signal_id] = embeddings[cache_id]
        data["signals"] = result
        np.savez_compressed(dataset / "dataset.npz", **data)
        metadata = json.loads((dataset / "metadata.json").read_text(encoding="utf-8"))
        metadata["text_encoder"] = provenance
        metadata["statistics"] = audit_dataset(dataset / "dataset.npz")
        write_json(dataset / "metadata.json", metadata)
        print(f"Updated {dataset.name}: {metadata['statistics']['sha256']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default="data/llm_cache")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()
    encode(args.datasets, args.model, args.output, args.device, args.batch_size, args.max_length)
