"""Optional local LLM extraction with evidence-span and entity validation.

Input JSONL: document_id, text, timestamp, optional entities (id/name),
entity_description, neighborhood_summary. Output never invents a timestamp;
each signal inherits its document timestamp, with literal supporting evidence.
Invalid or unsupported model outputs go to a separate audit log.
"""
import argparse
import json
from pathlib import Path

from .data import sha256, write_json

SYSTEM = '''Extract industrial risk evidence from the supplied text. Return ONLY a JSON object
with a "signals" list. Each signal must contain event, entity_id, category,
uncertainty (0 to 1), sentiment (-1 to 1), confidence (0 to 1), evidence.
Use only entity IDs in the supplied entity list. Evidence must be an exact literal
substring of the supplied text. Do not predict future labels or fabricate dates.
If no supported risk event is present, return {"signals": []}.'''


def validate(raw, document):
    left, right = raw.find("{"), raw.rfind("}")
    if left < 0 or right < left:
        raise ValueError("No JSON object")
    parsed = json.loads(raw[left:right + 1])
    permitted = {str(entity["id"]) for entity in document.get("entities", [])}
    if not isinstance(parsed.get("signals"), list):
        raise ValueError("signals is not a list")
    signals = []
    for item in parsed["signals"]:
        if str(item.get("entity_id")) not in permitted:
            raise ValueError("Entity not in supplied alignment vocabulary")
        if not item.get("evidence") or item["evidence"] not in document["text"]:
            raise ValueError("Evidence is not a literal document substring")
        for field, lower, upper in [("uncertainty", 0, 1), ("sentiment", -1, 1), ("confidence", 0, 1)]:
            if not isinstance(item.get(field), (float, int)) or not lower <= item[field] <= upper:
                raise ValueError(f"Invalid {field}")
        for field in ("event", "category"):
            if not isinstance(item.get(field), str) or not item[field].strip():
                raise ValueError(f"Missing {field}")
        item["document_id"] = document["document_id"]
        item["timestamp"] = document["timestamp"]
        item["extraction_kind"] = "local_LLM_unreviewed"
        signals.append(item)
    return signals


def extract(input_path, output, model_path, device, limit=0, max_new_tokens=256):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32).to(device).eval()
    records = [json.loads(line) for line in Path(input_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit:
        records = records[:limit]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    accepted, invalid = [], []
    raw_records = []
    for number, document in enumerate(records):
        prompt = tokenizer.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": json.dumps(document, ensure_ascii=False)}], tokenize=False, add_generation_prompt=True)
        tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536).to(device)
        with torch.inference_mode():
            result = model.generate(**tokens, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        raw = tokenizer.decode(result[0, tokens["input_ids"].shape[1]:], skip_special_tokens=True)
        raw_records.append({"document_id": document["document_id"], "output": raw})
        try:
            accepted.extend(validate(raw, document))
        except (ValueError, TypeError, KeyError) as error:
            invalid.append({"document_id": document["document_id"], "error": str(error), "output": raw})
        print(f"Extracted {number + 1}/{len(records)}; accepted signals={len(accepted)} invalid documents={len(invalid)}", flush=True)
    for filename, rows in [("signals.jsonl", accepted), ("invalid.jsonl", invalid), ("raw_outputs.jsonl", raw_records)]:
        (output / filename).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    write_json(output / "manifest.json", {"input_sha256": sha256(input_path), "model": Path(model_path).name, "documents": len(records), "accepted_signals": len(accepted), "invalid_documents": len(invalid), "status": "machine_extracted_not_human_validated", "training_use": "example only; not used to create future labels or included in cached benchmark features"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    extract(args.input, args.output, args.model, args.device, args.limit, args.max_new_tokens)
