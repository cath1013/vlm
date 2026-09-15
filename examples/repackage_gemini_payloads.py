"""Repackage saved Gemini payloads as equivalent OpenAI Chat Completions requests.

The scene JSON, system instructions, user prompt, and response schema are copied
without changing their meaning.  This permits an LLM-only A/B test: trajectory,
geometry, reranking, and ground truth stay fixed.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from traffic_llm import providers  # noqa: E402


def schema_to_json_schema(value: Any) -> Any:
    """Convert the Gemini/OpenAPI schema dialect back to ordinary JSON Schema."""
    if isinstance(value, list):
        return [schema_to_json_schema(v) for v in value]
    if not isinstance(value, dict):
        return value
    out = {}
    for key, child in value.items():
        if key == "propertyOrdering":
            continue
        if key == "type" and isinstance(child, str):
            out[key] = child.lower()
        else:
            out[key] = schema_to_json_schema(child)
    # Gemini's OpenAPI dialect does not retain JSON Schema's closed-object
    # marker.  OpenAI strict structured outputs requires it on every object.
    if out.get("type") == "object":
        out["additionalProperties"] = False
    return out


def convert(source: Path, destination: Path, model: str) -> None:
    body = json.loads(source.read_text(encoding="utf-8"))
    system = "\n\n".join(
        part["text"] for part in body["systemInstruction"]["parts"]
        if part.get("text")
    )
    blocks = [
        part["text"]
        for content in body["contents"]
        for part in content.get("parts", [])
        if part.get("text")
    ]
    generation = body.get("generationConfig", {})
    schema = schema_to_json_schema(generation["responseSchema"])
    converted = providers.build_request(
        "openai",
        system=system,
        blocks=blocks,
        schema=schema,
        model=model,
        max_tokens=int(generation.get("maxOutputTokens", 32000)),
        effort="high",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(converted, ensure_ascii=False, indent=1) + "\n",
                           encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--relative-payload", action="append", required=True,
                    help="Path relative to --source-root; may be repeated.")
    args = ap.parse_args(argv)
    source_root, out = Path(args.source_root), Path(args.out)
    written = []
    for rel_text in args.relative_payload:
        rel = Path(rel_text)
        source = source_root / rel
        if not source.is_file():
            raise SystemExit(f"payload not found: {source}")
        convert(source, out / rel, args.model)
        # ask_llm.py scores beside the payload, so retain the corresponding GT.
        label = source.stem.removeprefix("llm_payload_")
        gt = source.parent / f"ground_truth_{label}.json"
        shutil.copy2(gt, out / rel.parent / gt.name)
        written.append(str(rel))
    (out / "ab_manifest.json").write_text(json.dumps({
        "source_root": str(source_root), "provider": "openai", "model": args.model,
        "payloads": written,
        "invariant": "Only provider request wrapper/model changed.",
    }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"written": len(written), "out": str(out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
