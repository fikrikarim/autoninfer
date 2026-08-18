"""DSpark speculator ingestion for the Qwen3.6-27B experimental lane.

Autoninfer backlog #1 (H6, docs/autoninfer/inspiration.md): the DSpark
speculator (RadixArk/Qwen3.8-27B-DSpark) is SpecForge's DFlash
block-diffusion drafter - five full-attention GQA layers (40 Q / 8 KV,
head 128), standard causal attention with a PERSISTENT draft KV cache,
YaRN RoPE (factor 32, original max 8192) - plus two heads:

  * a vanilla Markov head (low-rank bigram bias: ``logits + w2(w1[prev])``,
    rank 256) and
  * an AcceptRatePredictor confidence head (one linear per draft position
    over ``hidden + markov rank``) that enables adaptive block length.

The drafter is conditioned on target auxiliary residual features at target
layers 4/16/28/40/52 (concat -> ``fc`` -> context norm) and uses the
target's token embedding; it has none of its own. ``block_size = 7`` is the
total block row count: one anchor plus six proposals (verify width 8).

This module ingests the published HF checkpoint (a single
``model.safetensors``, 62 BF16 tensors, 1.36B parameters, 2.72 GiB) into
the experimental DSpark section artifact - an NInfer v2 object directory
holding the drafter weights in exact BF16 (identity ingestion, the
published "unquantized draft" profile). It owns the DSpark config
validation, the complete tensor inventory and source recipes, the exact
source preflight, the byte-exact write, and a full round-trip
verification (re-read, decode, bit-compare against the source).

This is an experimental-lane deliverable, not a registered product
identity: the section artifact carries no frontend resources and is not
bound by the engine yet. Combining it with the 27B NVFP4 base into the
experimental verify artifact, and any product adoption of the DSpark
backend, are later steps requiring BLOCKERS ratification.

Canonical invocation::

    python3 -m tools.convert.qwen3_6_27b.dspark \
      --model-dir models/dspark --out out/dspark_27b.ninfer
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import struct
import time
from typing import Mapping, Sequence

import torch

from tools.artifact.container import (
    Artifact,
    ArtifactIdentity,
    ArtifactWriter,
)
from tools.artifact.layouts import decode_direct
from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3_6.common import conversion as family_conversion
from tools.convert.qwen3_6.common import recipe as family_recipe
from tools.convert.qwen3_6.common.inventory import (
    BF16,
    CONTIGUOUS_LAYOUT,
    TensorSpec,
    tensor_spec,
)

from . import inventory as base

MODEL_ID = base.MODEL_ID
WEIGHTS_ID = "dspark-bf16"
RECIPE_ID = "qwen3_6_27b-dspark-bf16-v1"
DEFAULT_SOURCE = Path("models/dspark")

# DSpark checkpoint facts (models/dspark/config.json + dspark.py; the config
# validation below re-derives each of these from the source file).
HIDDEN_SIZE = 5120
INTERMEDIATE_SIZE = 10240
DSPARK_LAYERS = tuple(range(5))
Q_HEADS = 40
KV_HEADS = 8
HEAD_DIM = 128
VOCAB_SIZE = 248320
BLOCK_SIZE = 7  # total block rows: one anchor + six proposals (verify width 8)
MASK_TOKEN_ID = 248077
TARGET_LAYER_IDS = (4, 16, 28, 40, 52)
MARKOV_RANK = 256
QKV_WIDTH = (Q_HEADS + 2 * KV_HEADS) * HEAD_DIM  # 7168
GATE_UP_WIDTH = 2 * INTERMEDIATE_SIZE  # 20480

EXPECTED_OBJECT_COUNT = 2 + len(DSPARK_LAYERS) * 8 + 5  # 47
EXPECTED_SOURCE_TENSOR_COUNT = 62
# Identity BF16 ingestion: the artifact payload total equals the source
# safetensors data span byte-for-byte.
EXPECTED_SOURCE_DATA_BYTES = 2_718_569_474

_CONFIG = {
    "architectures": ["DSparkDraftModel"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "block_size": BLOCK_SIZE,
    "confidence_head_with_markov": True,
    "dtype": "bfloat16",
    "enable_confidence_head": True,
    "head_dim": HEAD_DIM,
    "hidden_act": "silu",
    "hidden_size": HIDDEN_SIZE,
    "intermediate_size": INTERMEDIATE_SIZE,
    "layer_types": ["full_attention"] * len(DSPARK_LAYERS),
    "markov_head_type": "vanilla",
    "markov_rank": MARKOV_RANK,
    "max_position_embeddings": 262144,
    "model_type": "qwen3",
    "num_attention_heads": Q_HEADS,
    "num_hidden_layers": len(DSPARK_LAYERS),
    "num_key_value_heads": KV_HEADS,
    "num_target_layers": 64,
    "rms_norm_eps": 1e-6,
    "sliding_window": None,
    "tie_word_embeddings": False,
    "use_sliding_window": False,
    "vocab_size": VOCAB_SIZE,
}
_ROPE_CONFIG = {
    "beta_fast": 32.0,
    "beta_slow": 1.0,
    "factor": 32.0,
    "original_max_position_embeddings": 8192,
    "rope_theta": 10000000,
    "rope_type": "yarn",
}
_DRAFT_CONFIG = {
    "attention_mode": "gqa",
    "confidence_head_alpha": 1.0,
    "confidence_head_with_markov": True,
    "enable_confidence_head": True,
    "markov_head_type": "vanilla",
    "markov_rank": MARKOV_RANK,
    "mask_token_id": MASK_TOKEN_ID,
    "projector_type": "dspark",
    "target_layer_ids": list(TARGET_LAYER_IDS),
}

# Fit constants for the experimental measurement (Ground rule 9): per-token
# resident costs added by the DSpark lane on top of the 27B NVFP4 base.
DRAFT_KV_BYTES_PER_TOKEN_BF16 = (
    len(DSPARK_LAYERS) * KV_HEADS * HEAD_DIM * 2 * 2  # K + V, BF16
)
AUX_TAP_BYTES_PER_TOKEN_BF16 = len(TARGET_LAYER_IDS) * HIDDEN_SIZE * 2
AUX_TAP_BYTES_PER_TOKEN_I8 = len(TARGET_LAYER_IDS) * HIDDEN_SIZE


def _tensor(name: str, shape: tuple[int, ...]) -> TensorSpec:
    return tensor_spec(name, shape, BF16)


def _build_dspark_specs() -> tuple[TensorSpec, ...]:
    specs: list[TensorSpec] = [
        _tensor(
            "dspark/feature_projection",
            (HIDDEN_SIZE, len(TARGET_LAYER_IDS) * HIDDEN_SIZE),
        ),
        _tensor("dspark/context_norm", (HIDDEN_SIZE,)),
    ]
    for layer in DSPARK_LAYERS:
        prefix = f"dspark/layers/{layer}/"
        specs.extend(
            (
                _tensor(prefix + "input_norm", (HIDDEN_SIZE,)),
                _tensor(
                    prefix + "attention/query_key_value",
                    (QKV_WIDTH, HIDDEN_SIZE),
                ),
                _tensor(prefix + "attention/query_norm", (HEAD_DIM,)),
                _tensor(prefix + "attention/key_norm", (HEAD_DIM,)),
                _tensor(prefix + "attention/output", (HIDDEN_SIZE, HIDDEN_SIZE)),
                _tensor(prefix + "post_attention_norm", (HIDDEN_SIZE,)),
                _tensor(prefix + "mlp/gate_up", (GATE_UP_WIDTH, HIDDEN_SIZE)),
                _tensor(prefix + "mlp/down", (HIDDEN_SIZE, INTERMEDIATE_SIZE)),
            )
        )
    specs.extend(
        (
            _tensor("dspark/final_norm", (HIDDEN_SIZE,)),
            _tensor("dspark/markov_w1", (VOCAB_SIZE, MARKOV_RANK)),
            _tensor("dspark/markov_w2", (VOCAB_SIZE, MARKOV_RANK)),
            _tensor("dspark/confidence_weight", (1, HIDDEN_SIZE + MARKOV_RANK)),
            _tensor("dspark/confidence_bias", (1,)),
        )
    )
    return tuple(specs)


DSPARK_TENSOR_SPECS = _build_dspark_specs()


def _build_dspark_recipes() -> tuple[family_recipe.TensorRecipe, ...]:
    recipes: list[family_recipe.TensorRecipe] = [
        family_recipe.TensorRecipe(
            "dspark/feature_projection",
            family_recipe.source(
                "fc.weight", (HIDDEN_SIZE, len(TARGET_LAYER_IDS) * HIDDEN_SIZE)
            ),
        ),
        family_recipe.TensorRecipe(
            "dspark/context_norm",
            family_recipe.source("hidden_norm.weight", (HIDDEN_SIZE,)),
        ),
    ]
    for layer in DSPARK_LAYERS:
        source_prefix = f"layers.{layer}."
        object_prefix = f"dspark/layers/{layer}/"
        recipes.extend(
            (
                family_recipe.TensorRecipe(
                    object_prefix + "input_norm",
                    family_recipe.source(
                        source_prefix + "input_layernorm.weight", (HIDDEN_SIZE,)
                    ),
                ),
                family_recipe.TensorRecipe(
                    object_prefix + "attention/query_key_value",
                    family_recipe.Concat(
                        (
                            family_recipe.source(
                                source_prefix + "self_attn.q_proj.weight",
                                (Q_HEADS * HEAD_DIM, HIDDEN_SIZE),
                            ),
                            family_recipe.source(
                                source_prefix + "self_attn.k_proj.weight",
                                (KV_HEADS * HEAD_DIM, HIDDEN_SIZE),
                            ),
                            family_recipe.source(
                                source_prefix + "self_attn.v_proj.weight",
                                (KV_HEADS * HEAD_DIM, HIDDEN_SIZE),
                            ),
                        ),
                        0,
                    ),
                ),
                family_recipe.TensorRecipe(
                    object_prefix + "attention/query_norm",
                    family_recipe.source(
                        source_prefix + "self_attn.q_norm.weight", (HEAD_DIM,)
                    ),
                ),
                family_recipe.TensorRecipe(
                    object_prefix + "attention/key_norm",
                    family_recipe.source(
                        source_prefix + "self_attn.k_norm.weight", (HEAD_DIM,)
                    ),
                ),
                family_recipe.TensorRecipe(
                    object_prefix + "attention/output",
                    family_recipe.source(
                        source_prefix + "self_attn.o_proj.weight",
                        (HIDDEN_SIZE, HIDDEN_SIZE),
                    ),
                ),
                family_recipe.TensorRecipe(
                    object_prefix + "post_attention_norm",
                    family_recipe.source(
                        source_prefix + "post_attention_layernorm.weight",
                        (HIDDEN_SIZE,),
                    ),
                ),
                family_recipe.TensorRecipe(
                    object_prefix + "mlp/gate_up",
                    family_recipe.Concat(
                        (
                            family_recipe.source(
                                source_prefix + "mlp.gate_proj.weight",
                                (INTERMEDIATE_SIZE, HIDDEN_SIZE),
                            ),
                            family_recipe.source(
                                source_prefix + "mlp.up_proj.weight",
                                (INTERMEDIATE_SIZE, HIDDEN_SIZE),
                            ),
                        ),
                        0,
                    ),
                ),
                family_recipe.TensorRecipe(
                    object_prefix + "mlp/down",
                    family_recipe.source(
                        source_prefix + "mlp.down_proj.weight",
                        (HIDDEN_SIZE, INTERMEDIATE_SIZE),
                    ),
                ),
            )
        )
    recipes.extend(
        (
            family_recipe.TensorRecipe(
                "dspark/final_norm", family_recipe.source("norm.weight", (HIDDEN_SIZE,))
            ),
            family_recipe.TensorRecipe(
                "dspark/markov_w1",
                family_recipe.source(
                    "markov_head.markov_w1.weight", (VOCAB_SIZE, MARKOV_RANK)
                ),
            ),
            family_recipe.TensorRecipe(
                "dspark/markov_w2",
                family_recipe.source(
                    "markov_head.markov_w2.weight", (VOCAB_SIZE, MARKOV_RANK)
                ),
            ),
            family_recipe.TensorRecipe(
                "dspark/confidence_weight",
                family_recipe.source(
                    "confidence_head.proj.weight", (1, HIDDEN_SIZE + MARKOV_RANK)
                ),
            ),
            family_recipe.TensorRecipe(
                "dspark/confidence_bias",
                family_recipe.source("confidence_head.proj.bias", (1,)),
            ),
        )
    )
    return tuple(recipes)


DSPARK_RECIPES = _build_dspark_recipes()
DSPARK_RECIPES_BY_NAME: Mapping[str, family_recipe.TensorRecipe] = {
    recipe.object_name: recipe for recipe in DSPARK_RECIPES
}


def validate_dspark_config(config: Mapping[str, object]) -> dict[str, object]:
    """Validate every DSpark fact that fixes storage or execution shape."""

    family_conversion.check_members("dspark config", config, _CONFIG)
    rope = config.get("rope_parameters")
    draft = config.get("dflash_config")
    if not isinstance(rope, Mapping) or not isinstance(draft, Mapping):
        raise ValueError(
            "DSpark config.json must contain rope_parameters and dflash_config"
        )
    family_conversion.check_members(
        "dspark config.rope_parameters", rope, _ROPE_CONFIG
    )
    family_conversion.check_members("dspark config.dflash_config", draft, _DRAFT_CONFIG)
    return (
        {name: config[name] for name in _CONFIG}
        | {
            "rope_parameters": {name: rope[name] for name in _ROPE_CONFIG},
            "dflash_config": {name: draft[name] for name in _DRAFT_CONFIG},
        }
    )


def preflight_inventory() -> None:
    """Prove the DSpark section inventory before any payload is written."""

    if len(DSPARK_TENSOR_SPECS) != EXPECTED_OBJECT_COUNT:
        raise ValueError(
            f"dspark inventory is incomplete: {len(DSPARK_TENSOR_SPECS)}"
        )
    if len({spec.name for spec in DSPARK_TENSOR_SPECS}) != len(DSPARK_TENSOR_SPECS):
        raise ValueError("dspark inventory has duplicate object names")
    if any(
        spec.format != BF16 or spec.layout != CONTIGUOUS_LAYOUT
        for spec in DSPARK_TENSOR_SPECS
    ):
        raise ValueError("dspark section must store exact BF16 contiguous payloads")
    payload_bytes = family_conversion.tensor_payload_bytes(DSPARK_TENSOR_SPECS)
    if payload_bytes != EXPECTED_SOURCE_DATA_BYTES:
        raise ValueError(
            f"dspark payload total {payload_bytes} != source data span "
            f"{EXPECTED_SOURCE_DATA_BYTES}"
        )
    requirements = family_recipe.source_requirements(DSPARK_RECIPES)
    if len(requirements) != EXPECTED_SOURCE_TENSOR_COUNT:
        raise ValueError(
            f"dspark recipes cover {len(requirements)} unique sources, "
            f"expected {EXPECTED_SOURCE_TENSOR_COUNT}"
        )
    family_recipe.validate_recipe_coverage(DSPARK_RECIPES, DSPARK_TENSOR_SPECS)


def preflight_dspark_sources(model_dir: str | Path) -> family_recipe.SourcePreflight:
    """Exact preflight of the single-file HF DSpark checkpoint."""

    model = Path(model_dir)
    reader = ShardReader.from_file(model / "model.safetensors")
    try:
        expected = set(family_recipe.source_requirements(DSPARK_RECIPES))
        actual = set(reader.names)
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        if missing or unexpected:
            raise ValueError(
                "DSpark checkpoint tensor set mismatch:\n  missing: "
                + ", ".join(missing)
                + "\n  unexpected: "
                + ", ".join(unexpected)
            )
        return family_recipe.preflight_source_reader(reader, DSPARK_RECIPES)
    finally:
        reader.close()


def source_file_facts(model_dir: str | Path) -> dict[str, object]:
    """Prove the safetensors container is complete and matches the inventory."""

    path = Path(model_dir) / "model.safetensors"
    file_bytes = path.stat().st_size
    digest = sha256()
    with path.open("rb") as handle:
        header_bytes = handle.read(8)
        (header_len,) = struct.unpack("<Q", header_bytes)
        digest.update(header_bytes)
        remaining = file_bytes - 8
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
            remaining -= len(chunk)
        if remaining != 0:
            raise ValueError("safetensors file size disagrees with stat")
    data_bytes = file_bytes - 8 - header_len
    if data_bytes != EXPECTED_SOURCE_DATA_BYTES:
        raise ValueError(
            f"safetensors data span {data_bytes} != expected {EXPECTED_SOURCE_DATA_BYTES}"
        )
    return {
        "path": str(path.resolve()),
        "bytes": file_bytes,
        "header_bytes": header_len,
        "data_bytes": data_bytes,
        "sha256": digest.hexdigest(),
    }


def object_provenance(
    specs: Sequence[TensorSpec],
) -> dict[str, dict[str, object]]:
    return {
        spec.name: {
            "sources": sorted(
                item.name
                for item in family_recipe.expression_sources(
                    DSPARK_RECIPES_BY_NAME[spec.name].expression
                )
            ),
            "shape": list(spec.shape),
            "format": spec.format,
            "layout": spec.layout,
        }
        for spec in specs
    }


def verify_artifact(out_path: str | Path, model_dir: str | Path) -> int:
    """Round-trip proof: every written object decodes bit-exact to its source."""

    checked = 0
    reader = ShardReader.from_file(Path(model_dir) / "model.safetensors")
    with reader, Artifact.open(out_path) as artifact:
        if (
            artifact.identity.model_id,
            artifact.identity.weights_id,
        ) != (MODEL_ID, WEIGHTS_ID):
            raise ValueError(
                f"artifact identity {artifact.identity} != {(MODEL_ID, WEIGHTS_ID)}"
            )
        if len(artifact.objects) != len(DSPARK_TENSOR_SPECS):
            raise ValueError(
                f"artifact has {len(artifact.objects)} objects, expected "
                f"{len(DSPARK_TENSOR_SPECS)}"
            )
        for spec in DSPARK_TENSOR_SPECS:
            obj = artifact.find(spec.name)
            if (
                tuple(obj.shape) != spec.shape
                or obj.format != spec.format
                or obj.layout != spec.layout
            ):
                raise ValueError(
                    f"{spec.name}: directory entry {obj} != inventory {spec}"
                )
            got = decode_direct(bytes(artifact.payload(obj)), spec.format, spec.shape)
            want = family_recipe.materialize_recipe(
                DSPARK_RECIPES_BY_NAME[spec.name], reader
            )
            if got.dtype != want.dtype or tuple(got.shape) != tuple(want.shape):
                raise ValueError(f"{spec.name}: decoded dtype/shape mismatch")
            if (
                got.contiguous().view(torch.uint8).numpy().tobytes()
                != want.contiguous().view(torch.uint8).numpy().tobytes()
            ):
                raise ValueError(f"{spec.name}: round-trip is not bit-exact")
            checked += 1
    return checked


def convert(model_dir: str | Path, out_path: str | Path, *, verify: bool = True) -> Path:
    """Run the complete DSpark section conversion and return its report path."""

    started = time.perf_counter()
    model = Path(model_dir)
    output = Path(out_path)
    config_summary = validate_dspark_config(
        family_conversion.load_json(model / "config.json")
    )
    preflight_inventory()
    source_preflight = preflight_dspark_sources(model)
    facts = source_file_facts(model)

    plan = family_conversion.build_object_plan(DSPARK_TENSOR_SPECS, {})
    identity = ArtifactIdentity(MODEL_ID, WEIGHTS_ID)
    output.parent.mkdir(parents=True, exist_ok=True)
    with ShardReader.from_file(model / "model.safetensors") as reader, ArtifactWriter(
        output, identity, plan.specs
    ) as writer:
        index = 0
        for spec in DSPARK_TENSOR_SPECS:
            tensor = family_recipe.materialize_recipe(
                DSPARK_RECIPES_BY_NAME[spec.name], reader
            )
            payload = family_conversion.encode_tensor_payload(tensor, spec, "cpu")
            del tensor
            writer.write(spec.name, payload)
            del payload
            index += 1
            print(
                f"[{index}/{len(DSPARK_TENSOR_SPECS)}] {spec.name}",
                flush=True,
            )

    objects_checked = verify_artifact(output, model) if verify else 0
    elapsed = time.perf_counter() - started
    final_bytes = output.stat().st_size

    with Artifact.open(output) as artifact:
        statistics = family_conversion.object_statistics(artifact.objects)
        objects_bytes = {obj.name: obj.bytes for obj in artifact.objects}

    arguments = {
        "model_dir": str(model_dir),
        "out": str(out_path),
        "verify": verify,
    }
    report = {
        "identity": {
            "model_id": identity.model_id,
            "weights_id": identity.weights_id,
        },
        "recipe_id": RECIPE_ID,
        "lane": "autoninfer experimental (not a registered product identity)",
        "source": {
            "model_path": str(model.resolve()),
            "safetensors": facts,
        },
        "arguments": arguments,
        "config_summary": config_summary,
        "source_preflight": {
            "recipes": source_preflight.recipe_count,
            "tensors": source_preflight.source_tensor_count,
            "shards": source_preflight.source_shard_count,
            "dtypes": dict(source_preflight.source_dtype_counts),
        },
        "objects": statistics,
        "object_bytes": objects_bytes,
        "provenance": object_provenance(DSPARK_TENSOR_SPECS),
        "fit": {
            "draft_weights_bytes": facts["data_bytes"],
            "draft_kv_bytes_per_token_bf16": DRAFT_KV_BYTES_PER_TOKEN_BF16,
            "aux_tap_bytes_per_token_bf16": AUX_TAP_BYTES_PER_TOKEN_BF16,
            "aux_tap_bytes_per_token_i8": AUX_TAP_BYTES_PER_TOKEN_I8,
            "block_size": BLOCK_SIZE,
            "proposals_per_block": BLOCK_SIZE - 1,
            "verify_width": BLOCK_SIZE + 1,
            "target_layer_ids": list(TARGET_LAYER_IDS),
            "mask_token_id": MASK_TOKEN_ID,
            "rope": "yarn (factor 32, original max 8192, theta 1e7)",
        },
        "verification": {
            "round_trip_objects": objects_checked,
            "round_trip": "bit-exact" if objects_checked else "skipped",
        },
        "elapsed_seconds": elapsed,
        "artifact": {
            "path": str(output),
            "bytes": final_bytes,
        },
        "converter": {
            "revision": family_conversion.converter_revision(_repo_root()),
            "environment": dict(
                family_conversion.environment(torch.device("cpu"))
            ),
        },
    }
    report_path = Path(str(output) + ".conversion.json")
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(
        f"complete: {final_bytes} bytes in {elapsed:.1f}s; "
        f"verification={'bit-exact' if objects_checked else 'skipped'}; "
        f"report={report_path}",
        flush=True,
    )
    return report_path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=DEFAULT_SOURCE, type=Path)
    parser.add_argument("--out", default=Path("out/dspark_27b.ninfer"), type=Path)
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args(argv)
    convert(args.model_dir, args.out, verify=not args.no_verify)


__all__ = [
    "AUX_TAP_BYTES_PER_TOKEN_BF16",
    "AUX_TAP_BYTES_PER_TOKEN_I8",
    "BLOCK_SIZE",
    "DRAFT_KV_BYTES_PER_TOKEN_BF16",
    "DSPARK_LAYERS",
    "DSPARK_RECIPES",
    "DSPARK_RECIPES_BY_NAME",
    "DSPARK_TENSOR_SPECS",
    "MASK_TOKEN_ID",
    "MARKOV_RANK",
    "MODEL_ID",
    "RECIPE_ID",
    "TARGET_LAYER_IDS",
    "WEIGHTS_ID",
    "convert",
    "main",
    "preflight_dspark_sources",
    "preflight_inventory",
    "validate_dspark_config",
    "verify_artifact",
]


if __name__ == "__main__":
    main()