# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

ROOT = Path(__file__).resolve().parents[4]

PIPELINES = [
    ROOT / "vllm_omni/diffusion/models/flux/pipeline_flux.py",
    ROOT / "vllm_omni/diffusion/models/flux/pipeline_flux_kontext.py",
]


def _source(path: Path) -> str:
    return path.read_text()


def _method_source(path: Path, method_name: str) -> str:
    source = _source(path)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return ast.get_source_segment(source, node) or ""
    raise AssertionError(f"{method_name} not found in {path}")


def test_flux_prompt_tensors_use_od_config_dtype():
    for path in PIPELINES:
        t5_source = _method_source(path, "_get_t5_prompt_embeds")
        clip_source = _method_source(path, "_get_clip_prompt_embeds")
        encode_source = _method_source(path, "encode_prompt")

        assert "prompt_embeds.to(dtype=self.od_config.dtype" in t5_source
        assert "prompt_embeds.to(dtype=self.od_config.dtype" in clip_source
        assert "prompt_embeds.to(dtype=self.od_config.dtype" in encode_source
        assert "pooled_prompt_embeds.to(dtype=self.od_config.dtype" in encode_source
        assert "dtype=self.od_config.dtype" in encode_source


def test_flux_decode_casts_latents_to_vae_dtype_before_scaling():
    for path in PIPELINES:
        source = _source(path)
        cast_line = source.index("latents = latents.to(self.vae.dtype)")
        scale_line = source.index("latents = (latents / self.vae.config.scaling_factor)")
        decode_line = source.index("self.vae.decode(latents")

        assert cast_line < scale_line < decode_line
