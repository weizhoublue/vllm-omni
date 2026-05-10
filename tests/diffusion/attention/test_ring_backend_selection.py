# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import types
from importlib import util
from pathlib import Path

import pytest
import torch

try:
    import vllm.config  # noqa: F401
except ImportError:
    vllm_mod = types.ModuleType("vllm")
    vllm_config_mod = types.ModuleType("vllm.config")

    class DeviceConfig:
        def __init__(self, device):
            self.device = device

    class VllmConfig:
        def __init__(self, device_config):
            self.device_config = device_config

    class _CurrentVllmConfig:
        def __init__(self, config):
            self.config = config

        def __enter__(self):
            return self.config

        def __exit__(self, exc_type, exc, tb):
            return False

    vllm_config_mod.DeviceConfig = DeviceConfig
    vllm_config_mod.VllmConfig = VllmConfig
    vllm_config_mod.set_current_vllm_config = _CurrentVllmConfig
    vllm_mod.config = vllm_config_mod
    sys.modules.setdefault("vllm", vllm_mod)
    sys.modules.setdefault("vllm.config", vllm_config_mod)


class _DummySpGroup:
    ring_group = object()


def _load_ring_module(monkeypatch):
    class _Logger:
        def warning_once(self, *args, **kwargs):
            pass

    logger_mod = types.ModuleType("vllm.logger")
    logger_mod.init_logger = lambda name: _Logger()

    ring_globals_mod = types.ModuleType("vllm_omni.diffusion.attention.backends.ring.ring_globals")
    ring_globals_mod.HAS_FA3 = True
    ring_globals_mod.HAS_FLASH_ATTN = True

    ring_selector_mod = types.ModuleType("vllm_omni.diffusion.attention.backends.ring.ring_selector")
    ring_selector_mod.AttnType = types.SimpleNamespace(FA3="fa3", FA="fa")

    base_mod = types.ModuleType("vllm_omni.diffusion.attention.parallel.base")

    class ParallelAttentionContext:
        def __init__(self, name):
            self.name = name

    base_mod.ParallelAttentionContext = ParallelAttentionContext

    group_mod = types.ModuleType("vllm_omni.diffusion.distributed.group_coordinator")
    group_mod.SequenceParallelGroupCoordinator = object

    for name in [
        "vllm",
        "vllm_omni",
        "vllm_omni.diffusion",
        "vllm_omni.diffusion.attention",
        "vllm_omni.diffusion.attention.backends",
        "vllm_omni.diffusion.attention.backends.ring",
        "vllm_omni.diffusion.attention.parallel",
        "vllm_omni.diffusion.distributed",
    ]:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))

    monkeypatch.setitem(sys.modules, "vllm.logger", logger_mod)
    monkeypatch.setitem(
        sys.modules,
        "vllm_omni.diffusion.attention.backends.ring.ring_globals",
        ring_globals_mod,
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_omni.diffusion.attention.backends.ring.ring_selector",
        ring_selector_mod,
    )
    monkeypatch.setitem(sys.modules, "vllm_omni.diffusion.attention.parallel.base", base_mod)
    monkeypatch.setitem(
        sys.modules,
        "vllm_omni.diffusion.distributed.group_coordinator",
        group_mod,
    )

    path = Path(__file__).parents[3] / "vllm_omni/diffusion/attention/parallel/ring.py"
    spec = util.spec_from_file_location("ring_under_test", path)
    assert spec is not None and spec.loader is not None
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ring_attention_rejects_unsupported_backend_pref(monkeypatch):
    ring_mod = _load_ring_module(monkeypatch)
    monkeypatch.setattr(ring_mod, "HAS_FA3", True)
    monkeypatch.setattr(ring_mod, "HAS_FLASH_ATTN", True)

    fake_flash_mod = types.ModuleType("vllm_omni.diffusion.attention.backends.ring_flash_attn")

    def _unexpected_flash_attn(*args, **kwargs):
        raise AssertionError("unsupported backend preference reached flash ring attention")

    fake_flash_mod.ring_flash_attn_func = _unexpected_flash_attn
    monkeypatch.setitem(
        sys.modules,
        "vllm_omni.diffusion.attention.backends.ring_flash_attn",
        fake_flash_mod,
    )

    attention = ring_mod.RingParallelAttention(_DummySpGroup(), attn_backend_pref="SAGE_ATTN")
    query = torch.empty(1, 1, 1, 8, dtype=torch.float16)
    key = torch.empty_like(query)
    value = torch.empty_like(query)

    with pytest.raises(ValueError, match="SAGE_ATTN.*does not support ring attention"):
        attention.run_attention(query, key, value, attn_metadata=None)
