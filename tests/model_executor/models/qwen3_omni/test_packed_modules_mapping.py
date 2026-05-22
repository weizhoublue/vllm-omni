import pytest
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
    Qwen3OmniMoeThinkerForConditionalGeneration,
)

@pytest.mark.core_model
@pytest.mark.cpu
def test_qwen3_omni_packed_modules_mapping():
    mapping = Qwen3OmniMoeThinkerForConditionalGeneration.packed_modules_mapping
    
    # Existing ones
    assert "qkv_proj" in mapping
    assert mapping["qkv_proj"] == ["q_proj", "k_proj", "v_proj"]
    assert "gate_up_proj" in mapping
    assert mapping["gate_up_proj"] == ["gate_proj", "up_proj"]
    
    # New ones added to match Qwen2.5-Omni
    assert "attn.qkv" in mapping
    assert mapping["attn.qkv"] == ["attn.q", "attn.k", "attn.v"]
    assert "attn_qkv_proj" in mapping
    assert mapping["attn_qkv_proj"] == ["attn_q_proj", "attn_k_proj", "attn_v_proj"]
    assert "qkv" in mapping
    assert mapping["qkv"] == ["q", "k", "v"]
