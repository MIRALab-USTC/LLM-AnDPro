from importlib.metadata import version
import warnings
import transformers
import transformers.models.mistral.modeling_mistral
from andpro_snapkv.monkeypatch.andpro_mistral_hijack import andpro_mistral_flash_attn2_forward, andpro_MistralModel_forward
from andpro_snapkv.monkeypatch.andpro_mistral_hijack import prepare_inputs_for_generation_mistral as andpro_prepare_inputs_for_generation_mistral
from andpro_snapkv.monkeypatch.andpro_llama_hijack import andpro_llama_flash_attn2_forward_2,andpro_LlamaModel_forward_2,andpro_llama_flash_attn2_forward,andpro_LlamaModel_forward
from andpro_snapkv.monkeypatch.andpro_llama_hijack import prepare_inputs_for_generation_llama as ada_prepare_inputs_for_generation_llama

def check_version():
    try:
        transformers_version = version("transformers")
    except Exception as e:
        print(f"Transformers not installed: {e}")
    version_list = ['4.37']
    warning_flag = True
    for x in version_list:
        if x in transformers_version:
            warning_flag = False
            break
    if warning_flag:
        warnings.warn(f"Transformers version {transformers_version} might not be compatible with SnapKV. SnapKV is tested with Transformers version {version_list}.")


def replace_mistral_andpro():
    check_version()
    transformers.models.mistral.modeling_mistral.MistralForCausalLM.prepare_inputs_for_generation = andpro_prepare_inputs_for_generation_mistral
    transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = andpro_mistral_flash_attn2_forward
    transformers.models.mistral.modeling_mistral.MistralModel.forward = andpro_MistralModel_forward

def replace_llama_andpro():
    check_version()
    transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = ada_prepare_inputs_for_generation_llama
    transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = andpro_llama_flash_attn2_forward_2
    transformers.models.llama.modeling_llama.LlamaModel.forward = andpro_LlamaModel_forward_2
