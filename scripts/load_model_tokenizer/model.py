import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from type_extension import MistralForCausalLM


def load_model(
    model_id: str, gradient_checkpointing: bool = True, device_map: str = "auto"
) -> MistralForCausalLM:

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        use_cache=False if gradient_checkpointing else True,
        quantization_config=bnb_config,
        device_map=device_map,
    )
    return model
