from peft import LoraConfig

# Define a configuration for the LoRA (Learnable Requantization Activation) method
lora_config = LoraConfig(
    r=8,  # Number of quantization levels
    lora_alpha=32,  # Hyperparameter for LoRA
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],  # Modules to apply LoRA to
    lora_dropout=0.05,  # Dropout probability
    bias="none",  # Type of bias
    task_type="CAUSAL_LM",  # Task type (in this case, Causal Language Modeling)
)
