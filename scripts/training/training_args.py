from config import OUTPUT_DIR_NAME
from transformers import TrainingArguments

# Define a configuration for the training
training_args_blueprint = TrainingArguments(
    per_device_train_batch_size=4,  # Batch size per device during training
    gradient_accumulation_steps=4,  # Number of gradient accumulation steps
    gradient_checkpointing=True,
    warmup_ratio=0.03,  # Number of warm-up steps for learning rate
    max_steps=60,  # Maximum number of training steps
    learning_rate=5e-5,  # Learning rate
    lr_scheduler_type="cosine",  # cosine learning rate scheduler
    # bf16=True,                          # Enable mixed-precision training (Amper GPU)
    fp16=True,
    logging_steps=1,  # Logging frequency during training
    output_dir=OUTPUT_DIR_NAME,  # Directory to save output files
    optim="paged_adamw_32bit",  # Optimizer type
    save_strategy="epoch",  # Strategy for saving checkpoints
    # push_to_hub=True                    # Push to the Hugging Face model hub
    report_to="wandb",
    do_eval=True,
    evaluation_strategy="steps",
    eval_steps=5,
)
