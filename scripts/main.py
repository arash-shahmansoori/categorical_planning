import torch
from config import OUTPUT_DIR_NAME, parse_arge
from datasets import load_from_disk
from load_model_tokenizer import load_model, load_tokenizer
from lora import lora_config
from training import (
    custom_collator,
    prepare_model_for_qlora_training,
    qlora_training_with_custom_collator,
)
from transformers import AutoTokenizer, TrainingArguments, set_seed


def training_function(args):
    # set seed
    set_seed(args.seed)

    dataset_train = load_from_disk(args.train_dataset_path)
    dataset_test = load_from_disk(args.test_dataset_path)

    tokenizer = load_tokenizer()

    model = load_model(args.model_id)

    # Resize token embeddings
    model.resize_token_embeddings(len(tokenizer))

    # Update pad token id in model and its config
    model.pad_token_id = tokenizer.pad_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model = prepare_model_for_qlora_training(model, lora_config)

    collator = custom_collator(tokenizer)

    # Define a configuration for the training
    training_args_blueprint = TrainingArguments(
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        warmup_ratio=0.03,  # Number of warm-up steps for learning rate
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",  # cosine learning rate scheduler
        bf16=True,  # Enable mixed-precision training (Amper GPU)
        # fp16=True,
        logging_steps=1,  # Logging frequency during training
        output_dir=OUTPUT_DIR_NAME,  # Directory to save output files
        optim="paged_adamw_32bit",  # Optimizer type
        save_strategy="epoch",  # Strategy for saving checkpoints
        # push_to_hub=True                    # Push to the Hugging Face model hub
        # report_to="wandb",
        do_eval=True,
        evaluation_strategy="steps",
        eval_steps=5,
    )

    # Start QLORA fine-tuning for the abstract data (i.e., easy task or layer-1 planning)
    trainer, tokenizer, model = qlora_training_with_custom_collator(
        training_args_blueprint,
        model,
        tokenizer,
        dataset_train,
        dataset_test,
        collator,
    )

    sagemaker_save_dir = "/opt/ml/model/"
    if args.merge_weights:
        # merge adapter weights with base model and save
        # save int 4 model
        trainer.model.save_pretrained(OUTPUT_DIR_NAME, safe_serialization=False)
        # clear memory
        del model
        del trainer
        torch.cuda.empty_cache()

        from peft import AutoPeftModelForCausalLM

        # load PEFT model in fp16
        model = AutoPeftModelForCausalLM.from_pretrained(
            OUTPUT_DIR_NAME,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
        # Merge LoRA and base model and save
        model = model.merge_and_unload()
        model.save_pretrained(
            sagemaker_save_dir, safe_serialization=True, max_shard_size="2GB"
        )
    else:
        trainer.model.save_pretrained(sagemaker_save_dir, safe_serialization=True)

    # save tokenizer for easy inference
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.save_pretrained(sagemaker_save_dir)


def main():
    args = parse_arge()
    training_function(args)


if __name__ == "__main__":
    main()
