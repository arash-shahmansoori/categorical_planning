from transformers import Trainer
from type_extension import LlamaTokenizerFast, MistralForCausalLM


def qlora_training_with_custom_collator(
    training_args,
    model: MistralForCausalLM,
    tokenizer: LlamaTokenizerFast,
    train_data,
    test_data,
    collator,
):
    # Create a trainer for fine-tuning a model
    trainer = Trainer(
        model=model,  # The model to be trained
        train_dataset=train_data,  # Training dataset
        eval_dataset=test_data,  # Evaluation dataset
        args=training_args,
        data_collator=collator,  # Custom data collator for completion
    )

    model.config.use_cache = (
        False  # silence the warnings. Please re-enable for inference!
    )

    trainer.train()

    return trainer, tokenizer, model
