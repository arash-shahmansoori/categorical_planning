from trl import DataCollatorForCompletionOnlyLM
from type_extension import LlamaTokenizerFast


def custom_collator(
    tokenizer: LlamaTokenizerFast,
    response_template_with_context: str = "\n### Response:",
):
    response_template_ids = tokenizer.encode(
        response_template_with_context, add_special_tokens=False
    )[2:]
    collator = DataCollatorForCompletionOnlyLM(
        response_template_ids, tokenizer=tokenizer
    )
    return collator
