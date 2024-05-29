from transformers import AutoTokenizer
from type_extension import LlamaTokenizerFast


def load_tokenizer(model_id: str) -> LlamaTokenizerFast:
    tokenizer = AutoTokenizer.from_pretrained(model_id, add_eos_token=True)

    # Check if the pad token is already in the tokenizer
    if "[<pad>]" not in tokenizer.get_vocab():
        # Add the pad token
        tokenizer.add_tokens(["[<pad>]"])
        print("Added pad token")

    # Set the pad token
    tokenizer.pad_token = "[<pad>]"

    tokenizer.padding_side = "left"

    # Print the pad token ids
    print("Tokenizer pad token ID:", tokenizer.pad_token_id)
    return tokenizer
