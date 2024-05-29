from peft import PeftModel

from scripts.config import BASE_MODEL_NAME, parse_arge
from scripts.load_model_tokenizer import load_model, load_tokenizer
from scripts.preprocess_data import (
    custom_dataset_load,
    format_dataset_blueprint_instruct_fn,
    format_dataset_detail_instruct_fn,
    format_prediction_blueprint_instruct_fn,
    format_prediction_detail_instruct_fn,
    shuffle_tokenize_batch,
    split_train_test_dataset,
)
from scripts.utils import KeywordsStoppingCriteria, get_completion


def main():
    args = parse_arge()

    tokenizer = load_tokenizer(BASE_MODEL_NAME)

    data = custom_dataset_load("data")

    data_blueprint = format_dataset_blueprint_instruct_fn(data)
    processed_data_blueprint = shuffle_tokenize_batch(data_blueprint, tokenizer)

    data_detail = format_dataset_detail_instruct_fn(data)
    processed_data_detail = shuffle_tokenize_batch(data_detail, tokenizer)

    _, test_data_blueprint = split_train_test_dataset(processed_data_blueprint)
    _, test_data_detail = split_train_test_dataset(processed_data_detail)

    model = load_model(args.model_id)

    # Resize token embeddings
    model.resize_token_embeddings(len(tokenizer))

    # Update pad token id in model and its config
    model.pad_token_id = tokenizer.pad_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # Check if they are equal
    assert (
        model.pad_token_id == tokenizer.pad_token_id
    ), "The model's pad token ID does not match the tokenizer's pad token ID"

    model.config.use_cache = True

    stop_keywords = ["</s>", " </s>", "</s> "]

    pred_prompt_blueprint = format_prediction_blueprint_instruct_fn(
        test_data_blueprint[0]
    )
    pred_prompt_detail = format_prediction_detail_instruct_fn(test_data_detail[0])

    output_base_blueprint = get_completion(
        pred_prompt_blueprint, model, tokenizer, stop_keywords, KeywordsStoppingCriteria
    )
    print(output_base_blueprint)

    output_base_detail = get_completion(
        pred_prompt_detail, model, tokenizer, stop_keywords, KeywordsStoppingCriteria
    )
    print(output_base_detail)

    adapter_id_blueprint = "path-to-blueprint-planning-adapter"
    adapter_id_detail = "path-to-detail-planning-adapter"

    model = PeftModel.from_pretrained(
        model, adapter_id_blueprint, adapter_name="blueprint"
    )
    _ = model.load_adapter(adapter_id_detail, adapter_name="detail")

    adapters = ["blueprint", "detail"]

    weights = [1.0, 1.0]
    adapter_name = "merge"
    density = 0.2
    combination_type = "ties"

    if adapter_name in model.peft_config:
        model.delete_adapter(adapter_name)

    model.add_weighted_adapter(
        adapters,
        weights,
        adapter_name,
        combination_type=combination_type,
        density=density,
    )

    model.set_adapter("merge")

    output_blueprint = get_completion(
        pred_prompt_blueprint, model, tokenizer, stop_keywords, KeywordsStoppingCriteria
    )
    print(output_blueprint)

    output_detail = get_completion(
        pred_prompt_detail, model, tokenizer, stop_keywords, KeywordsStoppingCriteria
    )
    print(output_detail)


if __name__ == "__main__":
    main()
