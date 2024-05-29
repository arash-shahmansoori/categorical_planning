from type_extension import (
    DataPointTest,
    DataPointTrain,
    DataPointType,
    LlamaTokenizerFast,
    Tuple,
)


def shuffle_tokenize_batch(
    data: DataPointType, tokenizer: LlamaTokenizerFast
) -> DataPointType:
    data = data.shuffle(seed=1234)  # Shuffle dataset here
    data = data.map(lambda samples: tokenizer(samples["prompt"]), batched=True)
    return data


def split_train_test_dataset(
    data: DataPointType, split_size: float = 0.1
) -> Tuple[DataPointTrain, DataPointTest]:
    data = data.train_test_split(test_size=split_size)
    train_data = data["train"]
    test_data = data["test"]
    return train_data, test_data
