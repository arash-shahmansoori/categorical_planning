from datasets import load_dataset

from type_extension import DataPointType


def custom_dataset_load(data_dir: str) -> DataPointType:
    data = load_dataset(data_dir, split="train")
    return data
