from typing import Dict, List, NoReturn, Tuple, TypeVar

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict

T = TypeVar("T")

DataPointType = DatasetDict | Dataset | IterableDatasetDict | IterableDataset
DataPointTrain = Dataset | List | T
DataPointTest = Dataset | List | T

from transformers.models.llama.tokenization_llama_fast import LlamaTokenizerFast
from transformers.models.mistral.modeling_mistral import MistralForCausalLM
