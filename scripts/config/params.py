import argparse

import torch
from huggingface_hub import HfFolder, login

# BASE_MODEL_NAME = "mistralai/Mistral-7B-v0.1"
BASE_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

OUTPUT_DIR_NAME = "/tmp/mistral_blueprint_planning"
ADAPTER_CHECKPOINTS_NAME = "ckpts_mistral_7B_blueprint_planning_decompn"


# hyperparameters, which are passed into the training job
hyperparameters = {
    "model_id": f"{BASE_MODEL_NAME}",  # pre-trained model
    "train_dataset_path": "/opt/ml/input/data/training",  # path where sagemaker will save training dataset
    "test_dataset_path": "/opt/ml/input/data/testing",  # path where sagemaker will save testing dataset
    "max_steps": 1,  # Maximum number of training steps
    "per_device_train_batch_size": 4,  # batch size for training
    "gradient_accumulation_steps": 4,  # Number of gradient accumulation steps
    "lr": 5e-5,  # learning rate used during training
    "hf_token": HfFolder.get_token(),  # huggingface token to access llama 2
    "merge_weights": False,  # wether to merge LoRA into the model (needs more memory)
}


def parse_arge():
    """Parse the arguments."""
    parser = argparse.ArgumentParser()
    # add model id and dataset path argument
    parser.add_argument(
        "--model_id",
        type=str,
        help="Model id to use for training.",
    )
    parser.add_argument(
        "--train_dataset_path",
        type=str,
        default="train_data_blueprint",
        help="Path to train dataset.",
    )
    parser.add_argument(
        "--test_dataset_path",
        type=str,
        default="test_data_blueprint",
        help="Path to test dataset.",
    )
    parser.add_argument(
        "--hf_token", type=str, default=HfFolder.get_token(), help="Path to dataset."
    )
    # add training hyperparameters for epochs, batch size, learning rate, and seed
    parser.add_argument(
        "--max_steps", type=int, default=1, help="Maximum number of training steps."
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Batch size to use for training.",
    )
    parser.add_argument(
        "--lr", type=float, default=5e-5, help="Learning rate to use for training."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed to use for training."
    )
    parser.add_argument(
        "--gradient_checkpointing",
        type=bool,
        default=True,
        help="Path to deepspeed config file.",
    )
    parser.add_argument(
        "--bf16",
        type=bool,
        default=True if torch.cuda.get_device_capability()[0] == 8 else False,
        help="Whether to use bf16.",
    )
    parser.add_argument(
        "--merge_weights",
        type=bool,
        default=False,
        help="Whether to merge LoRA weights with base model.",
    )
    args, _ = parser.parse_known_args()

    if args.hf_token:
        print(f"Logging into the Hugging Face Hub with token {args.hf_token[:10]}...")
        login(token=args.hf_token)

    return args
