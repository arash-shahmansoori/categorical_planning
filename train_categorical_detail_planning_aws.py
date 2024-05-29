import time

from sagemaker.huggingface import HuggingFace

from aws import create_aws_sess_role
from scripts.config import BASE_MODEL_NAME, hyperparameters
from scripts.load_model_tokenizer import load_tokenizer
from scripts.preprocess_data import (
    custom_dataset_load,
    format_dataset_detail_instruct_fn,
    shuffle_tokenize_batch,
    split_train_test_dataset,
)


def main():
    sess, role = create_aws_sess_role()

    tokenizer = load_tokenizer(BASE_MODEL_NAME)

    data = custom_dataset_load("data_bkup_update")
    data_detail = format_dataset_detail_instruct_fn(data)
    processed_data_detail = shuffle_tokenize_batch(data_detail, tokenizer)

    train_data_detail, test_data_detail = split_train_test_dataset(
        processed_data_detail
    )

    # save train_dataset to s3
    training_input_path_detail = (
        f"s3://{sess.default_bucket()}/processed/mistral/planning-detail/train"
    )
    train_data_detail.save_to_disk(training_input_path_detail)

    # save test_dataset to s3
    testing_input_path_detail = (
        f"s3://{sess.default_bucket()}/processed/mistral/planning-detail/test"
    )
    test_data_detail.save_to_disk(testing_input_path_detail)

    print("uploaded data to:")
    print(f"training dataset to: {training_input_path_detail}")
    print(f"testing dataset to: {testing_input_path_detail}")

    # define Training Job Name
    job_name = (
        f'huggingface-qlora-{time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())}'
    )

    # create the Estimator
    huggingface_estimator = HuggingFace(
        entry_point="main.py",  # train script
        source_dir="scripts",  # directory which includes all the files needed for training
        instance_type="ml.g5.2xlarge",  # instances type used for the training job
        instance_count=1,  # the number of instances used for training
        base_job_name=job_name,  # the name of the training job
        role=role,  # Iam role used in training job to access AWS ressources, e.g. S3
        volume_size=300,  # the size of the EBS volume in GB
        transformers_version="4.28",  # the transformers version used in the training job
        pytorch_version="2.0",  # the pytorch_version version used in the training job
        py_version="py310",  # the python version used in the training job
        hyperparameters=hyperparameters,  # the hyperparameters passed to the training job
        environment={
            "HUGGINGFACE_HUB_CACHE": "/tmp/.cache"
        },  # set env variable to cache models in /tmp
    )

    # define a data input dictonary with our uploaded s3 uris
    data = {"train": training_input_path_detail, "test": testing_input_path_detail}

    # starting the train job with our uploaded datasets as input
    huggingface_estimator.fit(data, wait=True)


if __name__ == "__main__":
    main()
