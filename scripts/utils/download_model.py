import boto3

s3_client = boto3.client("s3")


bucket_name = "<S3_BUCKET>"
job_prefix = f"huggingface-qlora"


def get_last_job_name(job_name_prefix):
    import boto3

    sagemaker_client = boto3.client("sagemaker")

    search_response = sagemaker_client.search(
        Resource="TrainingJob",
        SearchExpression={
            "Filters": [
                {
                    "Name": "TrainingJobName",
                    "Operator": "Contains",
                    "Value": job_name_prefix,
                },
                {
                    "Name": "TrainingJobStatus",
                    "Operator": "Equals",
                    "Value": "Completed",
                },
            ]
        },
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1,
    )

    return search_response["Results"][0]["TrainingJob"]["TrainingJobName"]


job_name = get_last_job_name(job_prefix)

# Donwload fine-tuned Peft model
s3_client.download_file(
    bucket_name, f"{job_name}/{job_name}/output/model.tar.gz", "model.tar.gz"
)

#! rm -rf ./model && mkdir -p ./model && tar -xf model.tar.gz -C ./model
