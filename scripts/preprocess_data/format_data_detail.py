from type_extension import DataPointType


def format_dataset_detail_fn(data_points: DataPointType) -> DataPointType:
    template = "Below is an instruction that describes a composite task. Provide the structured detail planning for the following task."

    INST = "### Instruction:"
    RESP = "### Response:"

    # Create the formatted text
    formatted_dataset = data_points.map(
        lambda x: {
            "prompt": "".join(
                [
                    f"{template}\n\n",
                    f"{INST}{x['Instruction'].strip()}\n\n",
                    f"{RESP}{x['Detailed'].strip()}",
                ]
            ),
            "response": "".join(
                [
                    f"{RESP}{x['Detailed'].strip()}",
                ]
            ),
        }
    )
    return formatted_dataset


def format_dataset_detail_instruct_fn(data_points: DataPointType) -> DataPointType:
    template = "Below is an instruction that describes a composite task. Provide the structured detail planning for the following task."

    B_INST, E_INST = "[INST]", "[/INST]"
    RESP = "### Response:"

    # Create the formatted text
    formatted_dataset = data_points.map(
        lambda x: {
            "prompt": "".join(
                [
                    f"{B_INST}{template.strip()}{x['Instruction'].strip()}{E_INST}\n\n",
                    f"{RESP}{x['Detailed'].strip()}",
                ]
            ),
            "response": "".join(
                [
                    f"{RESP}{x['Detailed'].strip()}",
                ]
            ),
        }
    )
    return formatted_dataset
