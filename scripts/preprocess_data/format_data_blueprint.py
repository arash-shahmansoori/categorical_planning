from type_extension import DataPointType


def format_dataset_blueprint_fn(data_points: DataPointType) -> DataPointType:
    template = "Below is an instruction that describes a composite task. Provide the structured blueprint planning for the following task."

    INST = "### Instruction:"
    RESP = "### Response:"

    # Create the formatted text
    formatted_dataset = data_points.map(
        lambda x: {
            "prompt": "".join(
                [
                    f"{template}\n\n",
                    f"{INST}{x['Instruction'].strip()}\n\n",
                    f"{RESP}{x['Blueprint'].strip()}",
                ]
            ),
            "response": "".join(
                [
                    f"{RESP}{x['Blueprint'].strip()}",
                ]
            ),
        }
    )
    return formatted_dataset


def format_dataset_blueprint_instruct_fn(data_points: DataPointType) -> DataPointType:
    template = "Below is an instruction that describes a composite task. Provide the structured blueprint planning for the following task."

    B_INST, E_INST = "[INST]", "[/INST]"
    RESP = "### Response:"

    # Create the formatted text
    formatted_dataset = data_points.map(
        lambda x: {
            "prompt": "".join(
                [
                    f"{B_INST}{template.strip()}{x['Instruction'].strip()}{E_INST}\n\n",
                    f"{RESP}{x['Blueprint']}",
                ]
            ),
            "response": "".join(
                [
                    f"{RESP}{x['Blueprint']}",
                ]
            ),
        }
    )
    return formatted_dataset
