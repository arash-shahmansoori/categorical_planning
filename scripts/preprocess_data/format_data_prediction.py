from type_extension import DataPointType


def format_prediction_blueprint_fn(data_point: DataPointType) -> str:
    template = "Below is an instruction that describes a composite task. Provide the structured blueprint planning for the following task."

    INST = "### Instruction:"
    RESP = "### Response:"

    # Create the formatted text
    formatted_prediction = "".join(
        [
            f"{template}\n\n",
            f"{INST}{data_point['Instruction'].strip()}\n\n",
            f"{RESP}",
        ]
    )
    return formatted_prediction


def format_prediction_blueprint_instruct_fn(data_point: DataPointType) -> str:
    template = "Below is an instruction that describes a composite task. Provide the structured blueprint planning for the following task."

    B_INST, E_INST = "[INST]", "[/INST]"
    RESP = "### Response:"

    # Create the formatted text
    formatted_prediction = "".join(
        [
            f"{B_INST}{template.strip()}{data_point['Instruction'].strip()}{E_INST}\n\n",
            f"{RESP}",
        ]
    )
    return formatted_prediction


def format_prediction_detail_fn(data_point: DataPointType) -> str:
    system_prompt = "Below is an instruction that describes a composite task. Provide the structured detail planning for the following task."

    INST = "### Instruction:"
    RESP = "### Response:"

    # Create the formatted text
    formatted_prediction = "".join(
        [
            f"{system_prompt}\n\n",
            f"{INST}{data_point['Instruction'].strip()}\n\n",
            f"{RESP}",
        ]
    )
    return formatted_prediction


def format_prediction_detail_instruct_fn(data_point: DataPointType) -> str:
    template = "Below is an instruction that describes a composite task. Provide the structured detail planning for the following task."

    B_INST, E_INST = "[INST]", "[/INST]"
    RESP = "### Response:"

    # Create the formatted text
    formatted_prediction = "".join(
        [
            f"{B_INST}{template.strip()}{data_point['Instruction'].strip()}{E_INST}\n\n",
            f"{RESP}",
        ]
    )
    return formatted_prediction
