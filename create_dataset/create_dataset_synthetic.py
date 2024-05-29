import json
import re
from typing import NoReturn

from tqdm import tqdm

from model import create_client
from prompts import sys_prompt_categorical_planning, usr_prompt_categorical_planning


def create_dataset_rows(count: int = 0, N: int = 10) -> NoReturn:
    """Create rows of dataset.

    Args:
        count (int, optional): Count number. Defaults to 0.
        N (int, optional): Number of dataset files each with given rows. Defaults to 25.

    Returns:
        NoReturn:
    """

    client = create_client()

    for i in tqdm(range(N)):

        file_name = f"categorical_planning_{count+i}.json"

        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system",
                    "content": sys_prompt_categorical_planning,
                },
                {"role": "user", "content": usr_prompt_categorical_planning},
            ],
            temperature=1,
        )

        result = response.choices[0].message.content

        result_match = re.search("```json", result)

        try:
            if result_match:
                result = json.loads(result.strip("```json"))
            else:
                result = json.loads(result)

            with open(f"data_v2/{file_name}", "w") as file:
                json.dump(result, file, indent=4)

        except json.decoder.JSONDecodeError:
            print("Failed to decode:")
            continue  # Continue to the next item in the loop
