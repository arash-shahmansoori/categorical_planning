from model import create_client
from prompts import sys_prompt_graph, usr_prompt_graph


def main():
    client = create_client()

    prompt = (
        usr_prompt_graph
        + """
        {
            "Plan video content": {
                "next_modes": [
                    "Record video",
                    "Write text instructions"
                ],
                "failure": [
                    "Retry",
                    "Exit",
                    "Request Human Intervention",
                    "Log and Analyze"
                ]
            },
            "Record video": {
                "next_modes": [
                    "Edit video"
                ],
                "failure": [
                    "Retry",
                    "Switch to Different Mode",
                    "Fallback Content",
                    "Partial Content Delivery"
                ]
            },
            "Write text instructions": {
                "next_modes": [
                    "Edit text"
                ],
                "failure": [
                    "Retry",
                    "Fallback Content",
                    "Request Human Intervention",
                    "Quality Assurance Check"
                ]
            },
            "Edit video": {
                "next_modes": [
                    "Combine video and text"
                ],
                "failure": [
                    "Retry",
                    "Exit",
                    "Log and Analyze",
                    "Quality Assurance Check"
                ]
            },
            "Edit text": {
                "next_modes": [
                    "Combine video and text"
                ],
                "failure": [
                    "Retry",
                    "Fallback Content",
                    "Quality Assurance Check",
                    "Alternative Data Sources"
                ]
            },
            "Combine video and text": {
                "next_modes": [
                    "Analyzer"
                ],
                "failure": [
                    "Retry",
                    "Exit",
                    "Mode Optimization",
                    "Partial Content Delivery"
                ]
            },
            "Analyzer": {
                "next_modes": [],
                "failure": null
            }
        }
    """
    )

    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {
                "role": "system",
                "content": sys_prompt_graph,
            },
            {"role": "user", "content": prompt},
        ],
        temperature=1,
    )

    result = response.choices[0].message.content

    print(result)


if __name__ == "__main__":
    main()
