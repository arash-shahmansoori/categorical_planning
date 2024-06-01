# Categorical Planning for Multi-Modal Tasks in Large Language Models

![Train-Validation-Loss](assets/cat_plan_2.png)


## Abstract

This research explores the significance of planning ahead in artificial intelligence, specifically within multi-modal composite tasks. We introduce a novel planning method that incorporates abstraction and category theory to enhance planning efficiency. Our approach uses concepts from category theory—objects and morphisms, redefined as subtasks and transitions—to establish planning categories at varying levels of abstraction. Our main contributions include the development of a specialized dataset for multi-modal planning, the application of category theory for detailed and abstract planning, and the introduction of a structured learning approach for integrating information across different planning categories. This work ultimately aims to improve the quality of multi-modal generated content through effective transition and transformation techniques.

## How to Use

### Quick Start
To begin exploring and utilizing the proposed method please refer to the `notebooks`.

Alternatively, you can run the scripts locally by following the steps below.

```sh
python -m venv .venv
```

```sh
source .venv/bin/activate
```

Upgrade `pip` and install all the necessary requirements as follows.

```sh
pip install --upgrade pip
```

### Installation

Install all the necessary requirements.

```sh
pip install -r requirements.txt
```

### Dataset Generation

To create the dataset from scratch use the script provided in the `run_dataset_creation.py`. We have already provided the 1k rows of the dataset for you to use in the `data` directory.

### Environment Variables

Create a .env file and set your ```openai``` API key and other required API keys of your choice, e.g., wandb and huggingface.


### Fine-Tuning

To run the fine-tuning, save the fine-tuned model and tokenizer, and the evaluation results use the following command.

For blueprint category:

```sh
python train_categorical_blueprint_planning_aws.py
```

For detailed category:

```sh
python train_categorical_detail_planning_aws.py
```

For merged categorical planning:

```sh
python merge_categorical_blueprint_detail_planning.py
```

## Checkpoints

The checkpoints for blueprint, detailed, and merged adapters have been made available in the following HuggingFace repositories, respectively:

```sh
arashmsn/Blueprint_Planning_Mistral-7B-Instruct-v0.2
arashmsn/Detailed_Planning_Mistral-7B-Instruct-v0.2
arashmsn/Merge_Planning_Mistral-7B-Instruct-v0.2
```

## Results

The evaluation results for categorical planning using blueprint and detailed planning adapters together with the merged adapter for multi-modal composite tasks are stored in the `results` directory. Other metrics including training and evaluation losses are provided in the `assets` directory.

To plot the categorical planner in the form of a graph using mermaid use `plot_graph.py`.

## Author

Arash Shahmansoori (arash.mansoori65@gmail.com)

## License

This project is open-sourced under the [MIT License](LICENSE), allowing for widespread use and contribution back to the community. Please refer to the [MIT License](LICENSE) file for detailed terms and conditions.

## Acknowledgements

We extend our gratitude to the AI research community for the open-source tools that have made this work possible. Special thanks to the contributors of the Mistral LLM and the creators of the QLORA technique for their groundbreaking contributions to the field of AI.

---

We invite contributors, researchers, and AI enthusiasts to join us in advancing the field of categorical planning for multi-modal AI research. Together, we can build powerful AI systems capable of understanding and executing complex composite tasks.
