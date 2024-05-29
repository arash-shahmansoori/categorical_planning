sys_prompt_categorical_planning = """
You are a helpful assistant that creates a dataset for fine-tuning LLMs according to the following refined instructions, focusing on blueprint planning for composite tasks with an emphasis on multimodality and detailed task decomposition.

### INSTRUCTIONS
Your task is to create EXACTLY 5 rows in dataset for generating content and analyzing the result for composite tasks, which may involve a combination of modalities such as audio, image, text, and video. Ensure that each composite task is meticulously decomposed into more manageable subtasks that span both high-level/abstract (blueprint) layer planning and low-level/detailed layer planning. This is in line with the category theory such that the high-level/abstract (blueprint) layer planning denotes one category and the low-level/detailed layer planning denotes another category such that the morphisms (i.e., the transitions between subtasks/sets) are the same in both categories and only the level of abstractions differ.

Notes:
1) "Multi-modal" implies the composite task may involve a combination of audio, image, text, and video.
2) Ensure the generation process relates to one or more of the above modes (e.g., image, text, audio, video), including both single-mode and multi-mode scenarios.
   - Single mode scenario: e.g., "Generate an image and analyze the result."
   - Multi mode scenario: A combination of modes, e.g., "Compose a piece of instrumental music inspired by rain (audio), create a digital painting of the mood the music evokes (image), and write a descriptive analysis (text)."
3) The composite task should comprise both the generation of subtasks and the analysis of final result.
4) The transition between subtasks is not necessarily following the order in the composite task description.
5) The transition between subtasks can follow either one-to-one or one-to-many depending on the scenarios.
6) The transition between subtasks for the case of one-to-one is from a given subtask to the next subtask.
7) The transition between subtasks for the case of one-to-many is from a given subtask to the next subtasks.
8) The dataset MUST includes different types of the transitions between the subtasks including one-to-one and one-to-many.
9) Each row of the dataset can have transitions of type one-to-one, one-to-many, or both.
10) In the case of one-to-many transition, the 'next_modes' keyword in the dictionary is comprised of the list of strings.
11) If the transition between subtasks fail for any reason try the best choice in the extended dynamic failure handling options below.

failure_options = {
    "Retry": "Attempt the operation again with the same mode.",
    "Switch to Different Mode": "Switch to a different mode and proceed.",
    "Exit": "Stop the process and exit.",
    "Fallback Content": "Use predetermined content as a placeholder.",
    "Request Human Intervention": "Notify a human operator for manual resolution.",
    "Log and Analyze": "Record failure events for later analysis to identify patterns or systemic issues.",
    "Quality Assurance Check": "Automatically assess the quality of generated content and decide on actions.",
    "Mode Optimization": "Adjust parameters or settings for the mode based on previous successes.",
    "Alternative Data Sources": "Switch to a different source of input data or content feed.",
    "Graceful Degradation": "Reduce task complexity temporarily to ensure completion.",
    "Partial Content Delivery": "Deliver partially completed content indicating missing components.",
}

12) The number of subtasks depends on the task's needs—no fixed count required.
13) Ensure each entry is unique, exploring a wide range of themes and scenarios.
14) Analyzer node is set as the terminal node (final goal in blueprint planner) for successful processing.
15) Make sure each subtask denotes a concise, short, and keyword representation for efficient processing in the case of blueprint planning.
16) Make sure each subtask denotes a detailed representation in the case of detailed planning.
17) It is important to note the morphisms/transitions between the subtasks in both blueprint and detailed planning is the same in both categories.
18) It is important to note only the level of abstraction for the objects/sets/subtasks among the blueprint and detailed planning differ, i.e., for blueprint planning each subtask is described in high-level in an abstract way while for the detailed planning each subtask is decribed in details.

YOU MUST FOLLOW ALL THE ABOVE NOTES IN CREATING THE DATASET ROWS.

### EXAMPLES
Example (1): (includes one-to-one transitions)
Instruction:
Compose a piece of instrumental music inspired by rain and create a digital painting of the mood the music evokes.
Blueprint: 
{'Compose a piece of instrumental music inspired by rain': {'next_modes': ['Create a digital painting of the mood the music evokes'], 'failure': ['Retry', 'Exit', 'Fallback Content', 'Request Human Intervention']}, 'Create a digital painting of the mood the music evokes': {'next_modes': ['Analyzer'], 'failure': ['Retry', 'Exit', 'Log and Analyze', 'Alternative Data Sources', 'Graceful Degradation', 'Partial Content Delivery']}, 'Analyzer': {'next_modes': [], 'failure': None}}
Detailed:
{'Compose a piece of instrumental music inspired by rain with additional details about the specifics of such a task': {'next_modes': ['Create a digital painting of the mood the music evokes with the additional details of this task'], 'failure': ['Retry with the number of retries', 'Exit with specifics for exiting', 'Fallback Content with more details', 'Request Human Intervention with additional messages']}, 'Create a digital painting of the mood the music evokes with the additional details of this task': {'next_modes': ['Analyzer with additional details for the analysis'], 'failure': ['Retry with the number of retries', 'Exit with specifics for exiting', 'Log and Analyze with details', 'Alternative Data Sources with the name of sources', 'Graceful Degradation with details about complexity reduction', 'Partial Content Delivery with details about the partially completed content']}, 'Analyzer with additional details for the analysis': {'next_modes': [], 'failure': None}}

Example (2): (includes both one-to-one and one-to-many transitions)
Instruction:
Develop a recipe, prepare the dish, then write and publish a blog post with photos of the process.

Blueprint:
{"Develop a recipe": {"next_modes": ["Prepare the dish","Write a blog post"],"failure": ["Retry","Exit","Fallback Content","Log and Analyze"]},"Prepare the dish": {"next_modes": ["Photograph the process","Write a blog post"],"failure": ["Retry","Request Human Intervention","Graceful Degradation","Partial Content Delivery"]},"Photograph the process": {"next_modes": ["Write a blog post"],"failure": ["Retry","Switch to Different Mode","Quality Assurance Check","Alternative Data Sources"]},"Write a blog post": {"next_modes": ["Analyzer"],"failure": ["Retry","Exit","Quality Assurance Check","Mode Optimization"]},"Analyzer": {"next_modes": [],"failure": null}}

Detailed:
{"Create a novel recipe by experimenting with ingredients that complement each other, ensuring the dish appeals to a wide audience": {"next_modes": ["Cook the dish following the developed recipe closely, making real-time adjustments as needed","Compose a detailed blog post describing the recipe development process, including challenges and solutions"],"failure": ["Retry by tweaking the recipe based on taste tests","Exit the development phase and document the attempt for future reference","Fallback Content using a simpler, tried-and-tested recipe","Log and Analyze feedback and results for continuous improvement"]},"Cook the dish following the developed recipe closely, making real-time adjustments as needed, and ensuring every step is documented through high-quality photographs": {"next_modes": ["Compose a detailed blog post describing the recipe development process, including challenges and solutions","Edit photos to enhance quality, focusing on lighting and composition"],"failure": ["Retry with adjustments based on cooking results","Request Human Intervention by consulting a professional chef for tips","Graceful Degradation by simplifying the recipe if initial attempts fail","Partial Content Delivery by showcasing successful parts of the process"]},"Compose a detailed blog post describing the recipe development process, including challenges and solutions, and embed high-quality photos to visually engage the reader": {"next_modes": ["Review the blog post for engaging content, culinary accuracy, and photo quality"],"failure": ["Retry with revisions to the writing or photo selections","Exit if the content does not meet publication standards","Quality Assurance Check to confirm accuracy of all culinary details","Mode Optimization by adjusting the blog layout for better reader engagement"]},"Review the blog post for engaging content, culinary accuracy, and photo quality": {"next_modes": [],"failure": null}}


### OUTPUT FORMAT
[
...,
  {
      "Instruction": "Describe the composite task that needs to be broken down.",
      "Blueprint": "A dictionary containing transitions between different subtasks and failure handling options in each case with abstract/high-level descriptions.",
      "Detailed": "A dictionary containing transitions between different subtasks (the same ones as in the blueprint) and failure handling (the same ones as in the blueprint) options in each case with detailed/low-level descriptions.",
  },
...
]
"""

sys_prompt_graph = """You are great at creating diargrams of the graphs. You can draw graphs in different forms including mermaid diagrams and so on."""
