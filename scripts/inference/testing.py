from preprocess_data import format_prediction_instruct_fn
from tqdm import tqdm
from utils import get_completion


def testing_planner(
    model, tokenizer, stop_keywords, KeywordsStoppingCriteria, test_data
):
    outputs = []
    for i in tqdm(range(len(test_data))):
        pred_prompt = format_prediction_instruct_fn(test_data[i])

        output = get_completion(
            pred_prompt, model, tokenizer, stop_keywords, KeywordsStoppingCriteria
        )

        outputs.append(output)

    return outputs
