import torch
from transformers import StoppingCriteria
from type_extension import List, LlamaTokenizerFast, MistralForCausalLM


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if (
                len(cur_keyword_ids) > 1
                and cur_keyword_ids[0] == tokenizer.bos_token_id
            ):
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def call_for_batch(
        self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [
            keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids
        ]
        for keyword_id in self.keyword_ids:
            if (output_ids[0, -keyword_id.shape[0] :] == keyword_id).all():
                return True
        outputs = self.tokenizer.batch_decode(
            output_ids[:, -offset:], skip_special_tokens=True
        )[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False

    def __call__(
        self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        outputs = []
        for i in range(output_ids.shape[0]):
            outputs.append(self.call_for_batch(output_ids[i].unsqueeze(0), scores))
        return all(outputs)


def get_completion(
    prompt: str,
    model: MistralForCausalLM,
    tokenizer: LlamaTokenizerFast,
    keywords: List[str],
    keyword_stopping_criteria: KeywordsStoppingCriteria,
) -> str:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    stopping_criteria = keyword_stopping_criteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            inputs=input_ids,
            use_cache=True,
            max_new_tokens=512,
            stopping_criteria=[stopping_criteria],
            attention_mask=None,
            do_sample=False,
            output_attentions=True,
            output_scores=True,
            return_dict_in_generate=True,
        )

    output = tokenizer.decode(
        output_ids["sequences"][0, input_ids.shape[1] :], skip_spectial_tokens=True
    ).strip()

    return output
