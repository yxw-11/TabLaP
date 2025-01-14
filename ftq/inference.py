# This script is used to do inference for AnsSelecter or TwEvaluator
from datasets import Dataset

from utils.tool_func import read_txt_to_list, write_list_to_txt
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
)
import torch
from transformers.pipelines.pt_utils import KeyDataset
from transformers import pipeline
from transformers import AutoModelForCausalLM

def get_test_data(file_path):
    prompt_cls_t = read_txt_to_list(file_path)
    data = {
        'TD': [i for i in range(len(prompt_cls_t))],
        'text': prompt_cls_t
    }
    test_prompt = Dataset.from_dict(data)
    return test_prompt


# use to filter answers of verifier
def fetch_llama_verif(text):
    a_context = "Therefore, the final answer is: [True]"
    b_context = "Therefore, the final answer is: [False]"
    index_a = text.find(a_context)
    index_b = text.find(b_context)
    if index_a == -1:
        return False
    if index_b == -1:
        return True
    if index_a < index_b:
        return True
    else:
        return False


# use to filter answers of selecter
def fetch_llama_cls(text):
    a_context = "Therefore, the final answer is: [A]"
    b_context = "Therefore, the final answer is: [B]"
    index_a = text.find(a_context)
    index_b = text.find(b_context)
    if index_a == -1:
        return "[A]"
    if index_b == -1:
        return "[B]"
    if index_a < index_b:
        return "[A]"
    else:
        return "[B]"


if __name__ == '__main__':
    # use your test data path
    # either prompts for AnsSelector or TwEvaluator
    file_selector = "/TabLaP/results/exp_prompts/ftq/ans_sel_prompts.txt"
    # file_evaluator = "/TabLaP/results/exp_prompts/ftq/tw_eval_prompts.txt"
    test_prompt = get_test_data(file_selector)

    compute_dtype = getattr(torch, "float16")

    quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
    )
    # load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        quantization_config=quant_config,
        token="Your_Tokens",
        device_map='auto',
        cache_dir="cache_dir"
    )
    tokenizer_model = AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        trust_remote_code=True,
        token="Your_Tokens",
        cache_dir="cache_dir"
    )
    tokenizer_model.padding_side = "left"
    if tokenizer_model.pad_token_id is None:
        tokenizer_model.pad_token_id = tokenizer_model.eos_token_id

    # load adapter
    model.load_adapter("Model_Path")
    llama3_model = model  

    # create pipeline
    classifier = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer_model,
        max_new_tokens=3,
        # device=0,
        batch_size=10,
        pad_token_id=tokenizer_model.pad_token_id
    )

    ans_pred = []
    for res in classifier(KeyDataset(test_prompt, "text")):
        ans_pred.append(res)

    refine_pred = []
    for i in range(len(ans_pred)):
        refine_pred.append(ans_pred[i][0]['generated_text'])

    pred_ans = []
    # for verifier answers
    for text in refine_pred:
        ans = fetch_llama_verif(text)
        pred_ans.append(ans)

    # for selecter answers
    for text in refine_pred:
        ans = fetch_llama_cls(text)
        pred_ans.append(ans)

    file_path = "xxx.txt"

    # save the answers if you want
    write_list_to_txt(pred_ans, file_path)
