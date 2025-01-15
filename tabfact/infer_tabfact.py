def read_txt_to_list(file_path):
    with open(file_path, 'r') as file:
        # Read lines from the file and strip newline characters
        lines = [line.strip() for line in file]

    # Combine lines that are not blank into a single string
    combined_strings = []
    current_string = ''
    for line in lines:
        if line:
            current_string += line + '\n'  # Add a space to separate lines
        elif current_string:
            combined_strings.append(current_string.strip())
            current_string = ''

    # Add the last combined string if it's not empty
    if current_string:
        combined_strings.append(current_string.strip())

    return combined_strings


file_path_selector = "TabLaP/results/exp_prompts/tabfact/ans_sel_prompts.txt"
file_path_evaluator = "TabLaP/results/exp_prompts/tabfact/tw_eval_prompts.txt"
prompt_cls_t = read_txt_to_list(file_path_selector)


from datasets import Dataset

data = {
    'TD': [i for i in range(len(prompt_cls_t))],
    'text': prompt_cls_t
}
test_prompt = Dataset.from_dict(data)

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
)
import torch

### load model and tokenizer
from transformers import AutoModelForCausalLM
import transformers

print(transformers.__version__)


compute_dtype = getattr(torch, "float16")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)


from transformers.pipelines.pt_utils import KeyDataset
from transformers import pipeline


from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers.pipelines.pt_utils import KeyDataset

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    quantization_config=quant_config,
    token="YOUR_TOKEN",
    device_map='auto',
    cache_dir="cache_dir"
)
tokenizer3 = AutoTokenizer.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    trust_remote_code=True,
    token="YOUR_TOKEN",
    cache_dir="cache_dir"
)
tokenizer3.padding_side = "left"
if tokenizer3.pad_token_id is None:
    tokenizer3.pad_token_id = tokenizer3.eos_token_id

model.load_adapter("Your_Model_Path")
llama3_model = model  

# pipeline
classifier = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer3,
    max_new_tokens=3,
    # device=0,
    batch_size=10,
    pad_token_id=tokenizer3.pad_token_id
)

ans_pred = []
for res in classifier(KeyDataset(test_prompt, "text")):
    ans_pred.append(res)

refine_pred = []
for i in range(len(ans_pred)):
    refine_pred.append(ans_pred[i][0]['generated_text'])

def fetch_llama_ans(text):
    a_context = "Therefore, the final answer is: True"
    b_context = "Therefore, the final answer is: False"
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
    
pred_ans = []
for text in refine_pred:
    ans = fetch_llama_ans(text)
    pred_ans.append(ans)
    
print(pred_ans)
def write_list_to_txt(lst, file_path):
    with open(file_path, 'w') as file:
        for item in lst:
            file.write(str(item) + '\n')
    
# file_path = "tab_fact_bin.txt"
# write_list_to_txt(pred_ans, file_path)
