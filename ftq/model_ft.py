# This script is used to fine-tune AnsSelecter and TwEvaluator
from utils.tool_func import read_txt_to_list
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
)
import torch
from peft import LoraConfig
from trl import SFTTrainer
from datasets import DatasetDict
from datasets import Dataset
from accelerate import Accelerator


def get_train_cls():
    file_path = "your_ans_selector_trainset"
    prompt_cls_l = read_txt_to_list(file_path)
    return prompt_cls_l


def get_train_verif():
    file_path = "your_tw_evaluator_trainset"
    prompt_cls_v = read_txt_to_list(file_path)
    return prompt_cls_v


def build_train_set(prompt_l):
    dict_train = {"train": {"text": []}}

    for text in prompt_l:
        dict_train["train"]["text"].append(text)
    train_dataset = DatasetDict(dict_train)
    train_set = train_dataset["train"]
    train_data = Dataset.from_dict(train_set)
    return train_data


# load llama3 model
def load_model():
    compute_dtype = getattr(torch, "float16")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )
    llama3_cls = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        quantization_config=quant_config,
        token="#########",
        device_map='auto'
    )
    llama3_cls.config.use_cache = False
    llama3_cls.config.pretraining_tp = 1

    tokenizer_cls = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", trust_remote_code=True,
                                                  token="##########")
    tokenizer_cls.pad_token = tokenizer_cls.eos_token
    tokenizer_cls.padding_side = "right"

    return llama3_cls, tokenizer_cls


def model_train(model, train_data, tokenizer):
    training_params = TrainingArguments(
        output_dir="../data_sets",
        num_train_epochs=15,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=200,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
    )
    peft_params = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )
    accelerator = Accelerator()
    trainer = accelerator.prepare(SFTTrainer(
        model=model,
        train_dataset=train_data,
        peft_config=peft_params,
        dataset_text_field="text",
        max_seq_length=3000,
        tokenizer=tokenizer,
        args=training_params,
        packing=False,
    ))
    trainer.train()
    return trainer


if __name__ == '__main__':
    train_data = get_train_cls()
    train_l = build_train_set(train_data)

    # if train verifier
    # train_data = get_train_verif()
    # train_l = build_train_set(train_data)

    model, tokenizer = load_model()
    trainer_res = model_train(model, train_l, tokenizer)
    model_name = "llama3-AnsSelecter"
    # save trained model
    trainer_res.model.save_pretrained(model_name)
    trainer_res.tokenizer.save_pretrained(model_name)
