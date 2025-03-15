import os
import re
import subprocess
import json

from datasets import load_dataset
from data_sets.model import Model
from utils.tool_func import read_org_list
from prompts.prompt_feta import PROMPT_TEXT, PROMPT_PYTHON


def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        return data

# this function is used for getting training and testing data
def get_dataset(train_path, test_path):
    feta_train = []
    train_ans = []

    train_data = read_json_file(train_path)
    test_data = read_json_file(test_path)

    for i, _ in enumerate(train_data):
        title = train_data[i]['table_section_title']
        context = train_data[i]['table_array']
        question = train_data[i]['question']
        feta_train.append([title, context, question])
        train_ans.append(train_data[i]['answer'])

    feta_test = []
    test_ans = []

    for i, _ in enumerate(test_data):
        title = test_data[i]['table_section_title']
        context = test_data[i]['table_array']
        question = test_data[i]['question']
        feta_test.append([title, context, question])
        test_ans.append(test_data[i]['answer'])

    return feta_train, train_ans, feta_test, test_ans


def build_tables(header, rows):
    header_context = "Header: |"
    for col in header:
        header_context += " " + col + " |"
    header_context += "\n"
    row_context = "Rows: "
    for r in rows:
        row_context += "|"
        for c in r:
            row_context += " " + c + " |"
        row_context += "\n"
    final_table = header_context + row_context
    return final_table


def extract_reasoning(input_string):
    final_answer_index = input_string.find("```python")
    content = input_string[:final_answer_index].strip()
    return content


# generate_prompt using prompt template to generate each question
def generate_prompt(table, question, title, prompt_temp: str) -> str:
    prompt = prompt_temp.replace("[TITLE]", title)
    prompt = prompt.replace("[TABLE]", table)
    prompt = prompt.replace("[QUESTION]", question)
    return prompt


# get tables for training and testing
def get_tables(train_data, test_data):
    feta_tables = []
    for i in range(len(test_data)):
        header = test_data[i][1][0]
        rows = test_data[i][1][1:]
        feta_tables.append(build_tables(header, rows))

    feta_train_tables = []
    for i in range(len(train_data)):
        header = train_data[i][1][0]
        rows = train_data[i][1][1:]
        feta_train_tables.append(build_tables(header, rows))
    return feta_tables, feta_train_tables


# get final answer
def filter_answer(raw_context) -> str:
    pattern = r'Final Answer: (.*)'
    result = re.findall(pattern, raw_context)
    print(result)
    return result


def all_numbers(input_string):
    flag = True
    for w in input_string.split(','):
        numbers = re.findall(r'\d+', w)
        if not numbers:
            flag = False
    return flag


def extract_numbers(input_string):
    # use regex to find numerical values
    numbers = re.findall(r'\d+', input_string)
    return numbers if numbers else None


def query_openai(model, prompt, temperature):
    # encode the prompt to get the length of the prompt

    response = model.query(
        prompt=prompt,
        temperature=temperature if temperature is None else temperature,
    )
    return response


def run_string(code_string):
    try:
        # Run the script in a subprocess
        result = subprocess.run(['python', '-c', code_string], capture_output=True, text=True, check=True)

        # Print the captured output
        return (result.stdout.strip())

    except subprocess.CalledProcessError as e:
        return (f"An error occurred: {e}")


# used to extract python code from the results of math_solve_tab
def extract_python_code(input_string):
    # Define the pattern to match Python code
    pattern = r'```python(.*?)```'

    # Use re.findall to find all matches
    matches = re.findall(pattern, input_string, re.DOTALL)

    # Return the matched Python code
    return matches[0]


def call_model(prompt_t, prompt_p, model):
    refine_propose_1 = []
    org_propose_1 = []
    instruction_1 = []

    for i in range(len(prompt_p)):
        python_res = ""
        final_res = ""
        respon = query_openai(model, prompt_t[i], 0)
        instruct = respon
        org_propose_1.append(respon)
        final_res = filter_answer(respon)
        if final_res and all_numbers(final_res[0]):
            respon = query_openai(model, prompt_p[i], 0)
            instruct = extract_reasoning(respon)
            if "python" in respon:
                try:
                    python_code = extract_python_code(respon)
                    python_res = run_string(python_code)
                    if "Final Answer" in python_res and "error" not in python_res:
                        python_res = filter_answer(python_res)
                except:
                    python_res = ""
            if python_res and "error" not in python_res and all_numbers(python_res[0]):
                final_res = python_res
        instruction_1.append(instruct)
        if final_res:
            refine_propose_1.append([final_res[0]])
        else:
            refine_propose_1.append(final_res)
    return refine_propose_1, org_propose_1, instruction_1


def filter_answer_omit(raw_context) -> str:
    pattern = r'the final answer is: (.*)'
    result = re.findall(pattern, raw_context)
    return result


if __name__ == '__main__':
    train_set_path = "data_sets/ftq/ftq_test.json"
    test_set_path = "data_sets/ftq/ftq_train.json"
    feta_train, train_ans, feta_test, test_ans = get_dataset(train_set_path, test_set_path)
    test_tables, train_tables = get_tables(feta_train, feta_test)
    # generate prompts
    prompt_t = []
    for i in range(len(feta_test)):
        prompt_i = generate_prompt(test_tables[i], feta_test[i][-1], feta_test[i][0], PROMPT_TEXT)
        prompt_t.append(prompt_i)
    prompt_p = []
    for i in range(len(feta_test)):
        prompt_i = generate_prompt(test_tables[i], feta_test[i][-1], feta_test[i][0], PROMPT_PYTHON)
        prompt_p.append(prompt_i)
    model = Model(model_name="gpt-3.5-turbo-0125", provider="openai")
    refine_ans, org_ans, instructions = call_model(prompt_t, prompt_p, model)
    # further refine the answers
    for i, ans in enumerate(refine_ans):
        if not ans:
            refine_ans[i] = filter_answer_omit(org_ans[i])
        if feta_test[i]['question'].startswith("Did"):
            if "no" not in refine_ans[i][0] or "No" not in refine_ans[i][0]:
                refine_ans[i] = ["Yes"]
    # evaluate
    
