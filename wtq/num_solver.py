import json
import os
import random
import ast
import re
import subprocess

from prompts.prompt_wtq import PROMPT_MATH_SOLVER
from data_sets.model import Model
from wtq.evaluate import correct_num
from utils.tool_func import read_json_file, write_list_to_txt


def get_all_questions(table_qa):
    questions = []
    for ids in range(len(table_qa)):
        for idx, q in enumerate(table_qa[ids]['questions']):
            tmp = [table_qa[ids]['questions'][idx], table_qa[ids]['table_id'], table_qa[ids]['answers'][idx],
                   table_qa[ids]['title']]
            questions.append(tmp)
    return questions


def get_train_data(table_qa):
    random.seed(100)
    indices_range = range(421)
    random_indices = random.sample(indices_range, 211)

    train_dataset = []

    for ids in random_indices:
        for idx, q in enumerate(table_qa[ids]['questions']):
            tmp = [table_qa[ids]['questions'][idx], table_qa[ids]['table_id'], table_qa[ids]['answers'][idx],
                   table_qa[ids]['title']]
            train_dataset.append(tmp)
    return train_dataset


def get_test_data(table_qa):
    random.seed(100)
    indices_range = range(421)
    random_indices = random.sample(indices_range, 211)
    test_index = [item for item in indices_range if item not in random_indices]
    test_dataset = []

    for ids in test_index:
        for idx, q in enumerate(table_qa[ids]['questions']):
            tmp = [table_qa[ids]['questions'][idx], table_qa[ids]['table_id'], table_qa[ids]['answers'][idx],
                   table_qa[ids]['title']]
            test_dataset.append(tmp)
    return test_dataset


# get_table getting table using table id
def get_table(table_id) -> str:
    table = id_table[table_id]
    header_context = "Header: |"
    for col in table['header']:
        header_context += " " + col + " |"
    header_context += "\n"
    row_context = "Rows: "
    for r in table['rows']:
        row_context += "|"
        for c in r:
            row_context += " " + c + " |"
        row_context += "\n"
    final_table = header_context + row_context
    return final_table


def mix_sc_wrong_pair():
    wrong_pair = []
    # Open the text file in read mode
    with open('../data_sets/wrong_pair.txt', 'r', encoding='utf-8') as file:
        # Read the lines of the file into a list
        for line in file:
            # Remove newline character and append the line to the list
            line_list = line.strip()
            list_from_string = ast.literal_eval(line_list)
            wrong_pair.append(list_from_string)
    return wrong_pair


# current sota model is mix-sc
def get_sota_information(table_info, wrong_pairs, train_data, test_data, train_i, test_i):
    sota_labels_all = table_info.copy()
    for item in wrong_pairs:
        idx = item[0] - 1
        label = item[2]
        sota_labels_all[idx][2] = label
    sota_train = train_data.copy()
    for i, idx in enumerate(train_i):
        sota_train[i] = sota_labels_all[idx]
    sota_test = test_data.copy()
    for i, idx in enumerate(test_i):
        sota_test[i] = sota_labels_all[idx]
    sota_labels = []
    for tab in sota_train:
        if type(tab[2]) != list:
            sota_labels.append([tab[2]])
        else:
            sota_labels.append(tab[2])
    truth_labels = []
    for tab in train_data:
        truth_labels.append(tab[2])
    test_labels = []
    for tab in test_data:
        test_labels.append(tab[2])
    return sota_test, sota_train, sota_labels, truth_labels, test_labels


# used to extract python code from the results of math_solve_tab
def extract_python_code(input_string):
    # Define the pattern to match Python code
    pattern = r'```python(.*?)```'

    # Use re.findall to find all matches
    matches = re.findall(pattern, input_string, re.DOTALL)

    # Return the matched Python code
    return matches[0]


# extract reasoning process of models
def extract_reasoning(input_string):
    final_answer_index = input_string.find("Final Answer")
    content = input_string[:final_answer_index].strip()
    final_answer_index = content.find("```python")
    content = content[:final_answer_index].strip()
    if content.endswith('Python script:'):
        final_answer_index = content.find("Python script:")
        content = content[:final_answer_index].strip()
    if content.endswith('Python Script:'):
        final_answer_index = content.find("Python Script:")
        content = content[:final_answer_index].strip()
    return content


# used to run python code in text format
def run_string(code_string):
    try:
        # Run the script in a subprocess
        result = subprocess.run(['python', '-c', code_string], capture_output=True, text=True, check=True)

        # Print the captured output
        return (result.stdout.strip())

    except subprocess.CalledProcessError as e:
        return (f"An error occurred: {e}")


# generate_prompt using prompt template to generate each question
def generate_prompt(question: list, prompt_temp: str) -> str:
    table_id = question[1]
    table_context = get_table(table_id)
    question_context = question[0]
    table_title = question[3]
    prompt = prompt_temp.replace("[TITLE]", table_title)
    prompt = prompt.replace("[TABLE]", table_context)
    prompt = prompt.replace("[QUESTION]", question_context)
    return prompt


def filter_answer(raw_context) -> str:
    pattern = r'Final Answer: (.*)'
    result = re.findall(pattern, raw_context)
    return result


def query_openai(model, prompt, temperature):
    # encode the prompt to get the length of the prompt

    response = model.query(
        prompt=prompt,
        temperature=temperature if temperature is None else temperature,
    )
    return response


def call_model(prompt_list, model):
    refine_propose = []
    org_propose = []
    instruction = []

    for i in range(len(prompt_list)):
        filter_ans = ""
        python_res = ""
        respon = query_openai(model, prompt_list[i], 0)
        org_propose.append(respon)
        instruct = extract_reasoning(respon)
        instruction.append(instruct)
        if "python" in respon:
            try:
                python_code = extract_python_code(respon)
                python_res = run_string(python_code)
                if "Final Answer" in python_res and "error" not in python_res:
                    python_res = filter_answer(python_res)
            except:
                python_res = ""
            respon = respon.replace(python_code, "")
        if "Final Answer" in respon:
            filter_ans = filter_answer(respon)

        if not filter_ans:
            refine_propose.append(python_res)
        elif not python_res:
            refine_propose.append(filter_ans)
        elif filter_ans and python_res:
            if "error" in python_res:
                refine_propose.append(filter_ans)
            else:
                if type(python_res) != list and type(
                        filter_ans) != list and python_res.isdigit() and not filter_ans.isdigit():
                    refine_propose.append(filter_ans)
                else:
                    refine_propose.append(python_res)
        else:
            refine_propose.append("no result")
    return refine_propose, org_propose, instruction


def evaluate_res(labels, predict):
    num = correct_num(labels, predict)
    return num


if __name__ == '__main__':
    table_qa_data = read_json_file("data_sets/wtq/wtq.json")
    question_all = get_all_questions(table_qa_data)
    train_dataset = get_train_data(table_qa_data)

    # training data_id
    train_id = []
    for x in train_dataset:
        for i, y in enumerate(question_all):
            if x == y:
                train_id.append(i)
    # testing data_id
    test_id = []
    for i in range(len(question_all)):
        if i not in train_id:
            test_id.append(i)

    id_table = read_json_file("data_sets/wtq/id_table.json")
    test_dataset = get_test_data(table_qa_data)
    wrong_pair = mix_sc_wrong_pair()
    # table_q is the information for all the test data
    table_q = []
    for i in range(len(table_qa_data)):
        tab = table_qa_data[i]
        q_l = tab["questions"]
        title = tab["title"]
        ans_l = tab["answers"]
        table_id = tab["table_id"]
        for idx, q in enumerate(q_l):
            table_q.append([q, table_id, ans_l[idx], title])
    # some information for the previous sota model mix-sc
    sota_test, sota_train, sota_labels, truth_labels, test_labels = get_sota_information(table_q, wrong_pair,
                                                                                         train_dataset, test_dataset,
                                                                                         train_id, test_id)
    # file_path = "../results/wtq_results/test_labels.txt"
    # write_list_to_txt(test_labels, file_path)

    # get prompt for each question
    prompt_list = []
    for q_l in train_dataset:
        prompt_i = generate_prompt(q_l, PROMPT_MATH_SOLVER)
        prompt_list.append(prompt_i)
    # set openai api key
    os.environ["OPENAI_API_KEY"] = "your_key"
    # openai model
    model = Model(model_name="gpt-3.5-turbo-0125", provider="openai")

    # obtain the answers
    refine_ans, org_ans, instructions = call_model(prompt_list, model)

    num = evaluate_res(refine_ans, test_labels)

    print("Accuracy is: ", num / len(refine_ans))
