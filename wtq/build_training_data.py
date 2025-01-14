import random

from utils.tool_func import read_jsonl, read_json_file, read_list_from_file, write_list_to_txt
from wtq.num_solver import get_sota_information, get_test_data, mix_sc_wrong_pair, get_all_questions, get_train_data

from prompts.prompt_wtq import PROMPT_CLS, PROMPT_VERIF


def get_sota_data(train_id, train_dataset, sota_train):
    file_path = '../results/wtq_results/wtq_cot_all/result_5.jsonl'
    result_5 = read_jsonl(file_path)
    f_1 = '../results/wtq_results/wtq-agent-all/result_sc1.jsonl'
    f_2 = '../results/wtq_results/wtq-agent-all/result_sc2.jsonl'
    f_3 = '../results/wtq_results/wtq-agent-all/result_sc3.jsonl'
    f_4 = '../results/wtq_results/wtq-agent-all/result_sc4.jsonl'
    f_5 = '../results/wtq_results/wtq-agent-all/result_sc5.jsonl'
    sc_1 = read_jsonl(f_1)
    sc_2 = read_jsonl(f_2)
    sc_3 = read_jsonl(f_3)
    sc_4 = read_jsonl(f_4)
    sc_5 = read_jsonl(f_5)
    sota_combine = []
    for x, i in enumerate(train_id):
        tmp = []
        tmp.extend(result_5[i]['text'])
        tmp.append(sc_1[i]['text'])
        tmp.append(sc_2[i]['text'])
        tmp.append(sc_3[i]['text'])
        tmp.append(sc_4[i]['text'])
        tmp.append(sc_5[i]['text'])
        # question
        tmp.append(train_dataset[x][0])
        # table_id
        tmp.append(train_dataset[x][1])
        # title
        tmp.append(train_dataset[x][3])
        tmp.append(sota_train[x][2])
        sota_combine.append(tmp)
    return sota_combine


def filter_reason(pair):
    answer = pair[-1][0]
    answer = "Final Answer: " + answer
    inst = ""
    for text in pair[:-1]:
        if text.startswith("To") and answer in text:
            return text
        elif answer in text:
            inst = text
        else:
            continue
    if inst:
        return inst
    return pair[0]


def check_bin(ans_1, ans_2, label):
    if ans_1 == label or ans_2 == label:
        return "True"
    else:
        return "False"


# get_header getting table header using table id
def get_header(i, id_table) -> str:
    table = id_table[i]
    header_context = "Header: |"
    for col in table['header']:
        header_context += " " + col + " |"
    return header_context


def remove_blank_lines(input_string):
    # Split the string into lines
    lines = input_string.splitlines()
    # Filter out empty lines
    non_empty_lines = [line for line in lines if line.strip()]
    # Join the non-empty lines back into a single string
    output_string = '\n'.join(non_empty_lines)
    return output_string


# generate_cls_prompt using prompt template to generate each question
def generate_cls_prompt(id_table, prompt, table, inst_1, ans_1, sota_info, res) -> str:
    table_id = table[1]
    header = get_header(table_id, id_table)
    question_context = table[0]
    table_title = table[3]
    prompt = prompt.replace("[TITLE]", table_title)
    prompt = prompt.replace("[HEADER]", header)
    prompt = prompt.replace("[QUESTION]", question_context)
    prompt = prompt.replace("[INST1]", remove_blank_lines(inst_1))
    prompt = prompt.replace("[ANSWER1]", ans_1)
    prompt = prompt.replace("[INST2]", remove_blank_lines(sota_info[0]))
    prompt = prompt.replace("[ANSWER2]", sota_info[-1][0])
    prompt = prompt.replace("ALPHA", res)
    return prompt


def check_label(ans_1, ans_2, label):
    if ans_1.lower() == ans_2.lower() and ans_1.lower() == label.lower():
        if random.random() >= 0.4:
            return "B"
        else:
            return "A"
    elif ans_1.lower() == label.lower():
        return "A"
    elif ans_2.lower() == label.lower():
        return "B"
    else:
        return "no results"

def check_bin(ans_1, ans_2, label):
    if ans_1 == label or ans_2 == label:
            return "True"
    else:
            return "False"


if __name__ == '__main__':
    table_qa_data = read_json_file("../data_sets/wtq/wtq.json")
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

    id_table = read_json_file("../data_sets/wtq/id_table.json")
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
    # Model A is proposed model and Model B is the current sota model
    sota_combine = get_sota_data(train_id, train_dataset, sota_train)
    sota_inst = []
    for ids, item in enumerate(sota_combine):
        pair = []
        pair.extend(item[0:10])
        if type(item[-1]) != list:
            ans = [item[-1]]
        else:
            ans = item[-1]
        pair.append(ans)
        res = filter_reason(pair)
        sota_inst.append([res, ans])

    cnt_A = 0
    cnt_B = 0

    # generate training data for AnsSelector
    file_path = "./training_data/final_res.txt"
    final_res = read_list_from_file(file_path)
    file_path = "./training_data/instruction.txt"
    instruction = read_list_from_file(file_path)
    prompt_cls_l = []
    for i in range(len(train_dataset)):
        table_info = train_dataset[i]
        ins_1 = instruction[i]
        ans_1 = final_res[i][0]
        sota_info = sota_inst[i]
        res = check_label(ans_1, sota_info[-1][0], table_info[2][0])
        if res == "A":
            cnt_A += 1
        elif res == "no results":
            continue
        else:
            cnt_B += 1
        prompt_res = generate_cls_prompt(id_table, PROMPT_CLS, table_info, ins_1, ans_1, sota_info, res)
        prompt_cls_l.append(prompt_res)
    print("cnt A is: ", cnt_A)
    print("cnt B is: ", cnt_B)

    # generate training data for TwEvaluator
    cnt_T = 0
    cnt_F = 0
    prompt_cls_v = []
    for i in range(len(train_dataset)):
        table_info = train_dataset[i]
        ins_1 = instruction[i]
        ans_1 = final_res[i][0]
        sota_info = sota_inst[i]
        res_v = check_bin(ans_1, sota_info[-1][0], table_info[2][0])
        prompt_res = generate_cls_prompt(id_table, PROMPT_VERIF, table_info, ins_1, ans_1, sota_info, res_v)
        if res_v == "True":
            cnt_T += 1
        else:
            cnt_F += 1
        prompt_cls_v.append(prompt_res)
    print("cnt T is: ", cnt_T)
    print("cnt F is: ", cnt_F)

    # you can save two datasets if you want
    # file_path = "xxxx.txt"
    # write_list_to_txt(prompt_cls_l, file_path)
