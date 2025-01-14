from num_solver import get_dataset
import random
from prompts.prompt_feta import PROMPT_CLS_TEST, PROMPT_VERIF_TEST
from utils.tool_func import read_list_from_file

def generate_header(l):
    header = "| "
    add = ' | '.join(l) + ' |'
    header += add
    return header

def refine_testset(test_data):
    feta_test = []
    for i in range(len(test_data)):
        title = test_data[i]['table_section_title']
        header = generate_header(test_data[i]['table_array'][0])
        question = test_data[i]['question']
        feta_test.append([title, header, question])
    return feta_test

def remove_blank_lines(input_string):
    # Split the string into lines
    lines = input_string.splitlines()
    # Filter out empty lines
    non_empty_lines = [line for line in lines if line.strip()]
    # Join the non-empty lines back into a single string
    output_string = '\n'.join(non_empty_lines)
    return output_string

## generate_cls_prompt using prompt template to generate each question
def generate_cls_prompt(prompt, test_info, inst_1, ans_1, inst_2, ans_2)->str:
    if len(ans_1) == 0:
        ans_1 = ['']
    elif len(ans_2) == 0:
        ans_2 = ['']
    prompt = prompt.replace("[TITLE]", test_info[0])
    prompt = prompt.replace("[HEADER]", generate_header(test_info[1][0]))
    prompt = prompt.replace("[QUESTION]", test_info[2])
    prompt = prompt.replace("[INST1]", remove_blank_lines(inst_1))
    prompt = prompt.replace("[ANSWER1]", ans_1[0])
    prompt = prompt.replace("[INST2]", remove_blank_lines(inst_2))
    prompt = prompt.replace("[ANSWER2]", ans_2[0])
    return prompt

if __name__ == '__main__':
    test_set_path = "../data_sets/ftq/ftq_test.json"
    train_set_path = "../data_sets/ftq/ftq_train.json"
    feta_train, train_ans, feta_test, test_ans = get_dataset(train_set_path, test_set_path)

    # NumSolver results
    ans_path = "./testing_data/num_solver_res.txt"
    inst_path = "./testing_data/instruction.txt"
    refine_propose = read_list_from_file(ans_path)
    # this file contains reasoning process
    propose_inst = read_list_from_file(inst_path)

    # Mix-sc results
    mix_sc_ans_path = "./testing_data/mix_sc_res.txt"
    mix_sc_inst_path = "./testing_data/mix_instruction.txt"
    mix_sc_test = read_list_from_file(mix_sc_ans_path)
    # this file contains reasoning process
    mix_test_inst = read_list_from_file(mix_sc_inst_path)
    

    # generate testing prompts for AnsSelector
    prompt_cls_test = []
    for i in range(len(mix_sc_test)):
        ans_1 = refine_propose[i]
        inst_1 = propose_inst[i]
        ans_2 = mix_sc_test[i]
        inst_2 = mix_test_inst[i]
        prompt_res = generate_cls_prompt(PROMPT_CLS_TEST, feta_train[i], inst_1, ans_1, inst_2, ans_2)
        prompt_cls_test.append(prompt_res)

    # generate testing prompts for TwEvaluator
    prompt_bin_test = []
    for i in range(len(mix_sc_test)):
        ans_1 = refine_propose[i]
        inst_1 = propose_inst[i]
        ans_2 = mix_sc_test[i]
        inst_2 = mix_test_inst[i]
        prompt_res = generate_cls_prompt(PROMPT_VERIF_TEST, feta_train[i], inst_1, ans_1, inst_2, ans_2)
        prompt_bin_test.append(prompt_res)

    # you can save two datasets if you want
    # file_path = "xxxx.txt"
    # write_list_to_txt(prompt_cls_l, file_path)
