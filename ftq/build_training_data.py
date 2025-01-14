from num_solver import get_dataset
import random
from prompts.prompt_feta import PROMPT_CLS, PROMPT_VERIF 
from utils.tool_func import read_list_from_file

def check_label_correct(ans_1, ans_2, label):
    # print(ans_1, ans_2, label)
    if ans_1 == ans_2:
        if random.random() < 0.5:
            return "A"
        else:
            return "B"
    elif ans_1 and ans_1[0].lower() == label[0].lower():
        return "A"
    elif ans_2 and ans_2[0].lower() == label[0].lower():
        return "B"
    else:
        return "A"
    
def check_label_verif(ans_1, ans_2, label):
    if ans_1 == label or ans_2 == label:
        return "True"
    else:
        return "False"

def generate_header(l):
    header = "| "
    add = ' | '.join(l) + ' |'
    header += add
    return header

# def refine_trainset(train_data):
#     feta_train = []
#     for i in range(len(train_data)):
#         title = train_data[i]['table_section_title']
#         header = generate_header(train_data[i]['table_array'][0])
#         question = train_data[i]['question']
#         feta_train.append([title, header, question])
#     return feta_train

def remove_blank_lines(input_string):
    # Split the string into lines
    lines = input_string.splitlines()
    # Filter out empty lines
    non_empty_lines = [line for line in lines if line.strip()]
    # Join the non-empty lines back into a single string
    output_string = '\n'.join(non_empty_lines)
    return output_string

## generate_cls_prompt using prompt template to generate each question
def generate_cls_prompt(prompt, train_info, inst_1, ans_1, inst_2, ans_2, res)->str:
    if len(ans_1) == 0:
        ans_1 = ['']
    elif len(ans_2) == 0:
        ans_2 = ['']
    prompt = prompt.replace("[TITLE]", train_info[0])
    prompt = prompt.replace("[HEADER]", generate_header(train_info[1][0]))
    prompt = prompt.replace("[QUESTION]", train_info[2])
    prompt = prompt.replace("[INST1]", remove_blank_lines(inst_1))
    prompt = prompt.replace("[ANSWER1]", ans_1[0])
    prompt = prompt.replace("[INST2]", remove_blank_lines(inst_2))
    prompt = prompt.replace("[ANSWER2]", ans_2[0])
    prompt = prompt.replace("ALPHA", res) 
    return prompt

if __name__ == '__main__':
    test_set_path = "../data_sets/ftq/ftq_test.json"
    train_set_path = "../data_sets/ftq/ftq_train.json"
    feta_train, train_ans, feta_test, test_ans = get_dataset(train_set_path, test_set_path)

    # NumSolver results
    ans_path = "./training_data/refine_propose.txt"
    inst_path = "./training_data/instruction.txt"
    refine_propose = read_list_from_file(ans_path)
    # this file contains reasoning process
    propose_inst = read_list_from_file(inst_path)

    # Mix-sc results
    mix_sc_ans_path = "./training_data/mix_sc_train.txt"
    mix_sc_inst_path = "./training_data/mix_sc_train_inst.txt"
    mix_sc_train = read_list_from_file(mix_sc_ans_path)
    # this file contains reasoning process
    mix_train_inst = read_list_from_file(mix_sc_inst_path)
    
    cnt_A = 0
    cnt_B = 0
    prompt_cls_train = []
    for i in range(len(mix_sc_train)):
        ans_1 = refine_propose[i]
        inst_1 = propose_inst[i]
        ans_2 = mix_sc_train[i]
        inst_2 = mix_train_inst[i]
        res = check_label_correct(ans_1, ans_2, train_ans[i])
        if res == "A":
            cnt_A += 1
        elif res == "B":
            cnt_B += 1
        if res:
            # print(feta_train[i])
            prompt_res = generate_cls_prompt(PROMPT_CLS, feta_train[i], inst_1, ans_1, inst_2, ans_2, res)
            prompt_cls_train.append(prompt_res)
    print("cnt A is: ", cnt_A)
    print("cnt B is: ", cnt_B)

    cnt_T = 0
    cnt_F = 0
    prompt_bin_train = []
    for i in range(len(mix_sc_train)):
        ans_1 = refine_propose[i]
        inst_1 = propose_inst[i]
        ans_2 = mix_sc_train[i]
        inst_2 = mix_train_inst[i]
        res = check_label_verif(ans_1, ans_2, train_ans[i])
        if res == "True":
            cnt_T += 1
        else:
            cnt_F += 1
        prompt_res = generate_cls_prompt(PROMPT_VERIF, feta_train[i], inst_1, ans_1, inst_2, ans_2, res)
        prompt_bin_train.append(prompt_res)
    print("cnt T is: ", cnt_T)
    print("cnt F is: ", cnt_F)

    # you can save two datasets if you want
    # file_path = "xxxx.txt"
    # write_list_to_txt(prompt_cls_l, file_path)
