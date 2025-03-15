import re
import random
import numpy as np

from utils.tool_func import read_txt_to_list, read_list_from_file
from common_eval.evaluator import official_eval


def get_verif_res():
    path = "results/ftq_results/TabLaP/tw_evaluator_res.txt"
    bin_res = read_txt_to_list(path)
    bin_res = bin_res[0].split('\n')
    false_id = []
    for j, flag in enumerate(bin_res):
        if flag == "False":
            false_id.append(j)
    return false_id


def combine_elements_into_list(elements):
    combined_string = ', '.join(elements)
    return [combined_string]

def extract_numbers(input_string):
    numbers = re.findall(r'\d+', input_string)
    return numbers if numbers else None

# determine whether a single answer is correct
# Note: for numerical answers with unit; we only consider the correctness of numerical values
def check_single_ans(model_ans, label):
    text_p = model_ans[0].lower().replace(" ","")
    text_l = label[0].lower().replace(" ", "")
    if text_p == text_l:
        return True
    elif set(text_p.split(',')) == set(text_l.split(',')):
        return True
    elif len(text_p.split(',')) == len(text_l.split(',')):
        flag = True
        ans = text_p.split(',')
        lab = text_l.split(',')
        for j in range(len(ans)):
            if ans[j] == lab[j]:
                continue
            elif extract_numbers(ans[j]) and extract_numbers(ans[j]) == extract_numbers(lab[j]):
                continue
            else:
                flag =False
        if flag:             
            return True
    else:
        return False

def expanding_window(model_ans, sota_ans, combine_ans, test_labels, false_id):
    res_l = []
    # this number can be modified
    number = 120

    for x in range(5000):
        right_case = 0
        for i in range(number):
            fd = false_id[i]
            if model_ans[fd] != test_labels[fd] and sota_ans[fd] != test_labels[fd]:
                right_case += 1
        prob = right_case / number

        for i in range(number, len(false_id)):
            fd = false_id[i]
            if random.random() >= prob:
                res = combine_ans[fd]
                if res == test_labels[fd]:
                    right_case += 1
            else:
                if model_ans[fd] != test_labels[fd] and sota_ans[fd] != test_labels[fd]:
                    right_case += 1
            prob = right_case / (i + 1)
        res_l.append(right_case)
    return np.mean(res_l)

class UCBBandit:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.arm_counts = np.zeros(n_arms)  # number of arm chosen
        self.arm_rewards = np.zeros(n_arms)  # accumulate reward
        self.arm_means = np.zeros(n_arms)  # avg reward
        self.total_count = 0  # total number of choices

    def choose_arm(self):
        if self.total_count < self.n_arms:
            return self.total_count 
        else:
            ucb_values = self.arm_means + np.sqrt(2 * np.log(self.total_count) / self.arm_counts)
            return np.argmax(ucb_values)

    def update(self, chosen_arm, reward):
        self.total_count += 1
        self.arm_counts[chosen_arm] += 1
        self.arm_rewards[chosen_arm] += reward
        self.arm_means[chosen_arm] = self.arm_rewards[chosen_arm] / self.arm_counts[chosen_arm]

def mab_ucb(model_ans, sota_ans, combine_ans, test_labels, false_id):
    n_arms = 2
    n_rounds = len(false_id)
    bandit = UCBBandit(n_arms)

    round_ans = []
    for _ in range(5000):
        chosen_arms = []
        for i in range(n_rounds):
            fd = false_id[i]
            chosen_arm = bandit.choose_arm()
            if chosen_arm == 0:
                if model_ans[fd] != test_labels[fd] and sota_ans[fd] != test_labels[fd]:
                    bandit.update(chosen_arm, 1)
                else:
                    bandit.update(chosen_arm, -1)
            else:
                if combine_ans[fd] == test_labels[fd]:
                    bandit.update(chosen_arm, 1)
                else:
                    bandit.update(chosen_arm, -1)
            chosen_arms.append(chosen_arm)
        round_ans.append(chosen_arms)

    cnt_all = []
    for chosen_arms in round_ans:
        cnt = 0
        for i in range(n_rounds):
            fd = false_id[i]
            if chosen_arms[i] == 0:
                if model_ans[fd] != test_labels[fd] and sota_ans[fd] != test_labels[fd]:
                    cnt += 1
            else:
                if combine_ans[fd] == test_labels[fd]:
                    cnt += 1
        cnt_all.append(cnt)
    return np.mean(cnt_all)


if __name__ == '__main__':
    file_path = "results/ftq_results/TabLaP/ans_selector_res.txt"
    cls_ans = read_txt_to_list(file_path)
    cls_ans = cls_ans[0].split('\n')

    file_path = "results/ftq_results/num_solver_res.txt"
    predict_ans = read_list_from_file(file_path)

    label_path = "results/ftq_results/test_labels.txt"
    test_labels = read_list_from_file(label_path)

    mix_path = "results/ftq_results/mix_sc_res.txt"
    mix_sc_res = read_list_from_file(mix_path)

    # format results
    ##proposed_model_ans
    for i, text in enumerate(predict_ans):
        if len(text) >1:
            predict_ans[i] = combine_elements_into_list(text)
    ##mix_sc_final
    for i, text in enumerate(mix_sc_res):
        if len(text) >1:
            mix_sc_res[i] = combine_elements_into_list(text)
    ##test_labels
    for i, text in enumerate(test_labels):
        if len(text) >1:
            test_labels[i] = combine_elements_into_list(text)

    # get TabLaP results
    combine_ans = []
    for i in range(len(test_labels)):
        if cls_ans[i] == '[A]':
            combine_ans.append(predict_ans[i])
        elif cls_ans[i] == '[B]':
            combine_ans.append(mix_sc_res[i])
    
    # TabLaP performance
    correct_num = 0
    for i in range(1245):
        if check_single_ans(combine_ans[i], test_labels[i]):
            correct_num += 1
    print("TabLaP Accuracy is: ", correct_num / len(test_labels))
    count = official_eval(test_labels, combine_ans)
    print("Exact Matching (Official Evaluator) Accuracy is: ", count / len(test_labels))

    false_id = get_verif_res()
    # MAB UCB
    correct_n = 0
    for i in range(len(test_labels)):
        if i not in false_id and check_single_ans(test_labels[i], combine_ans[i]):
            correct_n += 1
    mab_num = mab_ucb(predict_ans, mix_sc_res, combine_ans, test_labels, false_id)
    print("Tw Acc. of TabLaP with Multi-arm Bandits: ", (correct_n + mab_num) / len(test_labels))

    # Expanding Window
    correct_t = 0
    for i in range(len(test_labels)):
        if i not in false_id and check_single_ans(test_labels[i], combine_ans[i]):
            correct_t += 1
    exp_num = expanding_window(predict_ans, mix_sc_res, combine_ans, test_labels, false_id)
    # print(correct_t)
    print("Tw Acc. of TabLaP + Expanding Window: ", (correct_t + exp_num) / len(test_labels))
