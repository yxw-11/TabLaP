import re
import random
import numpy as np

from utils.tool_func import read_txt_to_list, read_list_from_file
from wtq.evaluate import format_string, correct_num, exact_match, normalize_string
from common_eval.evaluator import official_eval


def get_verif_res():
    path = "results/wtq_results/TabLaP/tw_evaluator_res.txt"
    bin_res = read_txt_to_list(path)
    bin_res = bin_res[0].split('\n')
    false_id = []
    for j, flag in enumerate(bin_res):
        if flag == "False":
            false_id.append(j)
    return false_id


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


def check_sample(label, model_ans):
    c = False
    if model_ans[0].lower() == label[0].lower():
        c = True
    elif label[0].lower().strip('"') == model_ans[0].lower():
        c = True
    else:
        # deal with numerical answers
        if model_ans[0].isdigit():
            try:
                norm_s1 = normalize_string(model_ans[0])
                norm_s2 = normalize_string(label[0])
                if int(norm_s1) == int(norm_s2):
                    c = True
            except:
                pass
    return c


if __name__ == '__main__':
    file_path = "results/wtq_results/TabLaP/ans_selector_res.txt"
    cls_ans = read_txt_to_list(file_path)
    cls_ans = cls_ans[0].split('\n')

    file_path = "results/wtq_results/TabLaP/refine_ans/num_solver_res.txt"
    predict_ans = read_list_from_file(file_path)
    predict_ans = format_string(predict_ans)

    label_path = "results/wtq_results/test_labels.txt"
    test_labels = read_list_from_file(label_path)
    test_labels = format_string(test_labels)

    mix_path = "results/wtq_results/TabLaP/refine_ans/mix_sc_res.txt"
    mix_sc_res = read_list_from_file(mix_path)
    mix_sc_res = format_string(mix_sc_res)

    # file_path = "results/wtq_results/TabLaP/refine_ans/num_solver_res.txt"
    # write_list_to_txt(predict_ans, file_path)

    # file_path = "results/wtq_results/TabLaP/refine_ans/mix_sc_res.txt"
    # write_list_to_txt(mix_sc_res, file_path)

    # get TabLaP results
    combine_ans = []
    for idx, pred in enumerate(cls_ans):
        if pred == '[A]':
            combine_ans.append(predict_ans[idx])
        else:
            combine_ans.append(mix_sc_res[idx])

    for i, item in enumerate(combine_ans):
        if type(item) != list:
            combine_ans[i] = [item]
    combine_ans = format_string(combine_ans)

    # for i in range(len(cls_ans)):
    #     if predict_ans[i] == test_labels[i] or mix_sc_res[i] == test_labels[i]:
    #         if combine_ans[i] != test_labels[i]:
    #             print(i, cls_ans[i], predict_ans[i], mix_sc_res[i], combine_ans[i], test_labels[i])

    num = correct_num(test_labels, combine_ans)
    exact_num = exact_match(test_labels, combine_ans)
    count = official_eval(test_labels, combine_ans)
    print("Ignore Unit Accuracy is: ", num / len(test_labels))
    print("Official Evaluate Accuracy is: ", count / len(test_labels))
    print("Exact Matching Accuracy is: ", exact_num / len(test_labels))

    # TW ACC. (ablation)
    false_id = get_verif_res()
    # print(len(false_id))
    # cnt = 0
    # for i in range(len(test_labels)):
    #     if i in false_id:
    #         if predict_ans[i] != test_labels[i] and mix_sc_res[i] != test_labels[i]:
    #             cnt += 1
    #     else:
    #         if check_sample(test_labels[i], combine_ans[i]):
    #             cnt += 1
    # print("Tw Acc. of TabLaP w/o MAB or EW: ", cnt / len(test_labels))
    
    # MAB UCB
    correct_n = 0
    for i in range(len(test_labels)):
        if i not in false_id and check_sample(test_labels[i], combine_ans[i]):
            correct_n += 1
    mab_num = mab_ucb(predict_ans, mix_sc_res, combine_ans, test_labels, false_id)
    print("Tw Acc. of TabLaP with Multi-arm Bandits: ", (correct_n + mab_num) / len(test_labels))

    # Expanding Window
    correct_t = 0
    for i in range(len(test_labels)):
        if i not in false_id and check_sample(test_labels[i], combine_ans[i]):
            correct_t += 1
    exp_num = expanding_window(predict_ans, mix_sc_res, combine_ans, test_labels, false_id)
    # print(correct_t)
    print("Tw Acc. of TabLaP + Expanding Window: ", (correct_t + exp_num) / len(test_labels))
