import re
import ast

from utils.tool_func import read_list_from_file
from common_eval.evaluator import official_eval


def is_numerical_value(s):
    try:
        float(s)  # Attempt to convert the string to a float
        return True
    except ValueError:
        return False


def extract_numerical_value(s):
    # Use regular expression to extract numerical part
    numerical_part = re.search(r'[-+]?\d*\.?\d+', s)

    if numerical_part:
        return numerical_part.group()
    else:
        return None


def normalize_string(s):
    return ''.join(c for c in s if c.isdigit())


# this function is designed to calculate the correct cases
def correct_num(truth_labels, final_res):
    cnt_c = 0
    for i in range(len(truth_labels)):
        if final_res[i][0].lower() == truth_labels[i][0].lower():
            cnt_c += 1
        elif truth_labels[i][0].lower().strip('"') == final_res[i][0].lower():
            cnt_c += 1
        else:
            # deal with numerical answers
            if final_res[i][0].isdigit():
                try:
                    norm_s1 = normalize_string(final_res[i][0])
                    norm_s2 = normalize_string(truth_labels[i][0])
                    if int(norm_s1) == int(norm_s2):
                        cnt_c += 1
                except:
                    pass
    return cnt_c


# function to calculate the exact matching accuracy
def exact_match(truth_labels, final_res):
    cnt = 0
    for i in range(len(truth_labels)):
        if truth_labels[i][0].lower() == final_res[i][0].lower():
            cnt += 1
    return cnt


def format_string(data_list):
    for i in range(len(data_list)):
        if len(data_list[i]) > 1:
            data_list[i] = ', '.join(data_list[i])
    return data_list


if __name__ == '__main__':
    file_path = "results/wtq_results/num_solver_res.txt"
    predict_ans = read_list_from_file(file_path)
    predict_ans = format_string(predict_ans)
    label_path = "results/wtq_results/test_labels.txt"
    test_labels = read_list_from_file(label_path)
    test_labels = format_string(test_labels)
    num = correct_num(test_labels, predict_ans)
    exact_num = exact_match(test_labels, predict_ans)
    count = official_eval(test_labels, predict_ans)
    print("Ignore Unit Accuracy is: ", num / len(test_labels))
    print("Official Evaluate Acc is: ", count / len(predict_ans))
    print("Exact Matching Accuracy is: ", exact_num / len(test_labels))
