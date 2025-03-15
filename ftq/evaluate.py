import re
from utils.tool_func import read_list_from_file


def combine_elements_into_list(elements):
    combined_string = ', '.join(elements)
    return [combined_string]


def extract_numbers(input_string):
    # use regex to find all the numbers
    numbers = re.findall(r'\d+', input_string)
    return numbers if numbers else None


def correct_num(model_ans, test_label):
    cnt = 0
    for i in range(1245):
        # two strings equal
        text_p = model_ans[i][0].lower().replace(" ", "")
        text_l = test_label[i][0].lower().replace(" ", "")
        if text_p == text_l:
            cnt += 1
            continue
        elif set(text_p.split(',')) == set(text_l.split(',')):
            cnt += 1
            continue
        elif len(text_p.split(',')) == len(text_l.split(',')):
            flag = True
            ans = text_p.split(',')
            label = text_l.split(',')
            for j in range(len(ans)):
                if ans[j] == label[j]:
                    continue
                elif extract_numbers(ans[j]) and extract_numbers(ans[j]) == extract_numbers(label[j]):
                    continue
                else:
                    flag = False
            if flag:
                cnt += 1
    return cnt


if __name__ == '__main__':
    file_path = "results/ftq_results/test_labels.txt"
    test_labels = read_list_from_file(file_path)
    # math_solver_tab results
    ans_path = "results/ftq_results/num_solver_res.txt"
    refine_propose = read_list_from_file(ans_path)
    # format list
    for i, text in enumerate(test_labels):
        if len(text) > 1:
            test_labels[i] = combine_elements_into_list(text)
    for i, text in enumerate(refine_propose):
        if len(text) > 1:
            refine_propose[i] = combine_elements_into_list(text)
    num = correct_num(refine_propose, test_labels)
    print("NumSolver Accuracy is: ", num / len(test_labels))

