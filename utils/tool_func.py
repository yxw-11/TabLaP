# read json file
import json
import ast


def read_txt_to_list(file_path):
    with open(file_path, 'r') as file:
        # Read lines from the file and strip newline characters
        lines = [line.strip() for line in file]

    # Combine lines that are not blank into a single string
    combined_strings = []
    current_string = ''
    for line in lines:
        if line:
            current_string += line + '\n'  # Add a space to separate lines
        elif current_string:
            combined_strings.append(current_string.strip())
            current_string = ''

    # Add the last combined string if it's not empty
    if current_string:
        combined_strings.append(current_string.strip())

    return combined_strings


def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        return data


def write_list_to_txt(lst, file_path):
    with open(file_path, 'w') as file:
        for item in lst:
            file.write(str(item) + ',\n')


def read_org_list(file_path):
    with open(file_path, 'r') as file:
        content = file.read().strip()
        # convert text to list
        numbers_list = eval(content)
    return numbers_list


def read_list_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        data_list = ast.literal_eval(content)
    return data_list


def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data


if __name__ == '__main__':
    a = read_org_list("../data_sets/test_id.txt")
    print(len(a))
