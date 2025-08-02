import json
import os
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def count_subject(json_data):
    return len(json_data)

def main():
    input_dir = "loc04_data/feature_bank"
    for file in os.listdir(input_dir):
        if file.endswith('.json'):
            json_data = load_json(os.path.join(input_dir, file))
            print(file.split('.')[0], count_subject(json_data))

if __name__ == '__main__':
    main()