import json
import argparse
import os

# Set the path to the crops.json file
CROPS_JSON_PATH = os.path.join('cropped_images', 'crops.json')

def remove_subject_id(subject_id):
    # Load the JSON data
    if not os.path.exists(CROPS_JSON_PATH):
        print(f"File not found: {CROPS_JSON_PATH}")
        return
    with open(CROPS_JSON_PATH, 'r') as f:
        try:
            data = json.load(f)
        except Exception as e:
            print(f"Error loading JSON: {e}")
            return

    # Remove the subject_id from all frames
    for entry in data:
        persons = entry.get('persons', [])
        entry['persons'] = [p for p in persons if p.get('subject_id') != subject_id]

    # Optionally, remove frames with no persons left
    data = [entry for entry in data if entry.get('persons')]

    # Save the updated JSON
    with open(CROPS_JSON_PATH, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Removed subject_id {subject_id} from all frames.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remove a subject_id from all frames in crops.json')
    parser.add_argument('--subject_id', type=int, help='The subject_id to remove')
    args = parser.parse_args()
    remove_subject_id(args.subject_id)
