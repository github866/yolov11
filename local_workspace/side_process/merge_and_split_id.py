import json
import os
import argparse



def load_json(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

def merge(input_dir, output_file):
    # Initialize merged data list
    merged_data = []
    
    # Load all json files
    for i in range(1, 7):
        json_file = f"{input_dir}/clip{i}_missing.json"
        data = load_json(json_file)
        # Increment frames by (i - 1) * 900
        for item in data:
            item['frame'] += (i - 1) * 900
        # Add to merged data
        merged_data.extend(data)
    
    with open(output_file, 'w') as f:
        json.dump(merged_data, f, indent=2)

def split_id(output_dir, input_json_file):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    merged_data = load_json(input_json_file)
    
    # Dictionary to store data for each subject
    subject_data = {}
    
    # Process each frame and extract person data
    for frame_item in merged_data:
        frame = frame_item['frame']
        persons = frame_item['persons']
        
        for person in persons:
            subject_name = person['subject_name']
            
            # Initialize subject data if not exists
            if subject_name not in subject_data:
                subject_data[subject_name] = []
            
            # Convert coordinate array to object format
            coord_array = person['coordinate']
            coordinates = {
                "x1": coord_array[0],
                "y1": coord_array[1],
                "x2": coord_array[2],
                "y2": coord_array[3]
            }
            
            # Create person data with desired format
            person_data = {
                'frame': frame,
                'frame_path': f"frames/frame_{frame:04d}.png",
                'coordinates': coordinates,
                'timestamp': 0
            }
            
            # Add to subject's data
            subject_data[subject_name].append(person_data)
    
    # Write each subject's data to a separate JSON file
    for subject_name, data in subject_data.items():
        output_file = os.path.join(output_dir, f"{subject_name}.json")
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Created {subject_name}.json with {len(data)} entries")

def main():
    """
    Visualize_crops.py is used to visualize the data separately, so no visualization is needed here.
    Assume that the first frame is 1, so end frame is 5394 NOT 5393.
    This script is used to merge the data and split the data into separate files for each subject.
    Format is also changed to fit the feature_bank format in github.
    The output is in the feature_bank directory in loc data.

    python3 side_process/merge_and_split_id.py --loc_number loc03
    """
    parser = argparse.ArgumentParser(description='Merge and split ID')
    parser.add_argument('--loc_number', type=str, default='loc01', help='Location number')
    args = parser.parse_args()

    loc_number = args.loc_number

    input_dir = f"{loc_number}_data/cropped_images"
    output_json_file = f"{loc_number}_data/cropped_images/merged_missing.json"
    output_dir = f"{loc_number}_data/feature_bank"
    merge(input_dir, output_json_file)
    split_id(output_dir, output_json_file)
    


if __name__ == "__main__":
    main()
