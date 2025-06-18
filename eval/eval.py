import json
import argparse
import sys
sys.path.append("..")  # Adjust path to import from parent directory
from utils_loc.utility import id_to_name

def eval(gt_path, pred_path, output_file):
    gt = json.load(open(gt_path, 'r'))
    pred = json.load(open(pred_path, 'r'))

    result = {}
    for person_id, positions in gt.items():
        pred_positions = pred[person_id]['room'] 

        total_secs = len(positions)
        correct_pred = 0
        for i, gt_pos in enumerate(positions):
            # find the corresponding prediction for this second
            pred_sec = pred_positions[i*30:i*30+30] if i*30+30 <= len(pred_positions) else pred_positions[i*30:]
            # get the mode of the predicted positions for this second
            pred_room = max(set(pred_sec), key=pred_sec.count) if pred_sec else None

            if pred_room == gt_pos:
                correct_pred += 1
        accuracy = correct_pred / total_secs if total_secs > 0 else 0
        print(f"Person ID: {id_to_name(int(person_id))}, Accuracy: {accuracy*100:.2f}%")

        result[person_id] = {
            'total_frames': total_secs,
            'correct_predictions': correct_pred,
            'accuracy': accuracy
        }
    
    # Save the result to a JSON file
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate tracking results against ground truth')
    parser.add_argument("--loc_num", type=str, default="02", help="Location number for the evaluation")
    parser.add_argument("--clip_num", type=str, default="06", help="Clip number for the evaluation")
    args = parser.parse_args()

    eval(
        f"/data/leohsu/human_dataset/humans/GT/clip_{args.loc_num}.json", 
        f"/home/agenuinedream/repo/yolov11/results/yolov11/by_detect/loc_{args.loc_num}_{args.clip_num}.json",
        f"/home/agenuinedream/repo/yolov11/results/eval/loc_{args.loc_num}_{args.clip_num}_eval.json"
    )