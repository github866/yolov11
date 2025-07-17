import json
import argparse
import pandas as pd
import sys
sys.path.append("/home/agenuinedream/repo/yolov11/")  # Adjust path to import from parent directory
from utils_loc.utility import id_to_name
from collections import Counter

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

def eval(
    gt_path: str, 
    pred_path: str, 
    missing_log_path: str, 
    output_file: str, 
    fps=30
): 
    with open(gt_path) as f:
        gt = json.load(f)
    with open(pred_path) as f:
        preds = json.load(f)
    with open(missing_log_path) as f:
        missing_log = json.load(f)

    for clip_id in gt.keys():
        total = 0
        correct = 0
        gt_second_labels = gt[clip_id]          # e.g. len=30 for 30s
        pred_labels = preds.get(clip_id, [])['room']   # e.g. len=900 for 30s@30fps
        missing_frames = set(missing_log.get(clip_id, []))  # set of ints

        # Expand GT to frame-wise
        gt_frame_labels = []
        for label in gt_second_labels:
            gt_frame_labels.extend([label] * fps)
        # If your last clip is shorter (e.g. 894 frames, not 900):
        start_missing_frames = []
        if preds.get(clip_id, [])["frame_id"][0] != 0:
            # Adjust the length of gt_frame_labels to match pred_labels
            for i in range(preds[clip_id]["frame_id"][0]):
                missing_frames.add(i)
                pred_labels.insert(0, None)  # Insert None for missing frames
        gt_frame_labels = gt_frame_labels[:len(pred_labels)]

        for frame_idx, gt_room in enumerate(gt_frame_labels):
            total += 1
            if frame_idx in missing_frames:
                continue  # counted as wrong, do not increment correct
            if frame_idx >= len(pred_labels):
                continue  # no prediction, wrong
            if pred_labels[frame_idx] == gt_room:
                correct += 1

        accuracy = correct / total if total > 0 else 0
        print(f"Frame-wise accuracy for {id_to_name(int(clip_id))}: {accuracy:.3f} ({correct}/{total})")
        print(f"Total frames: {total}, Correct predictions: {correct}, Missing frames: {len(missing_frames)}")

def eval_second_wise(
    gt_path: str, 
    pred_path: str, 
    missing_log_path: str, 
    output_file: str = None,
    fps=30
):
    with open(gt_path) as f:
        gt = json.load(f)
    with open(pred_path) as f:
        preds = json.load(f)
    with open(missing_log_path) as f:
        missing_log = json.load(f)

    for clip_id in gt.keys():
        gt_second_labels = gt[clip_id]                 # per-second GT, length = num seconds
        pred_labels = preds.get(clip_id, [])['room']   # per-frame pred
        missing_frames = set(missing_log.get(clip_id, []))
        frame_ids = preds[clip_id]["frame_id"]

        total_sec = 0
        correct_sec = 0
        num_seconds = len(gt_second_labels)
        for sec_idx in range(num_seconds):
            # Get frame indices for this second
            start_idx = sec_idx * fps
            end_idx = start_idx + fps
            frame_indices = list(range(start_idx, end_idx))
            
            # Use frame_ids if prediction is not strictly aligned with 0,1,2,... order
            # But assuming here that pred_labels is aligned to frame 0 = index 0
            # Exclude missing frames
            valid_preds = []
            for idx in frame_indices:
                if idx in missing_frames or idx >= len(pred_labels):
                    valid_preds.append(None)  # Mark as missing
                else:
                    valid_preds.append(pred_labels[idx])
            if not valid_preds:
                # No valid prediction for this second (all missing), count as incorrect
                total_sec += 1
                continue
        
            # Get the mode of valid predictions
            mode_pred = Counter(valid_preds).most_common(1)[0][0]

        
            gt_label = gt_second_labels[sec_idx]
            if mode_pred == gt_label:
                correct_sec += 1
            total_sec += 1

        acc = correct_sec / total_sec if total_sec > 0 else 0
        print(f"Second-wise accuracy for {id_to_name(int(clip_id))}: {acc:.3f} ({correct_sec}/{total_sec})")

        # # Optionally write to file
        # if output_file is not None:
        #     with open(output_file, "a") as outf:
        #         outf.write(f"{clip_id}: {acc:.4f} ({correct_sec}/{total_sec})\n")

def eval_csv( # second-wise
        gt_csv_path, 
        pred_csv_path
): 
    gt = pd.read_csv(gt_csv_path)
    pred = pd.read_csv(pred_csv_path)

    # calculate the number of:
    # 1. correct predictions
    # 2. incorrect predictions
    # 3. missing predictions
    correct = 0
    incorrect = 0
    missing = 0

    result = {}
    for col in gt.columns:
        # first row is the name
        print(f"Evaluating {col}")
        gt_col = gt[col].tolist()
        pred_col = pred[col].tolist()
        total = len(gt_col)
        correct = sum(1 for g, p in zip(gt_col, pred_col) if g == p)
        incorrect = sum(1 for g, p in zip(gt_col, pred_col) if g != p and p != 'missing')
        missing = sum(1 for g, p in zip(gt_col, pred_col) if p == 'missing' and g != 'missing')

        accuracy = correct / total if total > 0 else 0
        print(f"Person: {col}, Total: {total}, Correct: {correct}, Incorrect: {incorrect}, Missing: {missing}, Accuracy: {accuracy:.3f}")
    
def main():
    parser = argparse.ArgumentParser(description='Evaluate tracking results against ground truth')
    parser.add_argument("--loc_num", type=str, default="02", help="Location number for the evaluation")
    parser.add_argument("--clip_num", type=str, default="02", help="Clip number for the evaluation")
    args = parser.parse_args()
    print(f"Evaluating location {args.loc_num}, clip {args.clip_num}")

    pred_dir = '/home/agenuinedream/repo/yolov11/results/tracking/yolov8m_coco_ours_tuned/feature_bank_7'
    eval(
        f"/data/leohsu/human_dataset/humans/GT/clip_{args.loc_num}.json", 
        f"{pred_dir}/loc_{args.loc_num}_{args.clip_num}.json", 
        f"{pred_dir}/loc_{args.loc_num}_{args.clip_num}_missing_log.json", 
        f"/home/agenuinedream/repo/yolov11/results/eval/loc_{args.loc_num}_{args.clip_num}_eval.json"
    )
    eval_second_wise(
        f"/data/leohsu/human_dataset/humans/GT/clip_{args.loc_num}.json", 
        f"{pred_dir}/loc_{args.loc_num}_{args.clip_num}.json",
        f"{pred_dir}/loc_{args.loc_num}_{args.clip_num}_missing_log.json",
        f"/home/agenuinedream/repo/yolov11/results/eval/loc_{args.loc_num}_{args.clip_num}_eval.json"
    )

if __name__ == "__main__":
    # main()
    parser = argparse.ArgumentParser(description='Evaluate tracking results against ground truth')
    parser.add_argument("--loc_num", type=str, default="02", help="Location number for the evaluation")
    parser.add_argument("--pred_dir", type=str, default="/home/agenuinedream/repo/yolov11/results/tracking/yolov11x_ours_tuned/dino_vitb8", help="Prediction directory")
    args = parser.parse_args()

    pred_dir = args.pred_dir
    print(f"Evaluating location {args.loc_num}")
    eval_csv(
        gt_csv_path='/data/leohsu/human_dataset/humans/GT/combined_gt.csv',
        pred_csv_path=f'{pred_dir}/loc_02_combined_sec.csv'
    )