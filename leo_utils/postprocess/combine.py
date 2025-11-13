import cv2
import os
import json
import csv
import sys
sys.path.append("/home/agenuinedream/repo/yolov11/utils_loc")
from utility import id_to_name
from argparse import ArgumentParser

def combine_csv(root_dir, output_path, loc_num):
    combined_data = {}

    for i in range(1, 7): # 6 clips
        with open(f'{root_dir}/loc_{loc_num:02d}_{i:02d}.json', 'r') as f:
            data = json.load(f)

        with open(f'{root_dir}/loc_{loc_num:02d}_{i:02d}_missing_log.json', 'r') as f:
            missing_log = json.load(f)
        print(f'Processing loc_{loc_num:02d}_{i:02d} with {len(data)} people')
        for k, v in data.items():
            if int(k) > 7:
                continue
            person_name = id_to_name(int(k))
            if person_name not in combined_data:
                combined_data[person_name] = []

            clip_data = v['room']
            if v['frame_id'][0] != 0:
                # If the first frame is not 0, we need to pad the beginning with 'missing'
                clip_data = ['missing'] * v['frame_id'][0] + clip_data
            if i <= 5:
                clip_data = clip_data + ['missing'] * (900 - len(clip_data)) 
            else:
                clip_data = clip_data + ['missing'] * (894 - len(clip_data))
            
            for missing_frame_num in missing_log.get(k, []):
                clip_data[missing_frame_num] = 'missing'

            combined_data[person_name].extend(clip_data)

    people = list(combined_data.keys())
    num_frames = len(combined_data[people[0]])
    rows = []
    for i in range(num_frames):
        row = [combined_data[person][i] for person in people]
        rows.append(row)

    # csv writer
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(people)  # Write header
        writer.writerows(rows)    # Write data rows

def combine_gt(gt_dir):
    combined_gt = {}

    for i in range(1, 8):
        person_name = id_to_name(i)
        print(f'Processing {person_name}')
        json_file = f'{person_name}.json'
        with open(f'{gt_dir}/{json_file}', 'r') as f:
            data = json.load(f)

        if person_name not in combined_gt:
            combined_gt[person_name] = []
        for k, v in data.items():
            combined_gt[person_name].extend(v)

    # Write combined GT to a new csv file
    output_path = f'{gt_dir}/combined_gt.csv'
    keys = list(combined_gt.keys())
    num_frames = len(combined_gt[keys[0]])
    rows = []
    for i in range(num_frames):
        row = [combined_gt[person][i] for person in keys]
        rows.append(row)
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(keys)
        writer.writerows(rows)

def combine_videos(root_dir, output_path):
    files = [f for f in sorted(os.listdir(root_dir)) if f.endswith('.mp4')]
    if not files:
        print("No videos found.")
        return

    # Get info from the first video for FPS and size
    first_video_path = os.path.join(root_dir, files[0])
    cap = cv2.VideoCapture(first_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for file in files:
        video_path = os.path.join(root_dir, file)
        print(f'Processing {video_path}')
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        cap.release()

    out.release()
    print(f"Combined video saved to {output_path}")
                
def main():
    parser = ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='/home/agenuinedream/repo/yolov11/results/tracking/yolov11x_ours_tuned/dino_vitb8')
    parser.add_argument('--location_id', type=str, default='02')
    args = parser.parse_args()
    root_dir = args.root_dir
    combine_csv(root_dir, f'{root_dir}/loc_{args.location_id}_combined.csv', int(args.location_id))
    # combine_gt('/data/leohsu/human_dataset/humans/GT')
    combine_videos(root_dir, f'{root_dir}/loc_{args.location_id}_combined.mp4')

if __name__ == "__main__":
    main()