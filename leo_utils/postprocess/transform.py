import pandas as pd
import argparse

def transform_frame_to_sec(frame_wise_csv_path):
    df = pd.read_csv(frame_wise_csv_path)
    # every 30 frames is one second
    # take the mode of every 30 frames and save the result
    print(df.index)

    df_sec = df.groupby(df.index // 30).agg(lambda x: x.mode()[0] if not x.mode().empty else None)
    print(f"Transformed {len(df)} frames to {len(df_sec)} seconds.")
    print(df_sec)

    df_sec.to_csv(frame_wise_csv_path.replace('.csv', '_sec.csv'), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transform frame-wise CSV to second-wise CSV')
    parser.add_argument("--root_dir", type=str, default='/home/agenuinedream/repo/yolov11/results/tracking/yolov11x_ours_tuned/dino_vitb8')
    args = parser.parse_args()

    frame_wise_csv_path = f'{args.root_dir}/loc_02_combined.csv'
    transform_frame_to_sec(frame_wise_csv_path)