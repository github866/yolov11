import torch
import cv2

def read_pt(pt_path):
    data = torch.load(pt_path)
    print(data.keys())
    print(data['nurse 1'].shape)

def combine_two_videos(video1_path, video2_path, output_path):
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    if not cap1.isOpened() or not cap2.isOpened():
        print("Error opening video files.")
        return

    fps = min(cap1.get(cv2.CAP_PROP_FPS), cap2.get(cv2.CAP_PROP_FPS))
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 and not ret2:
            break

        if ret1:
            out.write(frame1)
        if ret2:
            out.write(frame2)

    cap1.release()
    cap2.release()
    out.release()
    print(f"Combined video saved to {output_path}")
    
def main():
    pt_path = '/home/agenuinedream/repo/yolov11/data/subject_features.pt'
    read_pt(pt_path)

if __name__ == "__main__":
    main()
    