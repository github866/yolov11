import cv2
import os
import glob
import argparse

def frames_to_video(frames_folder, output_video_path, fps=30):
    # Get a sorted list of all PNG files in the folder
    frame_paths = sorted(glob.glob(os.path.join(frames_folder, '*.png')))
    
    if not frame_paths:
        raise Exception(f"No PNG files found in folder: {frames_folder}")
    
    # Read the first frame to get frame dimensions
    first_frame = cv2.imread(frame_paths[0])
    if first_frame is None:
        raise Exception(f"Unable to read the first frame: {frame_paths[0]}")
    
    height, width, _ = first_frame.shape
    frame_size = (width, height)
    
    # Define the codec and initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)
    
    print(f"Writing video to {output_video_path} at {fps} fps using {len(frame_paths)} frames.")
    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Warning: Unable to read frame {frame_path}. Skipping.")
            continue
        video_writer.write(frame)
    
    video_writer.release()
    print("Video creation complete.")
    return frame_paths

def cleanup_frames(frame_paths):
    # Remove each frame file from the folder
    for frame_path in frame_paths:
        try:
            os.remove(frame_path)
            print(f"Deleted {frame_path}")
        except Exception as e:
            print(f"Error deleting {frame_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Assemble PNG frames into a video and clean up the frames afterwards."
    )
    parser.add_argument(
        '--video', required=True,
        help='Path to the output video file (e.g. reconstructed_video.mp4).'
    )
    parser.add_argument(
        '--output_folder', required=True,
        help='Folder containing the PNG frames.'
    )
    parser.add_argument(
        '--fps', type=int, default=30,
        help='Frames per second for the output video (default: 30).'
    )
    args = parser.parse_args()

    frames_folder = args.output_folder
    output_video_path = args.video
    fps = args.fps

    # Assemble frames into a video
    frame_files = frames_to_video(frames_folder, output_video_path, fps)
    
    # Clean up the frames after the video is successfully created
    # cleanup_frames(frame_files)
