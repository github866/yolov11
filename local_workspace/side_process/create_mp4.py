import cv2
import os
import glob
import re

def natural_sort_key(text):
    """Sort function for natural ordering of frame numbers"""
    return [int(c) if c.isdigit() else c.lower() for c in re.split('([0-9]+)', text)]

def create_video_from_frames(input_dir, output_video_path, fps=30):
    """
    Convert PNG frames to video with specified FPS
    
    Args:
        input_dir (str): Directory containing PNG frames
        output_video_path (str): Path for output video file
        fps (int): Frames per second (default: 30)
    """
    
    # Get all PNG files and sort them naturally
    frame_pattern = os.path.join(input_dir, "frame_*.png")
    frame_files = glob.glob(frame_pattern)
    frame_files.sort(key=natural_sort_key)
    
    if not frame_files:
        print(f"No PNG frames found in {input_dir}")
        return
    
    print(f"Found {len(frame_files)} frames")
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(frame_files[0])
    if first_frame is None:
        print(f"Could not read first frame: {frame_files[0]}")
        return
    
    height, width, layers = first_frame.shape
    print(f"Frame dimensions: {width}x{height}")
    
    # Define video codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID' for .avi
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        print("Error: Could not open video writer")
        return
    
    # Process each frame
    for i, frame_file in enumerate(frame_files):
        frame = cv2.imread(frame_file)
        if frame is not None:
            video_writer.write(frame)
            if i % 100 == 0:  # Progress indicator every 100 frames
                print(f"Processed {i+1}/{len(frame_files)} frames")
        else:
            print(f"Warning: Could not read frame {frame_file}")
    
    # Release video writer
    video_writer.release()
    print(f"Video created successfully: {output_video_path}")
    print(f"Total frames: {len(frame_files)}")
    print(f"Video duration: {len(frame_files)/fps:.2f} seconds")

def create_video(name):
    input_directory = f"output_bounding_boxes_{name}"
    output_video = f"{name}.mp4"
    fps = 30
    
    # Check if input directory exists
    if not os.path.exists(input_directory):
        print(f"Error: Input directory '{input_directory}' does not exist")
        return
    
    # Create video
    create_video_from_frames(input_directory, output_video, fps)

def main():
    create_video("clip1")
    create_video("clip2")
    create_video("clip3")
    create_video("clip4")
    create_video("clip5")
    create_video("clip6")

if __name__ == "__main__":
    main() 