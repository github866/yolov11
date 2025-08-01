import cv2
import os
import math

def split_video_into_frames(video_path, frames_per_clip=900, output_base_dir="./"):
    """
    Split a video into multiple clips with specified number of frames each,
    saving frames as individual PNG files.
    
    Args:
        video_path (str): Path to the input video file
        frames_per_clip (int): Number of frames per clip
        output_base_dir (str): Base directory for output clips
    """
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video properties:")
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {fps}")
    print(f"  Resolution: {width}x{height}")
    
    # Calculate number of clips needed
    num_clips = math.ceil(total_frames / frames_per_clip)
    print(f"Will create {num_clips} clips with {frames_per_clip} frames each")
    
    # Create output directories
    for i in range(1, num_clips + 1):
        clip_dir = os.path.join(output_base_dir, f"clip{i}")
        os.makedirs(clip_dir, exist_ok=True)
        print(f"Created directory: {clip_dir}")
    
    # Split video into clips and save frames
    for clip_num in range(1, num_clips + 1):
        print(f"\nProcessing clip {clip_num}/{num_clips}...")
        
        # Calculate start and end frame for this clip
        start_frame = (clip_num - 1) * frames_per_clip
        end_frame = min(clip_num * frames_per_clip, total_frames)
        
        # Set starting frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frames_written = 0
        for frame_idx in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Save frame as PNG
            frame_filename = f"frame_{frame_idx - start_frame + 1:04d}.png"
            frame_path = os.path.join(output_base_dir, f"clip{clip_num}", frame_filename)
            cv2.imwrite(frame_path, frame)
            frames_written += 1
            
            # Progress indicator
            if frame_idx % 100 == 0:
                print(f"  Frame {frame_idx - start_frame + 1}/{end_frame - start_frame}")
        
        print(f"  Completed clip {clip_num}: {frames_written} PNG frames saved to clip{clip_num}/")
    
    cap.release()
    print(f"\nVideo splitting completed! Created {num_clips} clips with PNG frames.")

if __name__ == "__main__":
    video_path = "Camera-Loc03.mp4"
    split_video_into_frames(video_path, frames_per_clip=900) 