import cv2
import json
import os
import numpy as np
import sys
from collections import defaultdict
import argparse

# Add parent directory to path for utility imports
sys.path.append('..')
try:
    from utils_loc.utility import color_to_room
except ImportError:
    print("Warning: utils_loc.utility not found, room detection will be disabled")
    color_to_room = None

RESIZE = 2/3

def load_json(json_path):
    """Load JSON data from file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def visualize_crops(input_dir, data, output_path, resize_factor=1.0, add_border=True):
    """
    Visualize crops for each person and organize them in subdirectories
    
    Args:
        input_dir: Directory containing frame images
        data: JSON data with person annotations
        output_path: Output directory for crops
        resize_factor: Factor to resize crops (1.0 = no resize)
        add_border: Whether to add border around crops for better visibility
    """
    # Statistics tracking
    crop_stats = defaultdict(int)
    total_crops = 0
    failed_crops = 0
    
    print(f"Processing crops from {len(data)} frames...")
    
    for frame_idx, frame in enumerate(data):
        frame_number = frame['frame']
        frame_path = f'{input_dir}/frame_{frame_number:04d}.png'
        frame_data = frame['persons']
        
        # Debug: Print first few frame paths to verify format
        if frame_idx < 5:
            print(f"Debug: Frame {frame_number} -> {frame_path}")
        
        # Progress indicator
        if frame_idx % 100 == 0:
            print(f"Processing frame {frame_idx + 1}/{len(data)} (Frame {frame_number})")
        
        # Read the frame image once
        img = cv2.imread(frame_path)
        if img is None:
            print(f"Warning: Could not read frame {frame_number} from {frame_path}")
            failed_crops += len(frame_data)
            continue
            
        h, w = img.shape[:2]  # should be 1080p: h=1080, w=1920
        
        for person_idx, person in enumerate(frame_data):
            subject_id = person['subject_id']
            subject_name = person['subject_name']
            coordinate = person['coordinate']
            x1, y1, x2, y2 = coordinate
            
            # Ensure coordinates are in correct order and within bounds
            x1, x2 = sorted([max(0, min(x1, w-1)), max(0, min(x2, w-1))])
            y1, y2 = sorted([max(0, min(y1, h-1)), max(0, min(y2, h-1))])
            
            if x2 > x1 and y2 > y1:
                # Extract crop
                crop_img = img[y1:y2, x1:x2]
                
                # Add border if requested
                if add_border:
                    border_size = 5
                    crop_img = cv2.copyMakeBorder(
                        crop_img, border_size, border_size, border_size, border_size,
                        cv2.BORDER_CONSTANT, value=[255, 255, 255]
                    )
                
                # Resize if requested
                if resize_factor != 1.0:
                    new_width = int(crop_img.shape[1] * resize_factor)
                    new_height = int(crop_img.shape[0] * resize_factor)
                    crop_img = cv2.resize(crop_img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
                
                # Create subject directory
                subject_dir = os.path.join(output_path, 'crops', subject_name)
                os.makedirs(subject_dir, exist_ok=True)
                
                # Save crop with descriptive filename - using frame format only
                crop_filename = f'frame_{frame_number:04d}.png'
                crop_path = os.path.join(subject_dir, crop_filename)
                
                # Debug: Print first few crop filenames to verify format
                if frame_idx < 3 and person_idx < 2:
                    print(f"Debug: Saving crop -> {crop_filename}")
                
                cv2.imwrite(crop_path, crop_img)
                
                # Update statistics
                crop_stats[subject_name] += 1
                total_crops += 1
            else:
                print(f"Warning: Invalid crop coordinates for {subject_name} in frame {frame_number}: {coordinate}")
                failed_crops += 1
    
    # Print statistics
    print(f"\n=== CROP STATISTICS ===")
    print(f"Total crops processed: {total_crops}")
    print(f"Failed crops: {failed_crops}")
    print(f"Successful crops by subject:")
    for subject_name, count in sorted(crop_stats.items()):
        print(f"  {subject_name}: {count} crops")
    
    # Save statistics to file
    stats_file = os.path.join(output_path, 'crop_statistics.json')
    stats_data = {
        'total_crops': total_crops,
        'failed_crops': failed_crops,
        'crops_by_subject': dict(crop_stats),
        'processing_info': {
            'input_directory': input_dir,
            'resize_factor': resize_factor,
            'border_added': add_border
        }
    }
    with open(stats_file, 'w') as f:
        json.dump(stats_data, f, indent=2)
    print(f"\nStatistics saved to: {stats_file}")

def create_visualization_summary(data, output_path):
    """
    Create a summary visualization showing all subjects and their crop counts
    """
    # Count crops per subject
    subject_counts = defaultdict(int)
    for frame in data:
        for person in frame['persons']:
            subject_counts[person['subject_name']] += 1
    
    # Create a simple text summary
    summary_file = os.path.join(output_path, 'crops_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("CROP VISUALIZATION SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total frames processed: {len(data)}\n")
        f.write(f"Total crops: {sum(subject_counts.values())}\n\n")
        f.write("Crops by subject:\n")
        f.write("-" * 20 + "\n")
        for subject_name, count in sorted(subject_counts.items()):
            f.write(f"{subject_name}: {count} crops\n")
    
    print(f"Summary saved to: {summary_file}")

def create_rectangle(data, output_rect_path, start_frame, end_frame, room_mask_path):
    # here we are drawing bounding boxes around the crops in the original images
    room_mask = cv2.imread(room_mask_path)


    if not os.path.exists(output_rect_path):
        os.makedirs(output_rect_path)
    # Build a mapping from frame number to frame data for quick lookup
    frame_map = {frame['frame']: frame for frame in data}
    for frame_number in range(start_frame, end_frame + 1):
        frame_path = f'frames/frame_{frame_number:04d}.png'
        output_path = f'{output_rect_path}/frame_{frame_number:04d}.png'
        if frame_number in frame_map:
            frame_data = frame_map[frame_number]['persons']
            img = cv2.imread(frame_path)
            if img is not None:
                for person in frame_data:
                    subject_id = person['subject_id']
                    subject_name = person['subject_name']
                    coordinate = person['coordinate']
                    x1, y1, x2, y2 = coordinate
                    # Draw rectangle and label in the same style as leo_run_track.py (omit room name)
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    cv2.putText(img, subject_name, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
                    # Find localization using room mask (center-bottom of bbox)
                    ref_x = int((x1 + x2) / 2)
                    ref_y = int(y2)
                    if room_mask is not None and 0 <= ref_y < room_mask.shape[0] and 0 <= ref_x < room_mask.shape[1]:
                        mask_value = room_mask[ref_y, ref_x]
                        mask_value = tuple(map(int, mask_value))
                        room_name = color_to_room(mask_value)
                        cv2.putText(img, room_name, (int(x1), int(y1)-35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, mask_value, 2)

                    else:
                        room_name = 'Unknown Room'
                        cv2.putText(img, room_name, (int(x1), int(y1)-35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

                cv2.imwrite(output_path, img)
        else:
            # If the frame is not in the JSON, create a blank image
            blank_img = 255 * np.ones((1080, 1920, 3), dtype=np.uint8)
            cv2.imwrite(output_path, blank_img)

def print_subject_count(data, subject_count={}):
    # 
    for frame in data:
        frame_data = frame['persons']
        for person in frame_data:
            subject_id = person['subject_id']
            subject_name = person['subject_name']
            if subject_name not in subject_count:
                subject_count[subject_name] = 0
            subject_count[subject_name] += 1
    for subject_name, count in subject_count.items():
        print(f'{subject_name}: {count}')
        
def execution(start_frame, end_frame, output_rect_path):
    # Collect all image filenames in order
    image_files = [
        os.path.join(output_rect_path, f"frame_{i:04d}.png")
        for i in range(start_frame, end_frame + 1)
        if os.path.exists(os.path.join(output_rect_path, f"frame_{i:04d}.png"))
    ]
    if not image_files:
        print("No images found in the specified frame range.")
        return

    # Read the first image to get frame size
    first_frame = cv2.imread(image_files[0])
    if first_frame is None:
        print("First image could not be read.")
        return
    height, width, layers = first_frame.shape

    # Define the codec and create VideoWriter object
    out = cv2.VideoWriter(
        f"output_video_{start_frame}_{end_frame}.mp4",
        cv2.VideoWriter_fourcc(*'mp4v'),
        30,  # fps
        (width, height)
    )

    for filename in image_files:
        img = cv2.imread(filename)
        if img is not None:
            out.write(img)
        else:
            print(f"Warning: Could not read {filename}, skipping.")

    out.release()
    print(f"Video saved as output_video_{start_frame}_{end_frame}.mp4")

def main():
    parser = argparse.ArgumentParser(description='Visualize person crops from video frames')
    parser.add_argument('--loc_number', type=str, default='loc01',
                       help='Location number')
    parser.add_argument('--input_dir', type=str, default='clip1', 
                       help='Input directory containing frame images')
    parser.add_argument('--output_dir', type=str, default='output_images',
                       help='Output directory for crops')
    parser.add_argument('--resize', type=float, default=1.0,
                       help='Resize factor for crops (1.0 = no resize)')
    parser.add_argument('--no_border', action='store_true',
                       help='Disable border around crops')
    
    
    args = parser.parse_args()
    
    # Set up paths
    input_dir = args.input_dir
    output_dir = args.output_dir+'/'+input_dir
    json_path = f'{args.loc_number}_data/cropped_images/{input_dir}_missing.json'
    
    # Load data
    print(f"Loading data from: {json_path}")
    data = load_json(json_path)
    
    # Print subject count
    print("\n=== SUBJECT COUNT ===")
    print_subject_count(data)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize crops
    print(f"\n=== PROCESSING CROPS ===")
    visualize_crops(
        input_dir=input_dir,
        data=data,
        output_path=output_dir,
        resize_factor=args.resize,
        add_border=not args.no_border
    )
    
    # Create summary
    create_visualization_summary(data, output_dir)
    
    print(f"\nCrops saved to: {os.path.join(output_dir, 'crops')}")
    print("Each subject has their own subdirectory with numbered crop images.")

if __name__ == '__main__':
    main()