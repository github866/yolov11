import json
import os
import argparse
from collections import defaultdict, Counter
from pathlib import Path

def analyze_subject_distribution(data_dir):
    """Analyze the distribution of subjects across clips"""
    
    # Define subjects from label_id.py
    IDENTITY_SUBJECT = [
        'Unlabeled',
        'nurse_1','nurse_2',
        'patient_1','patient_2','patient_3',
        'psychiatrist','psychologist','researcher',
        'person_1','person_2','person_3','person_4'
    ]
    
    # Statistics containers
    subject_clip_count = defaultdict(set)  # subject -> set of clips
    subject_frame_count = defaultdict(int)  # subject -> total frames
    clip_subject_count = defaultdict(Counter)  # clip -> subject -> frame count
    total_frames_per_clip = defaultdict(int)  # clip -> total frames
    
    print("Analyzing subject distribution across clips...")
    print("=" * 60)
    
    # Process all JSON files (clips)
    for json_file in os.listdir(data_dir):
        if json_file.endswith('.json') and json_file.startswith('clip'):
            clip_name = json_file.replace('_missing.json', '')
            json_path = os.path.join(data_dir, json_file)
            
            print(f"\nProcessing {clip_name}...")
            
            # Read JSON file
            with open(json_path, 'r') as f:
                frames_data = json.load(f)
            
            # Analyze each frame
            for frame_info in frames_data:
                frame_id = frame_info['frame']
                persons = frame_info['persons']
                
                total_frames_per_clip[clip_name] += 1
                
                # Count subjects in this frame
                for person in persons:
                    subject_name = person['subject_name']
                    if subject_name != 'Unlabeled':
                        subject_clip_count[subject_name].add(clip_name)
                        subject_frame_count[subject_name] += 1
                        clip_subject_count[clip_name][subject_name] += 1
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUBJECT DISTRIBUTION SUMMARY")
    print("=" * 60)
    
    print(f"\nTotal clips processed: {len(total_frames_per_clip)}")
    print(f"Total subjects found: {len(subject_clip_count)}")
    
    # Subject-wise summary
    print(f"\n{'Subject':<15} {'Clips':<8} {'Total Frames':<15} {'Clip List'}")
    print("-" * 60)
    
    for subject in IDENTITY_SUBJECT[1:]:  # Skip 'Unlabeled'
        if subject in subject_clip_count:
            clips = sorted(subject_clip_count[subject])
            clip_count = len(clips)
            frame_count = subject_frame_count[subject]
            clip_list = ", ".join(clips)
            print(f"{subject:<15} {clip_count:<8} {frame_count:<15} {clip_list}")
        else:
            print(f"{subject:<15} {0:<8} {0:<15} (not found)")
    
    # Clip-wise summary
    print(f"\n{'Clip':<10} {'Total Frames':<15} {'Subjects':<10} {'Subject Details'}")
    print("-" * 60)
    
    for clip_name in sorted(total_frames_per_clip.keys()):
        total_frames = total_frames_per_clip[clip_name]
        subjects = clip_subject_count[clip_name]
        subject_count = len(subjects)
        
        # Create subject details string
        subject_details = []
        for subject, count in subjects.most_common():
            subject_details.append(f"{subject}({count})")
        subject_details_str = ", ".join(subject_details)
        
        print(f"{clip_name:<10} {total_frames:<15} {subject_count:<10} {subject_details_str}")
    
    # Save detailed statistics to JSON
    stats = {
        "subject_clip_count": {subject: list(clips) for subject, clips in subject_clip_count.items()},
        "subject_frame_count": dict(subject_frame_count),
        "clip_subject_count": {clip: dict(counter) for clip, counter in clip_subject_count.items()},
        "total_frames_per_clip": dict(total_frames_per_clip),
        "identity_subjects": IDENTITY_SUBJECT
    }
    
    output_file = "subject_distribution_stats.json"
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nDetailed statistics saved to: {output_file}")
    
    return stats

def main():
    parser = argparse.ArgumentParser(description="Count clips and frames for each subject")
    parser.add_argument("--data_dir", type=str, default="loc01_data/cropped_images", 
                       help="Directory containing JSON label files")
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Directory {args.data_dir} does not exist!")
        return
    
    stats = analyze_subject_distribution(args.data_dir)
    
    # Additional analysis
    print(f"\n" + "=" * 60)
    print("ADDITIONAL ANALYSIS")
    print("=" * 60)
    
    # Find subjects with most/least clips
    subject_clip_counts = {subject: len(clips) for subject, clips in stats["subject_clip_count"].items()}
    
    if subject_clip_counts:
        most_clips = max(subject_clip_counts.items(), key=lambda x: x[1])
        least_clips = min(subject_clip_counts.items(), key=lambda x: x[1])
        
        print(f"\nSubject with most clips: {most_clips[0]} ({most_clips[1]} clips)")
        print(f"Subject with least clips: {least_clips[0]} ({least_clips[1]} clips)")
    
    # Find subjects with most/least frames
    if stats["subject_frame_count"]:
        most_frames = max(stats["subject_frame_count"].items(), key=lambda x: x[1])
        least_frames = min(stats["subject_frame_count"].items(), key=lambda x: x[1])
        
        print(f"\nSubject with most frames: {most_frames[0]} ({most_frames[1]} frames)")
        print(f"Subject with least frames: {least_frames[0]} ({least_frames[1]} frames)")

if __name__ == "__main__":
    main() 