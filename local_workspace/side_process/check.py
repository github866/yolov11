import json
from collections import defaultdict

json_path = 'cropped_images/missing.json'

with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Collect all frame numbers present in the file
frames_present = set(entry.get('frame') for entry in data)

# Define the custom range: 1 + 21*n for n >= 0, not exceeding 5251
full_range = set()
n = 0
while True:
    val = 1 +  1 * n
    if val > 900:
        break
    full_range.add(val)
    n += 1

# Find missing frames overall
missing_frames = sorted(full_range - frames_present)
print('Missing frames (overall):', missing_frames)

# Collect frames for each subject_name
subject_frames = defaultdict(set)
for entry in data:
    frame = entry.get('frame')
    for person in entry.get('persons', []):
        subject = person.get('subject_name')
        if subject:
            subject_frames[subject].add(frame)

# Find missing frames for each subject
print('\nMissing frames for each subject:')
for subject, frames in subject_frames.items():
    # If a frame is missing for both the subject and in the overall missing frames,
    # skip reporting it here to avoid redundancy.
    subject_missing = sorted(full_range - frames)
    # Remove frames that are already missing overall
    unique_missing = [f for f in subject_missing if f not in missing_frames]
    print(len(subject_frames[subject]))
    print(f'{subject}: {unique_missing}')
