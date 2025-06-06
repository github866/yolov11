import json
from collections import defaultdict

json_path = 'cropped_images/total.json'

with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Collect all frame numbers present in the file
frames_present = set(entry.get('frame') for entry in data)

# Define the custom range: 1 + 21*n for n >= 0, not exceeding 5251
full_range = set()
n = 0
while True:
    val = 1 + 21 * n
    if val > 5251:
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
    missing = sorted(full_range - frames)
    print(f'{subject}: {missing}')
