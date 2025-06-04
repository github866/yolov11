import pandas as pd
def read_init_frames(file_path): # the path endwith .data
    df = pd.read_csv(file_path, header=None)
    df.columns = ["frame_id", "id", "x", "y", "w", "h", "class", "flag1", "flag2", "flag3"]

    # Initialize output list
    output = []

    # Loop through IDs 1 to 7
    for target_id in range(1, 8):  # 1 to 7 inclusive
        row = df[df["id"] == target_id]
        if not row.empty:
            x, y, w, h = row.iloc[0][["x", "y", "w", "h"]].tolist()
            output.append([x, y, w, h])
        else:
            output.append([])

    return output

if __name__ == "__main__":
    file_path = '/data/leohsu/human_dataset/humans/metadata/init_frames/loc02-frame901.data'
    data = read_init_frames(file_path)
    print(data)



