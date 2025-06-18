import torch

def read_pt(pt_path):
    data = torch.load(pt_path)
    print(data.keys())
    print(data['nurse 1'].shape)
    
def main():
    pt_path = '/home/agenuinedream/repo/yolov11/data/subject_features.pt'
    read_pt(pt_path)

if __name__ == "__main__":
    main()
    