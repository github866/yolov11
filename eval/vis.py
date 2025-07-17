from matplotlib import pyplot as plt
from matplotlib.patches import Patch

import pandas as pd
import numpy as np

def draw_bar(pred_csv, gt_csv):
    pred = pd.read_csv(pred_csv)
    gt = pd.read_csv(gt_csv)

    for col in pred.columns:
        if col not in gt.columns:
            print(f"Column {col} not found in ground truth CSV.")
            continue

        pred_room = pred[col].values
        gt_room = gt[col].values
        comparison_result = []
        # if pred_room is missing, set it to 'missing'
        for p, g in zip(pred_room, gt_room):
            if p == 'missing':
                comparison_result.append('missing')
            elif p == g:
                comparison_result.append('correct')
            else:
                comparison_result.append('incorrect')

        color_map = {'correct': 'blue', 'missing': 'green', 'incorrect': 'red'}

        bar_colors = [color_map.get(val, 'gray') for val in comparison_result]

        plt.figure(figsize=(18, 2))
        plt.barh(0, 180, color='white')  # Background (optional)

        # Draw each second as a single thin bar segment
        for idx, color in enumerate(bar_colors):
            plt.barh(0, 1, left=idx, color=color, edgecolor='none')

        plt.yticks([])
        plt.xticks(np.arange(0, 181, 30))
        plt.xlim(0, 180)

        # Add legend
        legend_elements = [
            Patch(facecolor='blue', label='Correct'),
            Patch(facecolor='red', label='Incorrect'),
            Patch(facecolor='green', label='Missing')
        ]
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.xlabel('Seconds')
        plt.title(f'Prediction Result for {col}')
        plt.tight_layout()
        plt.savefig(f'{col}_comparison.png', dpi=300)
        plt.close()
        

def main():
    pred_csv = '/home/agenuinedream/repo/yolov11/results/tracking/yolov8m_coco_ours_tuned/feature_bank_7/loc_02_combined_sec.csv'
    gt_csv = '/data/leohsu/human_dataset/humans/GT/combined_gt.csv'
    draw_bar(pred_csv, gt_csv)

if __name__ == "__main__":
    main()