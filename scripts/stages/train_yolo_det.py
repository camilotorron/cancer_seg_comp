"""
    Description: 
        This script trains Yolov8 for object detection with all datasets available
"""
import json

from src.techniques.detection.yolo import Yolo

# from src.data_handling.brain_data import BrainData


def train(data: dict):
    yolo = Yolo()
    # braindata = BrainData(df_path=data.get("csv"))
    yolo.train(data=data.get("yml"))
    metrics = yolo.model.val()
    return metrics


if __name__ == "__main__":
    yolo_data = [
        {"csv": "datasets/brain/base_df_yolodet.csv", "yml": "datasets/brain/base_det.yml"},
        {"csv": "datasets/brain/augmented4_df_yolodet.csv", "yml": "datasets/brain/augmented4_det.yml"},
        {"csv": "datasets/brain/augmented10_df_yolodet.csv", "yml": "datasets/brain/augmented10_det.yml"},
    ]
    metrics_list = []
    for data in yolo_data:
        metrics_list.append(train(data=data))
    # Save to file
    with open("datasets/brain/train_detect_result.json", "w") as f:
        json.dump(data, f)
