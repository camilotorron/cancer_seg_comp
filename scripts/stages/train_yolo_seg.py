"""
    Description: 
        This script trains Yolov8 for instance segmentation with all datasets available
"""
import json

from loguru import logger

from src.techniques.segmentation.yolo_seg import YoloSeg


def train(data: dict = None):
    data_name = data.get("yml").split("/")[-1].split("_")[0]
    logger.debug(f"Training with Yolov8-seg with |{data_name}| dataset")
    yoloseg = YoloSeg()
    metrics = yoloseg.train(data=data.get("yml"))
    return metrics


if __name__ == "__main__":
    yolo_data = [
        {"csv": "datasets/brain/base_df_yoloseg.csv", "yml": "datasets/brain/base_seg.yml"},
        {"csv": "datasets/brain/augmented4_df_yoloseg.csv", "yml": "datasets/brain/augmented4_seg.yml"},
        {"csv": "datasets/brain/augmented10_df_yoloseg.csv", "yml": "datasets/brain/augmented10_seg.yml"},
    ]
    metrics_list = []
    for data in yolo_data:
        metrics_list.append(train(data=data))
    breakpoint()
    # Save to file
    with open("datasets/brain/train_seg_result.json", "w") as f:
        json.dump(metrics_list, f)
