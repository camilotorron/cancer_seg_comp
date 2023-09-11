"""
    Description: 
        This script trains Yolov8 for instance segmentation with all datasets available
"""
import json

from loguru import logger

from src.techniques.segmentation.yolo_seg import YoloSeg


def train(data: dict = None):
    data_name = data.get("yml").split("/")[-1].split("_")[0]
    yoloseg_ckpt = [
        "models/yolo-seg/yolov8s-seg.pt",
        "models/yolo-seg/yolov8m-seg.pt",
        "models/yolo-seg/yolov8l-seg.pt",
    ]
    for model in yoloseg_ckpt:
        model_name = model.split("/")[-1]

        logger.debug(f"Training {model_name} with |{data_name}| dataset")

        experiment_name = get_experiment_name(model=model)
        yoloseg = YoloSeg(path=model)
        _ = yoloseg.train(data=data.get("yml"), model=model, name=experiment_name, epochs=20)


def get_experiment_name(model: str) -> str:
    suffix = ""
    if "v8s" in model:
        suffix = "v8s"
    elif "v8m" in model:
        suffix = "v8m"
    elif "v8l" in model:
        suffix = "v8l"
    return f"train_{suffix}"


if __name__ == "__main__":
    yolo_data = [
        {"csv": "datasets/brain/base_df_yoloseg.csv", "yml": "datasets/brain/base_seg.yml"},
        {"csv": "datasets/brain/augmented4_df_yoloseg.csv", "yml": "datasets/brain/augmented4_seg.yml"},
        {"csv": "datasets/brain/augmented10_df_yoloseg.csv", "yml": "datasets/brain/augmented10_seg.yml"},
    ]
    metrics_list = []
    for data in yolo_data:
        train(data=data)
