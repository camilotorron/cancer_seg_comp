"""
    Description: 
        This script perform the inference to compute the time for 50 images for all the 
        finetuned versions of Yolov8 for instance segmentation and all the variation of YOLOv8-det +SAM with all datasets 
        available
"""
import time
import warnings

import pandas as pd
from loguru import logger
from tqdm import tqdm

from src.techniques.segmentation.yolo_seg import YoloSeg


def inferece_yoloseg(df: pd.DataFrame, models: dict):
    data_name = data.get("data")
    models = [data.get("v8s"), data.get("v8m"), data.get("v8l")]
    times = []
    for model in models:
        model_name = [key for key, value in data.items() if value == model][0]
        logger.debug(f"Inference on data {data_name} and model {model_name}")
        start_time = time.time()

        for i, row in tqdm(df.iterrows(), total=len(df)):
            image_path = f'{row["dir"]}/{row["filename"]}'
            mask_path = f'{row["dir"]}/{row["mask"]}'
            yoloseg = YoloSeg(path=model)
            masks = yoloseg.inference(image=image_path, checkout_path=model)
            end_time = time.time()

        elapsed_time = end_time - start_time
        times.append({"model_name": model_name, "data": data_name, "time": elapsed_time})

        logger.debug(f"Model {model_name} took {elapsed_time} seconds")
    return times


if __name__ == "__main__":
    data = pd.read_csv("datasets/brain/base_df.csv")
    true_examples = data[data["is_tumor"] == True].sample(25)
    false_examples = data[data["is_tumor"] == False].sample(25)
    df = pd.concat([true_examples, false_examples]).reset_index(drop=True)

    # Inference on brain data
    brain_data = [
        {
            "data": "base",
            "v8s": "runs/segment/base_train_v8s/weights/best.pt",
            "v8m": "runs/segment/base_train_v8m/weights/best.pt",
            "v8l": "runs/segment/base_train_v8l/weights/best.pt",
        },
        {
            "data": "augmented4",
            "v8s": "runs/segment/augmented4_train_v8s/weights/best.pt",
            "v8m": "runs/segment/augmented4_train_v8m/weights/best.pt",
            "v8l": "runs/segment/augmented4_train_v8l/weights/best.pt",
        },
        {
            "data": "augmented10",
            "v8s": "runs/segment/augmented10_train_v8s/weights/best.pt",
            "v8m": "runs/segment/augmented10_train_v8m/weights/best.pt",
            "v8l": "runs/segment/augmented10_train_v8l/weights/best.pt",
        },
    ]
    times = []
    # for data in brain_data:
    #     times.append(inferece_yoloseg(df=df, models=brain_data))

    # flattened_times = [item for sublist in times for item in sublist]
    # df = pd.DataFrame(flattened_times)
    logger.debug(times)

    breakpoint()
