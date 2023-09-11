"""
    Description: 
        This script evaluates the finetuned versions of Yolov8 for instance segmentation with all datasets available
"""
import pandas as pd
from loguru import logger
from tqdm import tqdm

from src.techniques.segmentation.yolo_seg import YoloSeg


def inferece_yoloseg(data: dict):
    df = pd.read_csv(data.get("csv"))
    test_df = df[df["split"] == "test"]
    models = [data.get("v8s"), data.get("v8m"), data.get("v8l")]
    for model in models:
        for i, row in tqdm(test_df.iterrows(), total=len(test_df)):
            image_path = f'{row["dir"]}/{row["filename"]}'
            mask_path = f'{row["dir"]}/{row["mask"]}'
            yoloseg = YoloSeg(path=model)
            masks = yoloseg.inference(image=image_path, checkout_path=model)
            breakpoint()

    # data_name = data.get("csv").replace(".csv", "_inference_results.csv")
    # test_df.to_csv(data_name)
    # logger.debug(f"Results exported to {data_name}")


if __name__ == "__main__":
    brain_data = [
        {
            "csv": "datasets/brain/base_df_yoloseg.csv",
            "v8s": "runs/segment/base_train_v8s/weights/best.pt",
            "v8m": "runs/segment/base_train_v8m/weights/best.pt",
            "v8l": "runs/segment/base_train_v8l/weights/best.pt",
        },
        {"csv": "datasets/brain/augmented4_df_yoloseg.csv", "v8s": "", "v8m": "", "v8l": ""},
        {"csv": "datasets/brain/augmented10_df_yoloseg.csv", "v8s": "", "v8m": "", "v8l": ""},
    ]

    for data in brain_data:
        results_df = inferece_yoloseg(data=data)
