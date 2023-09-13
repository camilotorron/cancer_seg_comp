"""
    Description: 
        This script evaluates the finetuned versions of Yolov8 for instance segmentation with all datasets available
"""
import warnings

import pandas as pd
from loguru import logger
from tqdm import tqdm

from src.evaluation.evaluator import Evaluator
from src.techniques.segmentation.yolo_seg import YoloSeg


def inferece_yoloseg(data: dict):
    df = pd.read_csv(data.get("csv"))
    test_df = df[df["split"] == "test"]
    data_name = data.get("csv").split("/")[-1].split("_")[0]
    models = [data.get("v8s"), data.get("v8m"), data.get("v8l")]
    for model in models:
        model_name = [key for key, value in data.items() if value == model][0]
        logger.debug(f"Inference on data {data_name} and model {model_name}")
        for i, row in tqdm(test_df.iterrows(), total=len(test_df)):
            image_path = f'{row["dir"]}/{row["filename"]}'
            mask_path = f'{row["dir"]}/{row["mask"]}'
            yoloseg = YoloSeg(path=model)
            masks = yoloseg.inference(image=image_path, checkout_path=model)
            # if len(masks) > 1:
            #     breakpoint()

            evaluator = Evaluator(true_mask=mask_path, pred_mask=masks)
            iou, dice, f1, prec, rec = (
                evaluator.iou(),
                evaluator.dice_coefficient(),
                evaluator.f1_score(),
                evaluator.precision(),
                evaluator.recall(),
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                test_df.loc[i, f"{model_name}_iou"] = iou
                test_df.loc[i, f"{model_name}_dice"] = dice
                test_df.loc[i, f"{model_name}_f1"] = f1
                test_df.loc[i, f"{model_name}_prec"] = prec
                test_df.loc[i, f"{model_name}_rec"] = rec

    data_name = data.get("csv").replace(".csv", "_inferencesegyolo.csv")
    test_df.to_csv(data_name)
    logger.debug(f"Results exported to {data_name}")


if __name__ == "__main__":
    brain_data = [
        {
            "csv": "datasets/brain/base_df_yoloseg.csv",
            "v8s": "runs/segment/base_train_v8s/weights/best.pt",
            "v8m": "runs/segment/base_train_v8m/weights/best.pt",
            "v8l": "runs/segment/base_train_v8l/weights/best.pt",
        },
        {
            "csv": "datasets/brain/augmented4_df_yoloseg.csv",
            "v8s": "runs/segment/augmented4_train_v8s/weights/best.pt",
            "v8m": "runs/segment/augmented4_train_v8m/weights/best.pt",
            "v8l": "runs/segment/augmented4_train_v8l/weights/best.pt",
        },
        {
            "csv": "datasets/brain/augmented10_df_yoloseg.csv",
            "v8s": "runs/segment/augmented10_train_v8s/weights/best.pt",
            "v8m": "runs/segment/augmented10_train_v8m/weights/best.pt",
            "v8l": "runs/segment/augmented10_train_v8l/weights/best.pt",
        },
    ]

    for data in brain_data:
        results_df = inferece_yoloseg(data=data)
