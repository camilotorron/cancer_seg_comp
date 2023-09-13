"""
    Description: 
        This script performs the inference a diferent SAM versions using as input prompt the output of yolov8 detection for tumor detection for all datasets available
"""

import ast

import pandas as pd
from loguru import logger
from tqdm import tqdm

from src.evaluation.evaluator import Evaluator
from src.settings.settings import env
from src.techniques.segmentation.sam import Sam
from src.tools.tools import generate_black_mask


def inference_braindata(data: str = None, models: dict = None) -> None:
    test_df = pd.read_csv(data)
    test_df["pred_boxes"] = test_df["pred_boxes"].apply(ast.literal_eval)
    test_df["image_size"] = test_df["image_size"].apply(ast.literal_eval)

    for model_name, model_path in models.items():
        data_name = data.split("/")[-1]
        logger.debug(f"Predicting with |{model_name}| on data |{data_name}|")
        sam = Sam(ckpt=model_path)
        for i, row in tqdm(
            test_df.iterrows(),
            total=len(test_df),
        ):
            original_image = f'{row["dir"]}/{row["filename"]}'
            mask_path = f'{row["dir"]}/{row["mask"]}'
            bboxs = row["pred_boxes"]

            if bboxs != []:
                mask = sam.predict(image=original_image, bboxs=bboxs)
            else:
                mask = generate_black_mask(row["image_size"][0], row["image_size"][1])

            evaluator = Evaluator(true_mask=mask_path, pred_mask=mask)
            iou, dice, f1, prec, rec = (
                evaluator.iou(),
                evaluator.dice_coefficient(),
                evaluator.f1_score(),
                evaluator.precision(),
                evaluator.recall(),
            )
            test_df.loc[i, f"{model_name}_iou"] = iou
            test_df.loc[i, f"{model_name}_dice"] = dice
            test_df.loc[i, f"{model_name}_f1"] = f1
            test_df.loc[i, f"{model_name}_prec"] = prec
            test_df.loc[i, f"{model_name}_rec"] = rec

    data_name = data.replace(".csv", "segsam.csv")
    test_df.to_csv(data_name)
    logger.debug(f"Results exported to {data_name}")


def inference_breastdata():
    pass


if __name__ == "__main__":
    brain_data = [
        "datasets/brain/base_df_yolodet_inference.csv",
        "datasets/brain/augmented4_df_yolodet_inference.csv",
        "datasets/brain/augmented10_df_yolodet_inference.csv",
    ]

    for data in brain_data:
        results_df = inference_braindata(data=data, models=env.SAM_BASE_PATHS_EXPERIMENTS)

    inference_breastdata()
