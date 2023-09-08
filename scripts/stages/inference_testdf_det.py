"""
    Description: 
        This script performs the inference  a finetuned version of Yolov8 for tumor detection for all datasets available
"""

from loguru import logger
from tqdm import tqdm

from src.data_handling.brain_data import BrainData
from src.techniques.detection.yolo import Yolo


def inference_braindata(data: dict = None) -> None:
    """
    Inference on brain data for Yolov8 finetuned

    Args:
        data (dict, optional): dict with keys "df" and "yolo" corresponding to each path. Defaults to None.
    """
    bdata = BrainData(df_path=data.get("df"))
    test_df = bdata.df[bdata.df["split"] == "test"]
    yolo = Yolo()
    boxes_list, confs_list, clss_list = [], [], []
    for i, row in tqdm(
        test_df.iterrows(),
        total=len(test_df),
    ):
        original_image = f'{row["dir"]}/{row["filename"]}'
        boxes, confs, clss = yolo.inference(image=original_image, checkout_path=data.get("yolo"))
        boxes_list.append(boxes)
        confs_list.append(confs)
        clss_list.append(clss)

    test_df["pred_boxes"] = boxes_list
    test_df["pred_confs"] = confs_list
    test_df["pred_class"] = clss_list
    data_name = data.get("df").replace(".csv", "_inference.csv")
    test_df.to_csv(data_name)
    logger.debug(f"Results exported to {data_name}")


def inference_breastdata():
    pass


if __name__ == "__main__":
    brain_data = [
        {
            "df": "datasets/brain/base_df_yolodet.csv",
            "yolo": "runs/detect/train/weights/best.pt",
        },
        {
            "df": "datasets/brain/augmented4_df_yolodet.csv",
            "yolo": "runs/detect/train1/weights/best.pt",
        },
        {
            "df": "datasets/brain/augmented10_df_yolodet.csv",
            "yolo": "runs/detect/train2/weights/best.pt",
        },
    ]
    for data in brain_data:
        results_df = inference_braindata(data=data)

    inference_breastdata()
