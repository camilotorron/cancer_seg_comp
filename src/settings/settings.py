import os
from pydantic import BaseSettings
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from the .env file
load_dotenv()


class Environment(BaseSettings):
    IMAGES_PATH: str = os.getenv("IMAGES_PATH")
    BREAST_US: str = "breast_ultrasound_images_dataset/Dataset_BUSI_with_GT_proc"
    YOLO_DATASET_OUTPUT: str = str(Path("data/yolo_ouput_dataset").resolve())
    labels_dict = {
        "benign": 0,
        "malignant": 1,
        "normal": 2,
    }
    DEF_YOLO_DET_THRESHOLD = 0.5
    FINETUNED_YOLO = "models/finetuned_yolo.pt"
    BRAIN_DATA = "brain_MRI_segmentation/kaggle_3m_png"
    # BRAIN_DATA = "brain_MRI_segmentation/kaggle_3m"
    BRAIN_DATA_DIR = "datasets/brain"
    SUB_DATASETS = [
        "original",
        "original_png",
        "yolo_det",
        "yolo_seg",
        "coco",
        "yolo_det_aug",
        "yolo_seg_aug",
    ]
    YOLOSEG_THRESHOLD = 0.5
    MEDSAM_PATH = "wanglab/medsam-vit-base"  # path for SamProcessor and SamModel


env = Environment()
