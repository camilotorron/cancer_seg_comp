import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseSettings

# Load environment variables from the .env file
load_dotenv()


class Environment(BaseSettings):
    IMAGES_PATH: str = os.getenv("IMAGES_PATH")
    BREAST_US: str = "breast_ultrasound_images_dataset/Dataset_BUSI_with_GT_proc"
    YOLO_DATASET_OUTPUT: str = str(Path("datasets/yolo_ouput_dataset").resolve())
    labels_dict = {
        "benign": 0,
        "malignant": 1,
        "normal": 2,
    }
    DEF_YOLO_DET_THRESHOLD = 0.5
    FINETUNED_YOLO = "models/finetuned_yolo.pt"

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

    SAM_BASE_PATHS_EXPERIMENTS = {
        "sam_b": "models/SAM/sam_vit_b_01ec64.pth",
        "sam_l": "models/SAM/sam_vit_l_0b3195.pth",
        "sam_h": "models/SAM/sam_vit_h_4b8939.pth",
        "medsam": "models/SAM/medsam_vit_b.pth",
        # "medsam": "models/SAM/medsam_vit_b.pth",
        # "sam_hq": "models/SAM/sam_hq_vit_l.pth",
    }
    MEDSAM_PATH = "wanglab/medsam-vit-base"  # path for SamProcessor and SamModel
    SAM_CKPTS = [
        "models/SAM/sam_vit_b_01ec64.pth",
        "models/SAM/sam_vit_h_4b8939.pth",
        "models/SAM/sam_hq_vit_l.pth",
    ]
    SAM_HQ_CKPTS = [
        "models/SAM/sam_hq_vit_b.pth",
        "models/SAM/sam_hq_vit_h.pth",
        "models/SAM/sam_hq_vit_l.pth",
        "models/SAM/sam_hq_vit_tiny.pth",
    ]


env = Environment()
