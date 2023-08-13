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
    labels_dict = {"benign": 0, "normal": 1, "malignant": 2}


env = Environment()
