import os
from pydantic import BaseSettings
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()


class Environment(BaseSettings):
    IMAGES_PATH: str = os.getenv("IMAGES_PATH")
    BREAST_US: str = "breast_ultrasound_images_dataset/Dataset_BUSI_with_GT_proc"


env = Environment()
