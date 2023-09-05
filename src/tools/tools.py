import os
from PIL import Image
import numpy as np
from loguru import logger


def get_png_files(path):
    tif_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".png"):
                full_path = os.path.join(root, file)
                tif_files.append(full_path)

    return tif_files


def get_tif_files(path):
    tif_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".tif"):
                full_path = os.path.join(root, file)
                tif_files.append(full_path)

    return tif_files


def get_image_size(filepath: str):
    with Image.open(filepath) as img:
        return img.size  # width, height


def check_is_tumor(mask_path: str) -> bool:
    # if mask_path.split("/")[-1] == "None":
    #     breakpoint()
    try:
        mask = Image.open(mask_path)
        mask_np = np.array(mask)
        if np.sum(mask_np) == 0:
            return False
        return True
    except:
        logger.warning(f"Mask: {mask_path} if not readable.")
