import os

import numpy as np
from loguru import logger
from PIL import Image


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


def get_bounding_box(mask_path):
    """
    Computes the bounding box a mask

    Args:
        mask_path (_type_): path of the mask

    Returns:
        _type_: bbox in pixels
    """
    mask = Image.open(mask_path)
    mask_np = np.array(mask)
    binarized_array = (mask_np > 125).astype(int)
    segmentation = np.where(binarized_array == True)

    x_min, x_max, y_min, y_max = 0, 0, 0, 0
    if len(segmentation) != 0 and len(segmentation[1]) != 0 and len(segmentation[0]) != 0:
        x_min = int(np.min(segmentation[1]))
        x_max = int(np.max(segmentation[1]))
        y_min = int(np.min(segmentation[0]))
        y_max = int(np.max(segmentation[0]))

    bbox = x_min, x_max, y_min, y_max
    return bbox


def check_all_black_mask(mask_path: str) -> bool:
    """
    Check if a mask is all black

    Args:
        mask_path (str): path of the mask

    Returns:
        bool: True if all black
    """
    img = Image.open(mask_path).convert("L")
    img_np = np.array(img)
    return np.all(img_np == 0)


def bbox_to_yoloformat(row):
    """
    Convert bbox in pixels to yolo format

    Args:
        row (_type_): row of the features df

    Returns:
        _type_: bbox in yolo format
    """

    bbox = row["bbox"]
    imgsz = row["image_size"]
    x_min, x_max, y_min, y_max = bbox[0], bbox[1], bbox[2], bbox[3]
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min

    x_center_n = x_center / imgsz[0]
    y_center_n = y_center / imgsz[1]
    width_n = width / imgsz[0]
    height_n = height / imgsz[1]

    yolo_bbox = (x_center_n, y_center_n, width_n, height_n)

    return yolo_bbox
