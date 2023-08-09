from src.settings.settings import env
import os
import pandas as pd
import numpy as np
from PIL import Image


class DataHandler:
    IMAGES_PATH: str

    def __init__(self):
        self.IMAGES_PATH = env.IMAGES_PATH

    def read_data(self, data_split: list = [0.6, 0.2, 0.2]):
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]
        path = f"{self.IMAGES_PATH}/{env.BREAST_US}"
        print(path)
        # Use os.walk to go through all directories and subdirectories
        image_files = [
            os.path.join(root, file)
            for root, dirs, files in os.walk(path)
            for file in files
            if any(file.lower().endswith(ext) for ext in image_extensions)
        ]

        df = pd.DataFrame({"file": image_files})

        df["diagnostic"] = df["file"].str.extract("(malignant|normal|benign)")
        df["is_mask"] = df["file"].str.contains("mask", case=False)
        df["filename"] = df["file"].apply(lambda x: x.split("/")[-1])

        df["mask"] = None
        for index, row in df.iterrows():
            if "_mask" not in row["filename"]:
                mask_name = row["filename"].replace(".png", "_mask")

                masks = []
                for f in df["filename"].values:
                    if mask_name in f:
                        masks.append(f)
                df.at[index, "mask"] = masks

        df = df[~df["filename"].str.contains("_mask")]

        splits = np.random.choice(
            ["train", "val", "test"],
            size=len(df),
            p=[data_split[0], data_split[1], data_split[2]],
        )

        df["split"] = splits

        bboxs = []
        for _, row in df.iterrows():
            masks_paths = [f"{path}/{mask}" for mask in row["mask"]]
            b_boxes = [self.get_bounding_box(mask_path) for mask_path in masks_paths]
            bboxs.append(b_boxes)
        df["bboxs"] = bboxs
        return df

    def get_bounding_box(self, mask_path) -> tuple:
        mask = Image.open(mask_path)
        mask_np = np.array(mask)
        segmentation = np.where(mask_np == 1)
        bbox = 0, 0, 0, 0
        if (
            len(segmentation) != 0
            and len(segmentation[1]) != 0
            and len(segmentation[0]) != 0
        ):
            x_min = int(np.min(segmentation[1]))
            x_max = int(np.max(segmentation[1]))
            y_min = int(np.min(segmentation[0]))
            y_max = int(np.max(segmentation[0]))

            bbox = x_min, x_max, y_min, y_max
        return bbox
