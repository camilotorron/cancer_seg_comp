from src.settings.settings import env
import os
import shutil
import pandas as pd
import numpy as np
from PIL import Image


class DataHandler:
    IMAGES_PATH: str
    df: pd.DataFrame

    def __init__(self, df_path: str = None):
        if df_path is not None:
            self.df = pd.read_csv(df_path)
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

        df["image_size"] = df["file"].apply(self.get_image_size)
        df["yolo_bbox"] = df.apply(self.bbox_to_yoloformat, axis=1)

        self.df = df

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

    def get_image_size(self, filepath: str):
        with Image.open(filepath) as img:
            return img.size  # width, height

    def bbox_to_yoloformat(self, row):
        bboxs = row["bboxs"]
        imgsz = row["image_size"]
        yolo_bboxs = []
        for bbox in bboxs:
            x_min, x_max, y_min, y_max = bbox
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            width = x_max - x_min
            height = y_max - y_min

            x_center_n = x_center / imgsz[0]
            y_center_n = y_center / imgsz[1]
            width_n = width / imgsz[0]
            height_n = height / imgsz[1]

            yolo_bbox = (x_center_n, y_center_n, width_n, height_n)
            yolo_bboxs.append(yolo_bbox)
        return yolo_bboxs

    def create_yolo_det_dataset(
        self,
        df: pd.DataFrame,
        data_folder: str = None,
        BASE_DIR: str = env.YOLO_DATASET_OUTPUT,
    ) -> None:
        if not data_folder:
            data_folder = f"{self.IMAGES_PATH}/{env.BREAST_US}"
        if not os.path.exists(BASE_DIR):
            os.makedirs(BASE_DIR)

        subdirs = ["train", "val", "test"]
        train_count, val_count, test_count = 0, 0, 0
        for subdir in subdirs:
            os.makedirs(os.path.join(BASE_DIR, subdir), exist_ok=True)

        for index, row in df.iterrows():
            split_value = row["split"]
            if split_value not in subdirs:
                raise ValueError(f"Invalid split value: {split_value}")
            if split_value == "train":
                train_count += 1
            elif split_value == "val":
                val_count += 1
            elif split_value == "test":
                test_count += 1
            source_image_path = row["file"]
            destination_path = os.path.join(
                BASE_DIR, split_value, os.path.basename(row["filename"])
            )
            df.at[index, "yolo_ds_original_image_path"] = destination_path
            shutil.copy2(source_image_path, destination_path)
        print(
            f"Train images: {train_count}\nVal images: {val_count}\nTest images: {test_count}"
        )
        self.df = df

    def create_anotations_txt(self, df: pd.DataFrame):
        for index, row in df.iterrows():
            output_path = env.YOLO_DATASET_OUTPUT + "/" + row["split"]
            filename = row["filename"]
            diagnostic = env.labels_dict.get(row["diagnostic"])
            if diagnostic == 2:
                continue
            texts = []
            print("................................")
            print(output_path)
            print(filename)
            for bbox in row["yolo_bbox"]:
                box = " ".join(map(str, bbox))
                text = f"{diagnostic} {box}"
                texts.append(text)
            destination_path = self.write_to_txt(
                lines=texts, filename=filename, output_path=output_path
            )
            df.at[index, "yolo_ds_annot_txt_path"] = destination_path
        self.df = df

    def write_to_txt(self, lines: list, filename: str, output_path: str = ".") -> str:
        text = "\n".join(lines)
        base_name = filename.split(".")[0]
        filename = f"{base_name}.txt"
        full_path = os.path.join(output_path, filename)
        print(text)
        print(filename)
        print(full_path)
        with open(full_path, "w") as file:
            file.write(text)
        return full_path

    def export_df_to_csv(self):
        self.df.to_csv("data/data.csv")

    def create_yolo_seg_dataset(self):
        pass
