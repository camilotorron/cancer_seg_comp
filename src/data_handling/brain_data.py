from src.settings.settings import env
import os
import shutil
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List
from src.data_handling.bbox import Bbox


class BrainData:
    IMAGES_PATH: str
    df: pd.DataFrame

    def __init__(self, df_path: str = None):
        if df_path is not None:
            self.df = pd.read_csv(df_path)

        self.IMAGES_PATH = Path(env.IMAGES_PATH).joinpath(env.BRAIN_DATA)

    def read_data(self, data_split: List = [0.6, 0.2, 0.2]):
        # read data
        files = self.get_png_files(path=str(self.IMAGES_PATH))
        df = pd.DataFrame({"file": files})
        df["dir"], df["filename"] = zip(
            *df["file"].apply(lambda x: (x.rsplit("/", 1)[0], x.rsplit("/", 1)[1]))
        )
        df.drop(columns=["file"], inplace=True)
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
        df["mask"] = df["mask"].apply(lambda x: x[0] if len(x) > 0 else None)

        df.reset_index(drop=True, inplace=True)
        splits = np.random.choice(
            ["train", "val", "test"],
            size=len(df),
            p=[data_split[0], data_split[1], data_split[2]],
        )
        df["split"] = splits

        image_sizes, bboxs = [], []
        for index, row in df.iterrows():
            mask_path = f"{row['dir']}/{row['mask']}"
            file_path = f"{row['dir']}/{row['filename']}"
            # compute img size
            img_size = self.get_image_size(file_path)
            image_sizes.append(img_size)
            # compute bbox
            bbox = self.get_bounding_box(mask_path)
            bboxs.append(bbox)
        df["image_size"] = image_sizes
        df["bbox"] = bboxs

        df["yolo_bbox"] = df.apply(self.bbox_to_yoloformat, axis=1)
        df["is_tumor"] = df["bbox"] != (0, 0, 0, 0)

        return df

    def get_tif_files(self, path):
        tif_files = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".tif"):
                    full_path = os.path.join(root, file)
                    tif_files.append(full_path)

        return tif_files

    def get_png_files(self, path):
        tif_files = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".png"):
                    full_path = os.path.join(root, file)
                    tif_files.append(full_path)

        return tif_files

    def get_bounding_box(self, mask_path):
        mask = Image.open(mask_path)
        mask_np = np.array(mask)
        binarized_array = (mask_np > 125).astype(int)
        segmentation = np.where(binarized_array == True)
        # print(mask_np.shape)
        # print(np.min(mask_np), np.max(mask_np))
        # self.create_img(mask=mask_np, name="original")
        # self.create_img(mask=binarized_array, name="original_binarized")
        # print(segmentation)
        # breakpoint()

        x_min, x_max, y_min, y_max = 0, 0, 0, 0
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

    def organize_images(
        self,
        df: pd.DataFrame,
        data_folder: str = None,
        BASE_DIR: str = env.YOLO_DATASET_OUTPUT,
    ) -> None:
        if not data_folder:
            data_folder = self.IMAGES_PATH
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
            source_image_path = f'{row["dir"]}/{row["filename"]}'
            destination_path = os.path.join(
                BASE_DIR, split_value, os.path.basename(row["filename"])
            )
            df.at[index, "yolo_ds_original_image_path"] = destination_path
            shutil.copy2(source_image_path, destination_path)
        print(
            f"Train images: {train_count}\nVal images: {val_count}\nTest images: {test_count}"
        )
        self.df = df

    def create_anotations_txt(self, df: pd.DataFrame, output_dir: str):
        for index, row in df.iterrows():
            output_path = output_dir + "/" + row["split"]
            filename = row["filename"]
            if row["bbox"] != (0, 0, 0, 0):
                box = " ".join(map(str, row["yolo_bbox"]))
                text = f"{0} {box}"

                destination_path = self.write_to_txt(
                    lines=text, filename=filename, output_path=output_path
                )
                df.at[index, "yolo_ds_annot_txt_path"] = destination_path
        self.df = df

    def write_to_txt(self, lines: list, filename: str, output_path: str = ".") -> str:
        if isinstance(lines, str):
            text = lines
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

    def convert_tif_to_png(self):
        image_dir = self.IMAGES_PATH
        image_dir_2 = f"{str(image_dir)}_png"

        Path(image_dir_2).mkdir(parents=True, exist_ok=True)

        for subdir, _, files in os.walk(image_dir):
            for file in files:
                if file.endswith(".tif"):
                    # Construct full file path
                    full_path = os.path.join(subdir, file)

                    # Construct destination path
                    rel_dir = os.path.relpath(subdir, image_dir)
                    dest_dir = os.path.join(image_dir_2, rel_dir)
                    Path(dest_dir).mkdir(parents=True, exist_ok=True)

                    dest_path = os.path.join(dest_dir, file.replace(".tif", ".png"))

                    # Convert image
                    img = Image.open(full_path)
                    img.save(dest_path, "PNG")

    def export_df_to_csv(self):
        self.df.to_csv("data2/data.csv")

    def create_img(self, mask, name):
        breakpoint()
        img = Image.fromarray(np.uint8(mask), "RGB")
        filename = f"outputs/{name}.png"
        img.save(filename)
