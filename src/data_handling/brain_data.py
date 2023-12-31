import json
import os
import shutil
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from loguru import logger
from PIL import Image
from tqdm import tqdm

from src.data_handling.data_augmentator import DataAugmentator
from src.settings.settings import env
from src.tools.tools import (
    bbox_to_yoloformat,
    check_all_black_mask,
    check_is_tumor,
    get_bounding_box,
    get_image_size,
    get_png_files,
)


class BrainData:
    IMAGES_PATH: str
    df: pd.DataFrame
    augment_dict: dict
    augmented_dataset_path: str = ""
    augment_scale: int = 1

    def __init__(self, df_path: str = None, augment_dict: dict = None):
        """
        Called on creation

        Args:
            df_path (str, optional): Create a braindata from a dataframe. Defaults to None.
            augment_dict (dict, optional): augmentdict with parameters of augmentation. Defaults to None.
        """
        if df_path is not None:
            self.df = pd.read_csv(df_path)

        self.IMAGES_PATH = env.BRAIN_DATA_DIR
        self.augment_dict = augment_dict
        if augment_dict is not None:
            self.augmented_dataset_path = self.augment_dict.get("augmented_dataset_path")
            self.augment_scale = self.augment_dict.get("augment_scale")

    def read_data(self) -> None:
        """
        Route read data if augment or not
        """

        if self.augment_dict is not None:
            self._read_augment_data()
        else:
            self._read_base_data()

    def _read_base_data(self, data_split: List = [0.6, 0.2, 0.2]) -> None:
        """
        Read base dataset folder and create dataset features

        Args:
            data_split (List, optional): splits for [train, val, test]. Defaults to [0.6, 0.2, 0.2].
        """
        # read data
        original_path = str(Path(self.IMAGES_PATH).joinpath(env.SUB_DATASETS[1]).resolve())
        files = get_png_files(path=original_path)
        df = pd.DataFrame({"file": files})
        df["dir"], df["filename"] = zip(*df["file"].apply(lambda x: (x.rsplit("/", 1)[0], x.rsplit("/", 1)[1])))
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
        logger.debug("Computing sizes and bbox")
        for index, row in tqdm(df.iterrows(), total=len(df)):
            mask_path = f"{row['dir']}/{row['mask']}"
            file_path = f"{row['dir']}/{row['filename']}"
            # compute img size
            img_size = get_image_size(file_path)
            image_sizes.append(img_size)
            # compute bbox
            bbox = get_bounding_box(mask_path)
            bboxs.append(bbox)
        df["image_size"] = image_sizes
        df["bbox"] = bboxs

        df["yolo_bbox"] = df.apply(bbox_to_yoloformat, axis=1)

        logger.debug("BBoxes and image_sizes computed.")
        logger.debug("Recomputing is_tumor")
        df["is_tumor"] = df["bbox"] != (0, 0, 0, 0)
        self.df = df
        logger.debug("All features computed. DF is ready.")

    def _read_augment_data(self, data_split: list = [0.6, 0.2, 0.2]) -> None:
        """
        Read base dataset folder augment_data and create dataset features

        Args:
            data_split (List, optional): splits for [train, val, test]. Defaults to [0.6, 0.2, 0.2].
        """

        # create augmented dir
        if not os.path.exists(self.augmented_dataset_path):
            os.makedirs(self.augmented_dataset_path)

        original_path = str(Path(self.IMAGES_PATH).joinpath(env.SUB_DATASETS[1]).resolve())
        files = get_png_files(path=original_path)

        for file in files:
            file_name = file.split("/")[-1]
            destination_path = f"{self.augmented_dataset_path}/{file_name}"
            shutil.copy2(file, destination_path)

        files = get_png_files(path=self.augmented_dataset_path)

        df = pd.DataFrame({"file": files})
        df["dir"], df["filename"] = zip(*df["file"].apply(lambda x: (x.rsplit("/", 1)[0], x.rsplit("/", 1)[1])))
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

        # augment and balance data
        # Create a new column 'is_tumor'
        for i, row in tqdm(df.iterrows(), total=len(df)):
            mask_path = f'{row["dir"]}/{row["mask"]}'
            df.loc[i, "is_tumor"] = check_is_tumor(mask_path=mask_path)

        tumor_images = df["is_tumor"].value_counts().get(True, 0)
        not_tumor_images = df["is_tumor"].value_counts().get(False, 0)

        positive_factor = round(
            (self.augment_scale * (tumor_images + not_tumor_images)) / (2 * tumor_images)
        )  # How much augment the positive class to have balances dataset

        negative_factor = round(
            (self.augment_scale * (tumor_images + not_tumor_images)) / (2 * not_tumor_images)
        )  # How much augment the negative class to have balances dataset
        previous_len = len(df)
        logger.debug("Augmenting images...")
        for i, row in tqdm(df.iterrows(), total=len(df)):
            file_path = f"{row['dir']}/{row['filename']}"
            mask_path = f"{row['dir']}/{row['mask']}"

            augment_factor = positive_factor if row["is_tumor"] == True else negative_factor

            daug = DataAugmentator(
                original_image=file_path,
                mask_image=mask_path,
                output_folder=self.augmented_dataset_path,
                num_augmentations=augment_factor,
            )
            augmented_images, augmented_masks = daug.apply_augmentations()

            for image, mask in zip(augmented_images, augmented_masks):
                dir = "/".join(image.split("/")[:-1])
                file_name = image.split("/")[-1]
                mask_name = mask.split("/")[-1]

                new_row = {
                    "dir": dir,
                    "filename": file_name,
                    "mask": mask_name,
                    "is_tumor": False,
                }
                df = df._append(new_row, ignore_index=True)

        logger.debug(f"Previous length: {previous_len}     Augmented length: {len(df)}")

        # Assign split to images and masks
        splits = np.random.choice(
            ["train", "val", "test"],
            size=len(df),
            p=[data_split[0], data_split[1], data_split[2]],
        )
        df["split"] = splits

        image_sizes, bboxs = [], []
        logger.debug("Computing BBoxes...")

        for i, row in tqdm(df.iterrows(), total=len(df)):
            mask_path = f"{row['dir']}/{row['mask']}"
            file_path = f"{row['dir']}/{row['filename']}"

            if not os.path.exists(file_path) or not os.path.exists(file_path):
                breakpoint()

            # compute img size
            img_size = get_image_size(file_path)
            image_sizes.append(img_size)
            # compute bbox
            bbox = get_bounding_box(mask_path)
            bboxs.append(bbox)
        df["image_size"] = image_sizes
        df["bbox"] = bboxs
        df["yolo_bbox"] = df.apply(bbox_to_yoloformat, axis=1)

        logger.debug("BBoxes and image_sizes computed.")
        logger.debug("Recomputing is_tumor")
        df["is_tumor"] = df["bbox"] != (0, 0, 0, 0)
        self.df = df
        logger.debug("All features computed. DF is ready.")

    def create_yolo_det_dataset(
        self,
        df: pd.DataFrame = None,
        folder_name: str = None,
    ) -> None:
        """
        Creates a yolo detection dataset with annotations

        Args:
            df (pd.DataFrame, optional): Df to avoid read_data. Defaults to None.
            folder_name (str, optional): output_folder. Defaults to None.

        Raises:
            ValueError: _description_
        """
        if not df:
            df = self.df

        output_folder = str(Path(self.IMAGES_PATH).joinpath(folder_name))
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        subdirs = ["train", "val", "test"]
        train_count, val_count, test_count = 0, 0, 0
        for subdir in subdirs:
            os.makedirs(os.path.join(output_folder, subdir), exist_ok=True)
        logger.debug(f"Copying images to {output_folder}")
        for index, row in tqdm(df.iterrows(), total=len(df)):
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
            destination_path = os.path.join(output_folder, split_value, os.path.basename(row["filename"]))
            df.at[index, "yolo_ds_original_image_path"] = destination_path
            shutil.copy2(source_image_path, destination_path)
        logger.debug(f"Train images: {train_count}\nVal images: {val_count}\nTest images: {test_count}")
        df = self._create_det_annotations_txt(df=df, output_dir=output_folder)
        self.df = df

    def _create_det_annotations_txt(self, df: pd.DataFrame, output_dir: str):
        """
        Creates detection annotation txt files

        Args:
            df (pd.DataFrame): dataframe to annotate
            output_dir (str): output directory

        Returns:
            _type_: modified dataframe
        """
        logger.debug("Creating annotations files for object detection")
        for index, row in tqdm(df.iterrows(), total=len(df)):
            output_path = output_dir + "/" + row["split"]
            filename = row["filename"]
            if row["bbox"] != (0, 0, 0, 0):
                box = " ".join(map(str, row["yolo_bbox"]))
                text = f"{0} {box}"

                destination_path = self._write_to_txt(lines=text, filename=filename, output_path=output_path)
                df.at[index, "yolo_ds_annot_txt_path"] = destination_path
        return df

    def _write_to_txt(self, lines: list, filename: str, output_path: str = ".") -> str:
        """
        Writes txt annotations

        Args:
            lines (list): text to write
            filename (str): name of the file
            output_path (str, optional): output path . Defaults to ".".

        Returns:
            str: full path of the annotation txt
        """
        if isinstance(lines, str):
            text = lines
        base_name = filename.split(".")[0]
        filename = f"{base_name}.txt"
        full_path = os.path.join(output_path, filename)

        with open(full_path, "w") as file:
            file.write(text)
        return full_path

    def create_yolo_seg_dataset(self, df: pd.DataFrame = None, folder_name=None):
        """
        Creates a yolo segmentation dataset with annotations

        Args:
            df (pd.DataFrame, optional): dataframe to avoid read_data. Defaults to None.
            folder_name (_type_, optional): name of the output folder. Defaults to None.

        Raises:
            ValueError: _description_
        """

        if not df:
            df = self.df

        output_folder = str(Path(self.IMAGES_PATH).joinpath(folder_name))
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        subdirs = ["train", "val", "test"]
        train_count, val_count, test_count = 0, 0, 0

        for subdir in subdirs:
            os.makedirs(os.path.join(output_folder, subdir), exist_ok=True)

        for index, row in tqdm(df.iterrows(), total=len(df)):
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
            destination_path = os.path.join(output_folder, split_value, os.path.basename(row["filename"]))
            df.at[index, "yolo_seg_ds_original_image_path"] = destination_path
            shutil.copy2(source_image_path, destination_path)
        logger.debug(f"Train images: {train_count}\nVal images: {val_count}\nTest images: {test_count}")
        df = self._create_seg_annotations_txt(df=df, out_dir=output_folder)
        self.df = df

    def _create_seg_annotations_txt(self, df: pd.DataFrame = None, out_dir=None) -> pd.DataFrame:
        """
        Create segmentation annotation txt files

        Args:
            df (pd.DataFrame, optional): dataframe. Defaults to None.
            out_dir (_type_, optional): output folder. Defaults to None.

        Returns:
            pd.DataFrame: modified dataframe
        """
        logger.debug("Creating segmentation annotations files")
        if out_dir == None:
            out_dir = Path(self.IMAGES_PATH).joinpath(env.SUB_DATASETS[3])
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        self.df["seg_annot_filepath"] = None
        for index, row in tqdm(df.iterrows(), total=len(df)):
            # if row["split"] == "train":
            original_path = row["yolo_seg_ds_original_image_path"]
            mask_path = f'{row["dir"]}/{row["mask"]}'

            if not check_all_black_mask(mask_path=mask_path):
                text = f"0 {self._create_seg_text_annot(mask_path = mask_path)}"
                annot_file = self._create_seg_text_file(lines=text, row=row)
                self.df.loc[index, "seg_annot_filepath"] = annot_file
        return df

    def _create_seg_text_annot(self, mask_path: str):
        """
        Creates segmentation txt annotations

        Args:
            mask_path (str): path of the mask

        Returns:
            _type_: annnotation text
        """
        img = Image.open(mask_path).convert("L")
        img_np = np.array(img)
        # Binarize the image if needed
        is_binary = all(val in [0, 255] for val in np.unique(img_np))
        if not is_binary:
            img_np[img_np < 128] = 0
            img_np[img_np >= 128] = 255

        # Image.open(mask_path).show()
        img = Image.open(mask_path).convert("RGB")

        # Find contour points
        contour_points = []
        for i in range(1, img_np.shape[0] - 1):
            for j in range(1, img_np.shape[1] - 1):
                center = img_np[i, j]
                neighbors = [
                    img_np[i - 1, j - 1],
                    img_np[i - 1, j],
                    img_np[i - 1, j + 1],
                    img_np[i, j - 1],
                    img_np[i, j + 1],
                    img_np[i + 1, j - 1],
                    img_np[i + 1, j],
                    img_np[i + 1, j + 1],
                ]

                # Check if this point is a contour point (boundary between black and white)
                if all(val == center for val in neighbors):
                    continue
                contour_points.append((j, i))
        size = img_np.shape
        text = " ".join("{} {}".format(x / size[0], y / size[1]) for x, y in contour_points)
        return text

    def _create_seg_text_file(self, lines: str | list, row=None):
        if isinstance(lines, str):
            text = lines
        if row is not None:
            basename = row["yolo_seg_ds_original_image_path"].split(".")[0]
            filename = f"{basename}.txt"
            with open(filename, "w") as file:
                file.write(text)
            return filename

    def _convert_tif_to_png(self):
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

    def _export_df_to_csv(self, destination: str):
        path = destination
        self.df.to_csv(path)
        return path

    def _create_img(self, mask, name):
        img = Image.fromarray(np.uint8(mask), "RGB")
        filename = f"outputs/{name}.png"
        img.save(filename)

    #############################################################
    """
    def create_det_coco_datasets(self, out_dir=None):
        if out_dir is None:
            out_dir = env.SUB_DATASETS[4]
        # Add index column
        self.df.insert(0, "index", self.df.index)

        train_images, test_images, train_annots, test_annots = [], [], [], []
        for i, row in self.df.iterrows():
            if row["is_tumor"]:
                original_path = f"{row['dir']}/{row['filename']}"
                image_dict = {
                    "id": int(row["index"]),
                    "file_name": original_path,
                    "height": int(row["image_size"][0]),
                    "width": int(row["image_size"][1]),
                }
                area = (row["bbox"][1] - row["bbox"][0]) * (row["bbox"][-1] - row["bbox"][-2])
                annot_dict = (
                    {
                        "id": int(row["index"]),
                        "image_id": int(row["index"]),
                        "category_id": 0,
                        "bbox": [int(x) for x in row["bbox"]],
                        "area": area,
                        "iscrowd": 0,
                    },
                )
                if row["split"] != "test":
                    train_images.append(image_dict)
                    train_annots.append(annot_dict)
                else:
                    test_images.append(image_dict)
                    test_annots.append(annot_dict)
        train_json: dict = self._create_det_coco_annot_json(images=train_images, annots=train_annots)
        test_json: dict = self._create_det_coco_annot_json(images=test_images, annots=test_annots)
        # create outputs json's and return its paths
        path = f"{self.IMAGES_PATH}/{out_dir}"
        if not os.path.exists(path):
            os.makedirs(path)
        train_json_path = f"{path}/train_json.json"
        test_json_path = f"{path}/test_json.json"
        with open(train_json_path, "w") as json_file:
            json.dump(train_json, json_file)
        with open(test_json_path, "w") as json_file:
            json.dump(test_json, json_file)
        return train_json_path, test_json_path

    def _create_det_coco_annot_json(self, images: list, annots: list) -> dict:
        return {
            "info": {
                "year": "2023",
                "version": "1.0",
                "description": "Brain data object detection",
                "contributor": "camilo.torron",
            },
            "categories": [
                {"id": 0, "name": "tumor"},
            ],
            "images": images,
            "annotations": annots,
        }
    """
