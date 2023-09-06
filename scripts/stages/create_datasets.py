"""
    Description: 
        This script reads data the images base folder and creates the necessary datasets and csv's for object detection and instance segmentation experiments.
"""

import pandas as pd
from loguru import logger

from src.data_handling.brain_data import BrainData
from src.data_handling.breast_data import BreastData


def create_braindata_datasets():
    # Create BrainData instance
    basedata = BrainData()
    # Read base data an export to Dataframe
    basedata.read_data()
    csv_path_base = f"{basedata.IMAGES_PATH}/base_df.csv"
    basedata.df.to_csv(csv_path_base, index=False)

    # Create BrainData instance
    augment4data = BrainData(
        augment_dict={
            "augmented_dataset_path": "datasets/brain/augmented_4",
            "augment_scale": 4,
        }
    )
    # Read base data and create augmented x 4 dataset
    augment4data.read_data()
    csv_path_4 = f"{augment4data.IMAGES_PATH}/augmented4_df.csv"
    augment4data.df.to_csv(csv_path_4, index=False)

    # Create BrainData instance
    augment10data = BrainData(
        augment_dict={
            "augmented_dataset_path": "datasets/brain/augmented_10",
            "augment_scale": 10,
        }
    )
    # Read base data and create augmented x 10 dataset
    augment10data.read_data()
    csv_path_10 = f"{augment10data.IMAGES_PATH}/augmented10_df.csv"
    augment10data.df.to_csv(csv_path_10, index=False)

    logger.debug(
        f"\nBase dataset length: {len(basedata.df)}\nAugmented 4x dataset length: {len(augment4data.df)}\nAugmented 10x dataset length: {len(augment10data.df)}"
    )

    # Create Yolo Object detection datasets
    basedata.create_yolo_det_dataset(folder_name="original_png_yolo_det")
    csv_path_base = f"{basedata.IMAGES_PATH}/base_df_yolodet.csv"
    basedata.df.to_csv(csv_path_base, index=False)

    augment4data.create_yolo_det_dataset(folder_name="augmented4_yolo_det")
    csv_path_base = f"{augment4data.IMAGES_PATH}/augmented4_df_yolodet.csv"
    augment4data.df.to_csv(csv_path_base, index=False)

    augment10data.create_yolo_det_dataset(folder_name="augmented10yolo_det")
    csv_path_base = f"{augment10data.IMAGES_PATH}/augmented10_df_yolodet.csv"
    augment10data.df.to_csv(csv_path_base, index=False)

    # Create Yolo instance segmentation datasets
    basedata.create_yolo_seg_dataset(folder_name="original_png_yolo_seg")
    csv_path_base = f"{basedata.IMAGES_PATH}/base_df_yoloseg.csv"
    basedata.df.to_csv(csv_path_base, index=False)

    augment4data.create_yolo_seg_dataset(folder_name="augmented4_yolo_det")
    csv_path_base = f"{augment4data.IMAGES_PATH}/augmented4_df_yoloseg.csv"
    augment4data.df.to_csv(csv_path_base, index=False)

    augment10data.create_yolo_seg_dataset(folder_name="augmented10yolo_seg")
    csv_path_base = f"{augment10data.IMAGES_PATH}/augmented10_df_yoloseg.csv"
    augment10data.df.to_csv(csv_path_base, index=False)


def create_breastdata_datasets():
    # Create BreastData instance
    basedata = BreastData()
    # Read base data an export to Dataframe
    basedata.read_data()
    csv_path_base = f"{basedata.IMAGES_PATH}/base_df.csv"
    basedata.df.to_csv(csv_path_base, index=False)

    # # Create BreastData instance
    # augment4data = BrainData(
    #     augment_dict={
    #         "augmented_dataset_path": "datasets/brain/augmented_4",
    #         "augment_scale": 4,
    #     }
    # )
    # # Read base data and create augmented x 4 dataset
    # augment4data.read_data()
    # csv_path_4 = f"{augment4data.IMAGES_PATH}/augmented4_df.csv"
    # augment4data.df.to_csv(csv_path_4, index=False)

    # # Create BrainData instance
    # augment10data = BrainData(
    #     augment_dict={
    #         "augmented_dataset_path": "datasets/brain/augmented_10",
    #         "augment_scale": 10,
    #     }
    # )
    # # Read base data and create augmented x 10 dataset
    # augment10data.read_data()
    # csv_path_10 = f"{augment10data.IMAGES_PATH}/augmented10_df.csv"
    # augment10data.df.to_csv(csv_path_10, index=False)

    # logger.debug(
    #     f"\nBase dataset length: {len(basedata.df)}\nAugmented 4x dataset length: {len(augment4data.df)}\nAugmented 10x dataset length: {len(augment10data.df)}"
    # )

    # # Create Yolo Object detection datasets
    # basedata.create_yolo_det_dataset(folder_name="original_png_yolo_det")
    # csv_path_base = f"{basedata.IMAGES_PATH}/base_df_yolodet.csv"
    # basedata.df.to_csv(csv_path_base, index=False)

    # augment4data.create_yolo_det_dataset(folder_name="augmented4_yolo_det")
    # csv_path_base = f"{augment4data.IMAGES_PATH}/augmented4_df_yolodet.csv"
    # augment4data.df.to_csv(csv_path_base, index=False)

    # augment10data.create_yolo_det_dataset(folder_name="augmented10yolo_det")
    # csv_path_base = f"{augment10data.IMAGES_PATH}/augmented10_df_yolodet.csv"
    # augment10data.df.to_csv(csv_path_base, index=False)

    # # Create Yolo instance segmentation datasets
    # basedata.create_yolo_seg_dataset(folder_name="original_png_yolo_seg")
    # csv_path_base = f"{basedata.IMAGES_PATH}/base_df_yoloseg.csv"
    # basedata.df.to_csv(csv_path_base, index=False)

    # augment4data.create_yolo_seg_dataset(folder_name="augmented4_yolo_det")
    # csv_path_base = f"{augment4data.IMAGES_PATH}/augmented4_df_yoloseg.csv"
    # augment4data.df.to_csv(csv_path_base, index=False)

    # augment10data.create_yolo_seg_dataset(folder_name="augmented10yolo_seg")
    # csv_path_base = f"{augment10data.IMAGES_PATH}/augmented10_df_yoloseg.csv"
    # augment10data.df.to_csv(csv_path_base, index=False)
    # pass


if __name__ == "__main__":
    create_braindata_datasets()
    create_breastdata_datasets()
