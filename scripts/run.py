import os

HOME = os.getcwd()


import ultralytics
from ultralytics import YOLO
import rarfile
import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches


def get_bounding_box(mask_path) -> tuple:
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


def plot_img_mask(df: pd.DataFrame, image_name: str = "default"):
    if image_name != "default":
        row = df.loc[df["file_path"] == image_name]
        if row.empty:
            print("image_name not found in df")

    else:
        row = df.sample(n=1)

    image_path = f"{DRIVE_DATA_DIR}/{row['file_path'].values[0]}"
    masks = row["masks"].values[0]
    masks_paths = [f"{DRIVE_DATA_DIR}/{mask[0]}" for mask in row["masks"]]
    bboxs = row["bboxs"].values[0]

    image = Image.open(image_path)
    masks = [Image.open(mask) for mask in masks_paths]
    print(image.size)
    [print(mask.size) for mask in masks]
    fig, axs = plt.subplots(1, len(masks) + 1)

    axs[0].imshow(image)
    axs[0].set_title("Original Image")

    for i, mask in enumerate(masks, start=1):
        axs[i].imshow(mask)
        axs[i].set_title(f"Mask {i}")
    for ax in axs:
        for bbox in bboxs:
            # bbox = x_min, x_max, y_min, y_max
            xmin = bbox[0]
            xmax = bbox[1]
            ymin = bbox[2]
            ymax = bbox[3]

            rect = patches.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                linewidth=2,
                edgecolor="b",
                facecolor="none",
            )
            ax.add_patch(rect)

    for ax in axs.flatten():
        ax.axis("off")

    plt.tight_layout()
    plt.title(row["diagnostic"].values[0])
    plt.show()


def create_dataset_df_from_folder(
    folder_path, train_split=0.6, val_split=0.2, test_split=0.2
):
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".png"):
                file_path = os.path.join(root, file)
                file_paths.append(str(file_path).split("/")[-1])

    data = {"file_path": file_paths}
    df = pd.DataFrame(data)
    df.sort_values(by="file_path", ascending=True, inplace=True)
    df["diagnostic"] = df["file_path"].str.extract("(malignant|normal|benign)")
    df["mask"] = df["file_path"].str.contains("mask", case=False)

    masks = []
    files = list(df["file_path"])
    for el in files:
        if "mask" in el:
            masks.append("-")

        else:
            masks_names = [
                mask_path
                for mask_path in files
                if el.split(".")[0] in mask_path and "mask" in mask_path
            ]
            masks.append(masks_names)

    df["masks"] = masks
    df = df[~df["file_path"].str.contains("_mask")]
    splits = np.random.choice(
        ["train", "val", "test"], size=len(df), p=[train_split, val_split, test_split]
    )

    df["split"] = splits
    bboxs = []
    for _, row in df.iterrows():
        masks_paths = [f"{DRIVE_DATA_DIR}/{mask}" for mask in row["masks"]]
        b_boxes = [get_bounding_box(mask_path) for mask_path in masks_paths]
        bboxs.append(b_boxes)
    df["bboxs"] = bboxs
    return df


def main():
    print("hello")
    ultralytics.checks()


if __name__ == "__main__":
    main()
