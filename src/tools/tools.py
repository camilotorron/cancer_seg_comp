import ast
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from PIL import Image
from sklearn.metrics import average_precision_score


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


def generate_black_mask(h: int = 256, w: int = 256):
    """
    Generate a black mask of size hxw

    Args:
        h (int, optional): height. Defaults to 256.
        w (int, optional): width. Defaults to 256.

    Returns:
        _type_: np array of the mask
    """
    zero_array = np.zeros((1, h, w), dtype=np.int)
    return zero_array


def compute_mean_metrics(df: pd.DataFrame) -> pd.DataFrame:
    ious, dices, f1s, precs, recs = [], [], [], [], []
    for column in [
        "sam_b_iou",
        "sam_b_dice",
        "sam_b_f1",
        "sam_b_prec",
        "sam_b_rec",
        "sam_l_iou",
        "sam_l_dice",
        "sam_l_f1",
        "sam_l_prec",
        "sam_l_rec",
        "sam_h_iou",
        "sam_h_dice",
        "sam_h_f1",
        "sam_h_prec",
        "sam_h_rec",
        "medsam_iou",
        "medsam_dice",
        "medsam_f1",
        "medsam_prec",
        "medsam_rec",
    ]:
        mean_value = round(df[df[column] != 0.0][column].mean(), 3)
        if "iou" in column:
            ious.append(mean_value)
        elif "dice" in column:
            dices.append(mean_value)
        elif "f1" in column:
            f1s.append(mean_value)
        elif "prec" in column:
            precs.append(mean_value)
        elif "rec" in column:
            recs.append(mean_value)

    data = {"iou_mean": ious, "dice_mean": dices, "f1_mean": f1s, "prec_mean": precs, "rec_mean": recs}
    row_indices = ["sam_b", "sam_l", "sam_h", "med_sam"]
    mean_df = pd.DataFrame(data, index=row_indices)
    return mean_df


def compute_tnr_tpr(df: pd.DataFrame) -> float:
    no_tumor_df = df[df["is_tumor"] == False]
    no_tumor_df.loc[:, "pred_boxes"] = no_tumor_df["pred_boxes"].apply(ast.literal_eval)
    tn = len(no_tumor_df[no_tumor_df["pred_boxes"].apply(lambda x: x == [])])
    tnfp = len(no_tumor_df)
    tnr = tn / tnfp

    tumor_df = df[df["is_tumor"] == True]
    tumor_df.loc[:, "pred_boxes"] = tumor_df["pred_boxes"].apply(ast.literal_eval)
    tp = len(tumor_df[tumor_df["pred_boxes"].apply(lambda x: x != [])])
    tpfp = len(tumor_df)
    rec = tp / tpfp
    return tnr, rec


def plot_is_tumor_distribution(dfs: list[pd.DataFrame], data_names: list[str] = None):
    fig, axes = plt.subplots(1, len(dfs), figsize=(16, 8))
    plt.subplots_adjust(wspace=0.4)
    if len(dfs) == 1:
        axes = [axes]
    for ax, df, data_name in zip(axes, dfs, data_names):
        value_counts = df["is_tumor"].value_counts().rename({True: "Sí", False: "No"})
        labels = [f"{label} ({count})" for label, count in zip(value_counts.index, value_counts)]

        ax.pie(value_counts, labels=labels, autopct="%1.2f%%", startangle=90, colors=["green", "red"])
        ax.set_title(f'Distribución de "is_tumor" en {data_name}')

    plt.show()


def plot_bbox_area_distribution(dfs: list[pd.DataFrame], data_names: list[str]):
    fig, axes = plt.subplots(1, len(dfs), figsize=(18, 8))
    plt.subplots_adjust(wspace=0.4)  # Adjust the spacing between subplots
    if len(dfs) == 1:
        axes = [axes]
    for ax, df, data_name in zip(axes, dfs, data_names):
        areas = df["bbox"].apply(lambda x: eval(x))
        areas = areas.apply(lambda x: (x[2] - x[0]) * (x[3] - x[1]))
        areas = areas[areas > 0]

        ax.hist(areas, bins=10, color="blue", edgecolor="black")
        ax.set_title(f'Área de "bbox" en {data_name}')
        ax.set_xlabel("Área [píxeles cuadrados]")
        ax.set_ylabel("Frecuencia [unidades]")
    plt.show()


def plot_split_distribution(dfs: list[pd.DataFrame], data_names: list[str]):
    fig, axes = plt.subplots(1, len(dfs), figsize=(18, 8))
    plt.subplots_adjust(wspace=0.4)
    if len(dfs) == 1:
        axes = [axes]
    desired_order = ["train", "val", "test"]
    for ax, df, data_name in zip(axes, dfs, data_names):
        value_counts = df["split"].value_counts().reindex(desired_order)
        total = sum(value_counts)
        percentages = (value_counts / total) * 100
        ax.bar(percentages.index, percentages, color=["blue", "green", "red"], edgecolor="black")
        ax.set_title(f'Distribución de "split" en {data_name}')
        ax.set_xlabel("Split [train/val/test]")
        ax.set_ylabel("Porcentaje del total [%]")
    plt.show()


def calculate_det_metrics_at_thresholds(df: pd.DataFrame, thresholds: list = [0.4, 0.5, 0.7, 0.9]) -> pd.DataFrame:
    results = []

    for threshold in thresholds:
        y_true, y_scores, iou_values, iou_values, f1_values = [], [], [], [], []

        for index, row in df.iterrows():
            if pd.isna(row["pred_boxes"]) or not eval(row["pred_confs"]):
                continue

            true_bbox = eval(row["bbox"])  # Format: [xmin, xmax, ymin, ymax]
            pred_bboxes = eval(row["pred_boxes"])  # Format: [[xmin, ymin, xmax, ymax], ...]
            pred_confs = eval(row["pred_confs"])  # Format: [conf1, conf2, ...]

            best_pred_index = pred_confs.index(max(pred_confs))
            best_pred_bbox = pred_bboxes[best_pred_index]
            best_pred_bbox = [best_pred_bbox[0], best_pred_bbox[2], best_pred_bbox[1], best_pred_bbox[3]]

            true_area = (true_bbox[1] - true_bbox[0]) * (true_bbox[3] - true_bbox[2])
            pred_area = (best_pred_bbox[1] - best_pred_bbox[0]) * (best_pred_bbox[3] - best_pred_bbox[2])

            x_overlap = max(0, min(true_bbox[1], best_pred_bbox[1]) - max(true_bbox[0], best_pred_bbox[0]))
            y_overlap = max(0, min(true_bbox[3], best_pred_bbox[3]) - max(true_bbox[2], best_pred_bbox[2]))
            intersection_area = x_overlap * y_overlap

            # Calculate the union area
            union_area = true_area + pred_area - intersection_area

            # Calculate IoU
            iou = intersection_area / union_area if union_area != 0 else 0
            iou_values.append(iou)

            # Prepare data for mAP and F1-Score calculation
            is_true_positive = int(iou >= threshold)
            y_true.append(is_true_positive)
            y_scores.append(max(pred_confs))

            # Calculate F1-Score for this instance
            if is_true_positive:
                f1 = 1  # TP, so F1-Score is 1
            else:
                if pred_area > 0:
                    f1 = 0  # FP, so F1-Score is 0
                else:
                    f1 = 0  # FN, so F1-Score is 0
            f1_values.append(f1)

        mAP = average_precision_score(y_true, y_scores)

        # Calculate average IoU and F1-Score
        avg_iou = np.mean(iou_values)
        avg_f1 = np.mean(f1_values)

        results.append({"threshold": threshold, "IoU": round(avg_iou, 3), "mAP": round(mAP, 3), "F1": round(avg_f1, 3)})

    results_df = pd.DataFrame(results)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    return results_df
