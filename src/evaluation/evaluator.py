from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.metrics import adapted_rand_error
from sklearn.metrics import (
    adjusted_rand_score,
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
)


class Evaluator:
    true_mask = None
    pred_mask = None

    def __init__(self, true_mask, pred_mask):
        # Load images if they are file paths
        if isinstance(true_mask, str):
            true_mask = Image.open(true_mask)
        if isinstance(pred_mask, str):
            pred_mask = Image.open(pred_mask)
        # Convert to numpy arrays
        self.true_mask = np.array(true_mask).flatten()
        self.pred_mask = np.array(pred_mask).flatten()
        if np.isin([True, False], self.pred_mask).all():
            self.pred_mask = np.where(self.pred_mask, 255, 0).astype(np.uint8)

    def iou(self):
        """
        Compute Intersection over Union (IoU).
        """
        return jaccard_score(
            y_true=self.true_mask,
            y_pred=self.pred_mask,
            pos_label=255,
            average="binary",
            zero_division=0,
        )

    def dice_coefficient(self):
        """
        Compute Dice Coefficient.
        """
        true_mask_norm = self.true_mask / 255
        pred_mask_norm = self.pred_mask / 255
        intersection = np.logical_and(
            true_mask_norm,
            pred_mask_norm,
        ).sum()

        # Compute the Dice Score
        return 2 * intersection / (true_mask_norm.sum() + pred_mask_norm.sum()) if intersection != 0 else 0

    def precision(self):
        TP = np.logical_and(self.true_mask, self.pred_mask).sum()
        FP = np.logical_and(np.logical_not(self.true_mask), self.pred_mask).sum()
        return TP / (TP + FP) if (TP + FP) != 0 else 0

    def recall(self):
        TP = np.logical_and(self.true_mask, self.pred_mask).sum()
        FN = np.logical_and(self.true_mask, np.logical_not(self.pred_mask)).sum()
        return TP / (TP + FN) if (TP + FN) != 0 else 0

    def f1_score(self):
        """
        Compute F1 Score.
        """
        prec = self.precision()
        recall = self.recall()
        return (2 * (prec * recall) / (prec + recall)) if (prec + recall) != 0 else 0

    def compute_all_metrics(self):
        return (
            self.iou(),
            self.dice_coefficient(),
            self.f1_score(),
            self.precision(),
            self.recall(),
        )

    def _create_comparison_image(self, true_mask, pred_mask, filename="image.png"):
        """
        Create an image comparing true_mask and pred_mask side by side.

        Parameters:
            true_mask (np.ndarray): The true mask as a 256x256 numpy array.
            pred_mask (np.ndarray): The predicted mask as a 256x256 numpy array.
            filename (str): The name of the output image file.
        """
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Plot true_mask
        axes[0].imshow(true_mask, cmap="gray")
        axes[0].set_title("True")
        axes[0].axis("off")

        # Plot pred_mask
        axes[1].imshow(pred_mask, cmap="gray")
        axes[1].set_title("Pred")
        axes[1].axis("off")

        # Save the image
        plt.tight_layout()
        plt.savefig(filename)
