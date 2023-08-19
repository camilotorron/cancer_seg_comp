from PIL import Image
import numpy as np
from sklearn.metrics import jaccard_score as iou_score
from sklearn.metrics import f1_score as f1
from sklearn.metrics import adjusted_rand_score as rand_index_score
from typing import Union


class Evaluator:
    true_mask = None
    pred_mask = None

    def __init__(self, true_mask, pred_mask):
        """
        Initialize with two masks.

        :param mask1: Path to the first mask image or PIL Image object
        :param mask2: Path to the second mask image or PIL Image object
        """
        # Load images if they are file paths
        if isinstance(true_mask, str):
            true_mask = Image.open(true_mask)
        if isinstance(pred_mask, str):
            pred_mask = Image.open(pred_mask)

        # Convert to numpy arrays
        self.true_mask = np.array(true_mask).flatten()
        self.pred_mask = np.array(pred_mask).flatten()

    def iou(self):
        """
        Compute Intersection over Union (IoU).
        """
        return iou_score(y_true=self.true_mask, y_pred=self.pred_mask, average="micro")

    def dice_coefficient(self):
        """
        Compute Dice Coefficient.
        """
        intersection = np.sum(self.true_mask * self.pred_mask)
        return (2.0 * intersection) / (np.sum(self.true_mask) + np.sum(self.pred_mask))

    def f1_score(self):
        """
        Compute F1 Score.
        """
        return f1(y_true=self.true_mask, y_pred=self.pred_mask, average="micro")

    def rand_index(self):
        """
        Compute Rand Index (RI).
        """
        return rand_index_score(labels_true=self.true_mask, labels_pred=self.pred_mask)
