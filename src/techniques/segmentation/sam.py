import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry

from src.settings.settings import env


class Sam:
    predictor: SamPredictor

    def __init__(self, ckpt: str = None):
        """
        Called on creation

        Args:
            ckpt (str, optional): path of SAM weights. Defaults to None.
        """
        if not ckpt:
            model = sam_model_registry["default"](checkpoint=env.SAM_CKPTS[1])
        else:
            if "vit_b" in ckpt:
                model = sam_model_registry["vit_b"](checkpoint=ckpt)
            elif "vit_l" in ckpt:
                model = sam_model_registry["vit_l"](checkpoint=ckpt)
            elif "vit_h" in ckpt:
                model = sam_model_registry["vit_h"](checkpoint=ckpt)

        self.predictor = SamPredictor(model)

    def predict(self, image: str = None, bboxs=None):
        """
        Predict for SAM

        Args:
            image (str, optional): path of the original image. Defaults to None.
            bbox (_type_, optional): prompt bounding box for SAM. Defaults to None.

        Returns:
            _type_: predicted masks
        """
        image_np = np.array(Image.open(image))
        self.predictor.set_image(image_np)
        if len(bboxs) > 1:
            # predict for each bbox
            masks_list = []
            for bbox in bboxs:
                pred_masks, _, _ = self.predictor.predict(box=np.array(bbox))
                masks_list.append(pred_masks)

            # generate votation merged masks
            merged_masks_list = []
            for mask in masks_list:
                merged_mask = self.merge_masks_votation(masks=mask)
                merged_masks_list.append(merged_mask)

            # merge all predictions in a single mask
            result_mask = self.generate_result_array(merged_masks_list)
            # self.plot_and_save_images(merged_masks_list, result_mask)
            # breakpoint()
        else:
            masks, _, _ = self.predictor.predict(box=np.array(bboxs))
            result_mask = self.merge_masks_votation(masks=masks)
        # self.plot_images_and_result(masks, result_mask)
        # breakpoint()
        return result_mask

    def merge_masks_votation(self, masks):
        sum_array = np.sum(masks, axis=0)
        result_masks = sum_array >= 2
        return result_masks

    def _create_masks_image(masks, output_name):
        # Create images for each of the 3 dimensions
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for i in range(3):
            axes[i].imshow(masks[i, :, :], cmap="gray")

        plt.savefig(output_name)

    def generate_result_array(self, array_list):
        """
        Generate a 256x256 result array based on a list of 256x256 arrays.

        Parameters:
        - array_list: List of Numpy arrays, each of shape (256, 256)

        Returns:
        - result_array: Numpy array of shape (256, 256)
        """
        # Stack all the arrays along a new dimension to create a (n, 256, 256) array
        stacked_arrays = np.stack(array_list)

        # Use np.all with axis=0 to check if a pixel is False in all arrays
        all_false = np.all(stacked_arrays == False, axis=0)

        # Create the result array: False if the pixel is False in all arrays, otherwise True
        result_array = np.logical_not(all_false)

        return result_array

    def plot_images_and_result(self, input_array, result_array):
        """
        Plot images for each dimension in the first row and the result_array in the second row.

        Parameters:
        - input_array: Numpy array of shape (n, height, width)
        - result_array: Numpy array of shape (height, width)
        """
        n = input_array.shape[0]  # Number of dimensions in the input array

        # Create a figure and axis objects for the plot
        fig, axes = plt.subplots(2, n, figsize=(15, 10))

        # Plot images for each dimension in the first row
        for i in range(n):
            axes[0, i].imshow(input_array[i, :, :], cmap="gray")
            axes[0, i].set_title(f"Dimension {i+1}")

        # Plot the result_array in the second row, centered
        axes[1, n // 2].imshow(result_array, cmap="gray")
        axes[1, n // 2].set_title("Result Array")

        # Hide unused subplots
        for i in range(n):
            if i != n // 2:
                axes[1, i].axis("off")

        plt.savefig("masks.png")

    def plot_and_save_images(self, array_list, result_array, filename="image.png"):
        """
        Plot images for each np.array in the list in the first row and the result_array in the second row.
        Save the plot as "image.png".

        Parameters:
        - array_list: List of Numpy arrays, each of shape (256, 256)
        - result_array: Numpy array of shape (256, 256)
        """
        n = len(array_list)  # Number of arrays in the list

        # Create a figure and axis objects for the plot
        fig, axes = plt.subplots(2, max(n, 1), figsize=(15, 10))

        # Plot images for each np.array in the first row
        for i in range(n):
            axes[0, i].imshow(array_list[i], cmap="gray")
            axes[0, i].set_title(f"Array {i+1}")

        # Plot the result_array in the second row, centered
        axes[1, n // 2].imshow(result_array, cmap="gray")
        axes[1, n // 2].set_title("Result Array")

        # Hide unused subplots
        for i in range(n):
            if i != n // 2:
                axes[1, i].axis("off")

        # Save the plot as "image.png"
        plt.savefig(filename)
