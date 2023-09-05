import random

import cv2


class DataAugmentator:
    """
    This class as as a data augmentation for an original image and a mask image
    """

    def __init__(self, original_image: str, mask_image: str, output_folder=".", num_augmentations: int = 5):
        """
        Called on creation

        Args:
            original_image (str): original image path
            mask_image (str): mask image path
            output_folder (str, optional): output folder. Defaults to ".".
            num_augmentations (int, optional): number of augmentation per image and mask. Defaults to 5.
        """
        self.image_path = original_image
        self.mask_path = mask_image
        self.image = self.read_image(image_path=original_image)
        self.mask = self.read_image(image_path=mask_image)
        self.output_folder = output_folder
        self.num_augmentations = num_augmentations

    def read_image(self, image_path: str):
        """
        Read an image from its path

        Args:
            image_path (str): image path

        Returns:
            _type_: image readed with cv2
        """
        return cv2.imread(image_path)

    def write_image(self, image, file_name):
        """
        Write the image in the specified location
        """
        cv2.imwrite(file_name, image)

    def horizontal_flip(self, image, mask):
        """
        Flip the image and mask horizontally.
        """
        return cv2.flip(image, 1), cv2.flip(mask, 1)

    def vertical_flip(self, image, mask):
        """
        Flip the image and mask vertically.
        """
        return cv2.flip(image, 0), cv2.flip(mask, 0)

    def rotate(self, image, mask, angle):
        """
        Rotate the image and mask.
        """
        M = cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), angle, 1)
        rotated_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        rotated_mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))
        return rotated_image, rotated_mask

    def color_jitter(self, image, brightness=0.2, contrast=0.2, saturation=0.2):
        """
        Randomly adjust brightness, contrast, and saturation.
        """
        # Brightness
        image = cv2.convertScaleAbs(image, alpha=(1 + random.uniform(-brightness, brightness)))

        # Contrast
        image = cv2.convertScaleAbs(image, beta=random.uniform(1 - contrast, 1 + contrast))

        # Saturation (convert to HSV space)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = cv2.convertScaleAbs(hsv[:, :, 1], alpha=(1 + random.uniform(-saturation, saturation)))
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return image

    def cutout(self, image, mask, patch_size=(50, 50)):
        """
        Remove a random rectangular region.
        """
        h, w, _ = image.shape
        x = random.randint(0, w - patch_size[0])
        y = random.randint(0, h - patch_size[1])

        image[y : y + patch_size[1], x : x + patch_size[0]] = 0
        mask[y : y + patch_size[1], x : x + patch_size[0]] = 0

        return image, mask

    def apply_augmentations(self):
        """
        Apply random augmentations and save the augmented images.
        """
        augmented_images, augmented_masks = [], []
        for i in range(self.num_augmentations):
            # Apply a random augmentation
            choice = random.choice(
                [
                    "horizontal_flip",
                    "vertical_flip",
                    "rotate",
                    "color_jitter",
                    "cutout",
                ]
            )

            if choice == "horizontal_flip":
                augmented_image, augmented_mask = self.horizontal_flip(self.image, self.mask)
            elif choice == "vertical_flip":
                augmented_image, augmented_mask = self.vertical_flip(self.image, self.mask)
            elif choice == "rotate":
                angle = random.randint(-90, 90)
                augmented_image, augmented_mask = self.rotate(self.image, self.mask, angle)

            elif choice == "color_jitter":
                augmented_image = self.color_jitter(self.image)
                augmented_mask = self.mask  # No change in mask
            elif choice == "random_patch":
                augmented_image, augmented_mask = self.random_patch(self.image, self.mask)
            elif choice == "cutout":
                augmented_image, augmented_mask = self.cutout(self.image, self.mask)

            # Save augmented images
            image_name = self.image_path.split(".")[0].split("/")[-1]
            mask_name = self.mask_path.split(".")[0].split("/")[-1]

            augmented_image_path = f"{self.output_folder}/{image_name}_aug{i}.png"
            augmented_mask_path = f"{self.output_folder}/{mask_name}_aug{i}.png"

            self.write_image(augmented_image, augmented_image_path)
            self.write_image(augmented_mask, augmented_mask_path)
            augmented_images.append(augmented_image_path)
            augmented_masks.append(augmented_mask_path)

        return augmented_images, augmented_masks
