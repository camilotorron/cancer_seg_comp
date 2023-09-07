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
            model = sam_model_registry["default"](checkpoint=ckpt)

        self.predictor = SamPredictor(model)

    def predict(self, image: str = None, bbox=None):
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
        masks, _, _ = self.predictor.predict(box=np.array(bbox))
        return masks
