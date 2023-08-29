from ultralytics import YOLO
from PIL import Image, ImageDraw
from src.settings.settings import env
import numpy as np


class YoloSeg:
    model = None

    def __init__(self, path: str = "models/yolov8s-seg.pt"):
        self.model = YOLO(path)

    def train(self, data: str, epochs: int = 30, imgsz: int = 256):
        results = self.model.train(
            data=data,
            epochs=epochs,
            imgsz=imgsz,
            device=0,
            patience=5,
            optimizer="Adam",
        )
        return results

    def inference(self, image: str, checkout_path: str):
        self.model = YOLO(checkout_path)
        # image_pil = Image.open(image)
        # results = self.model.predict(source=image_pil, save=False)
        results = self.model.predict(source=image, save=False)
        # imgs = ["TCGA_HT_8114_19981030_19.png", "TCGA_CS_4942_19970222_13.png"]
        # results = self.model.predict(source=imgs, save=True)
        return self._process_results(results)

    def _process_results(self, results):
        masks = []
        for result in results:
            if result.masks is not None:
                contour_points = result.masks.xy
                img = Image.new("L", (256, 256), 0)
                draw = ImageDraw.Draw(img)
                draw.polygon(contour_points[0], outline=255, fill=255)
                mask_array = np.array(img)
                masks.append(mask_array)
            else:
                img = Image.new("L", (256, 256), 0)
                mask_array = np.array(img)
                masks.append(mask_array)

        return masks
