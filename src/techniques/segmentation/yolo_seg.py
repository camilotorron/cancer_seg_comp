import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

from src.settings.settings import env


class YoloSeg:
    model = None

    def __init__(self, path: str = "models/yolov8s-seg.pt"):
        self.model = YOLO(path)

    def train(self, data: str, model="models/yolov8s-seg.pt", epochs: int = 20, imgsz: int = 256, name="train"):
        results = self.model.train(
            task="segment",
            model=model,
            data=data,
            epochs=epochs,
            imgsz=imgsz,
            device=0,
            patience=5,
            optimizer="Adam",
            name=name,
            pretrained=True,
        )

    def inference(self, image: str, checkout_path: str):
        self.model = YOLO(checkout_path)
        results = self.model.predict(source=image, save=False, verbose=False)
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
