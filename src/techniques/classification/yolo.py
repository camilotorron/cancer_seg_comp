from ultralytics import YOLO
from PIL import Image
from src.settings.settings import env
from torchvision.ops import nms
import torch


class Yolo:
    model: YOLO

    def __init__(self):
        self.model = YOLO("models/yolov8s.pt")
        # self.model.add_callback("on_train_start", self.freeze_layer)

    def train(self, data: str, epochs: int = 30, imgsz: int = 256):
        self.model.train(
            data=data,
            epochs=epochs,
            imgsz=imgsz,
            device=0,
            patience=5,
            optimizer="Adam",
        )

    def inference(self, image: str, checkout_path=env.FINETUNED_YOLO):
        self.model = YOLO(checkout_path)
        image_pil = Image.open(image)
        results = self.model.predict(source=image_pil, save=False)
        return self.process_results(results)

    def process_results(self, results):
        boxes = results[0].boxes.xyxy
        confs = results[0].boxes.conf
        clss = results[0].boxes.cls
        filtered = nms(
            boxes=boxes, scores=confs, iou_threshold=env.DEF_YOLO_DET_THRESHOLD
        )
        nms_boxes = [boxes[int(i)].tolist() for i in filtered.tolist()]
        nms_confs = [confs[int(i)].tolist() for i in filtered.tolist()]
        nms_clss = [clss[int(i)].tolist() for i in filtered.tolist()]
        return nms_boxes, nms_confs, nms_clss

    def freeze_layer(self, trainer):
        model = trainer.model
        num_freeze = 10
        print(f"Freezing {num_freeze} layers")
        freeze = [f"model.{x}." for x in range(num_freeze)]  # layers to freeze
        for k, v in model.named_parameters():
            v.requires_grad = True  # train all layers
            if any(x in k for x in freeze):
                print(f"freezing {k}")
                v.requires_grad = False
        print(f"{num_freeze} layers are freezed.")
