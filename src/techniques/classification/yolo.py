from ultralytics import YOLO


class Yolo:
    model: YOLO

    def __init__(self):
        self.model = YOLO("yolov8n.pt")

        print(type(self.model))

    def train(self):
        self.model.train(data="coco128.yaml", epochs=3)  # train the model
        metrics = self.model.val()  # evaluate model performance on the validation set
        results = self.model(
            "https://ultralytics.com/images/bus.jpg"
        )  # predict on an image
        self.path = self.model.export(format="onnx")  # export the model to ONNX format

    def inference():
        pass
