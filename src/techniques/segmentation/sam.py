from segment_anything import SamPredictor, sam_model_registry


class Sam:
    predictor: SamPredictor

    def __init__(self):
        model = sam_model_registry["default"](checkpoint="models/sam_vit_h_4b8939.pth")
        self.predictor = SamPredictor(model)

    def inference():
        pass
