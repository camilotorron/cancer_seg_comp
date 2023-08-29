from src.techniques.classification.yolo import Yolo
from src.techniques.segmentation.sam import Sam
import numpy as np
from PIL import Image, ImageDraw
from typing import Union


class YoloSam:
    yolo: Yolo
    sam: Sam

    def __init__(self):
        self.yolo = Yolo()
        self.sam = Sam()

    def inference(
        self, image: Union[str, np.array], save: bool = False, checkout_path: str = ""
    ):
        results = []
        bboxs, confidences, pred_diags = self.yolo.inference(
            image=image, checkout_path=checkout_path
        )

        if len(bboxs) > 0:
            for i, bbox in enumerate(bboxs):
                image_np = np.array(Image.open(image))
                self.sam.predictor.set_image(image_np)
                masks, _, _ = self.sam.predictor.predict(box=np.array(bbox))
                if save:
                    self._process_image(
                        image_path=image,
                        confidence=confidences[i],
                        diagnostic=int(pred_diags[i]),
                        mask=masks,
                        bbox=bbox,
                        output_folder="outputs/exp1",
                    )
                result_dict = {
                    "bbox": bbox,
                    "confidence": confidences[i],
                    "diag": pred_diags[i],
                    "mask": masks,
                }
                results.append(result_dict)
            return results

        else:
            print(f"Image: {image} hast no bbox detected")
            return [{"bbox": None, "confidence": None, "diag": None, "mask": None}]

    def process_image(
        image_path, confidence, diagnostic, mask, bbox, output_folder="outputs"
    ):
        original_image = Image.open(image_path)
        image_with_bbox = original_image.copy()
        draw = ImageDraw.Draw(image_with_bbox)
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline="yellow", width=2)
        text = f"Confidence: {confidence:.2f}, Diagnostic: {diagnostic}"
        draw.text((x1, y1 - 20), text, fill="yellow")
        # Generate black and white mask image (use the sum of the three channels to create a single-channel mask)
        bw_mask = np.sum(mask, axis=0)
        bw_mask = (bw_mask > 0).astype(
            np.uint8
        ) * 255  # Convert to binary and scale to 255
        mask_image = Image.fromarray(bw_mask, mode="L")

        # Generate an overlay image with a green transparent mask
        green_color = (0, 255, 0, 128)  # RGBA (last value is alpha for transparency)
        colored_mask = Image.new("RGBA", original_image.size, (0, 0, 0, 0))

        # Convert to binary mask where at least one channel has a positive value
        overlay_mask = np.any(mask, axis=0)
        for y in range(overlay_mask.shape[0]):
            for x in range(overlay_mask.shape[1]):
                if overlay_mask[y, x]:
                    colored_mask.putpixel((x, y), green_color)

        # Overlay the green transparent mask on the original image
        original_image_rgba = original_image.convert("RGBA")
        overlayed_image = Image.alpha_composite(original_image_rgba, colored_mask)

        filename = image_path.split("/")[-1].split(".")[0]
        original_name = f"{output_folder}/{filename}_0.png"
        image_bbox_name = f"{output_folder}/{filename}_1.png"
        mask_image_name = f"{output_folder}/{filename}_2.png"
        overlayed_image_name = f"{output_folder}/{filename}_3.png"
        # Save the images
        original_image.save(original_name)
        image_with_bbox.save(image_bbox_name)
        mask_image.save(mask_image_name)
        overlayed_image.save(overlayed_image_name)
