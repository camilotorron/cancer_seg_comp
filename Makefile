# Create all datasets for object detection and instance segmentation for Breast and Brain data
create-datasets:
	poetry run python -m scripts.stages.create_datasets

# Train yolov8 det for all datasets
train-det:
	poetry run python -m scripts.stages.train_yolo_det

# Inference of finetuned yolov8 det
infer-det:
	poetry run python -m scripts.stages.inference_testdf_det

# Evaluate finetuned yolov8 det and SAM for full pipeline segmentation
evaluate-det-sam:
	poetry run python -m scripts.stages.inference_testdf_seg_sam

# Train yolo seg for all datasets
train-seg:
	poetry run python -m scripts.stages.train_yolo_seg

# Evaluate yolo seg for all datasets
evaluate-seg:
	poetry run python -m scripts.stages.inference_testdf_yoloseg

	
#############
# run:
# 	poetry run python -m scripts.run

# yolosam:
# 	poetry run python -m scripts.test_yolo_sam

# evaluate-yolosam:
# 	poetry run python -m scripts.evaluate_yolo_sam

# pruebas:
# 	poetry run python -m scripts.pruebas
