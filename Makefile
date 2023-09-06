# Create all datasets for object detection and instance segmentation for Breast and Brain data
create-datasets:
	poetry run python -m scripts.stages.create_datasets

# Train yolov8 det for all datasets
train-det:
	poetry run python -m scripts.stages.train_yolo_det

run:
	poetry run python -m scripts.run

yolosam:
	poetry run python -m scripts.test_yolo_sam

evaluate-yolosam:
	poetry run python -m scripts.evaluate_yolo_sam

pruebas:
	poetry run python -m scripts.pruebas
