# Create all datasets for object detection and instance segmentation for Breast and Brain data
create-datasets:
	poetry run python -m scripts.stages.create_datasets

run:
	poetry run python -m scripts.run

yolosam:
	poetry run python -m scripts.test_yolo_sam

evaluate-yolosam:
	poetry run python -m scripts.evaluate_yolo_sam

pruebas:
	poetry run python -m scripts.pruebas
