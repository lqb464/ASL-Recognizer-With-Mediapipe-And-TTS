install:
	pip install -e .

collect:
	python -m src.data.collect

train:
	python -m src.models.train

infer:
	python -m tests.test_infer_webcam