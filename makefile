venv:
	python3 -m venv venv
	venv/bin/pip install --upgrade pip wheel setuptools

install:
	venv/bin/pip install -r requirements.txt

example:
	venv/bin/python3 example.py

classify:
	venv/bin/python3 -m classification.classify \
		--file=dataset/race/95/unlabeled-AA-1100.csv

compare:
	venv/bin/python3 -m classification.compare \
		--file_man=dataset/race/95/labeled-WHITE-100.csv \
		--file_api=classification/race/95/labeled-WHITE-100_TOXICITY.csv \

aggregate:
	venv/bin/python3 -m classification.aggregate \
		--file=classification/race/95/unlabeled-WHITE-1100_TOXICITY.csv \
