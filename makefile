venv:
	python3 -m venv venv
	venv/bin/pip install --upgrade pip wheel setuptools

install:
	venv/bin/pip install -r requirements.txt

example:
	venv/bin/python3 example.py

classify:
	venv/bin/python3 -m classification.classify --file=adversarial/community/gaming.csv
