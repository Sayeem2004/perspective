venv:
	python3 -m venv venv
	venv/bin/pip install --upgrade pip wheel setuptools

install:
	venv/bin/pip install -r requirements.txt

example:
	venv/bin/python3 example.py

classify:
	venv/bin/python3 -m classification.classify \
		--file=dataset/race/labeled-OTHER-100.csv \

compare:
	venv/bin/python3 -m classification.compare \
		--file_man=adversarial/community/novel_labeled.csv \
		--file_api=adversarial/community/novel_TOXICITY.csv \

aggregate:
	venv/bin/python3 -m classification.aggregate \
		--file=classification/race/95/unlabeled-WHITE-1100_TOXICITY.csv \
