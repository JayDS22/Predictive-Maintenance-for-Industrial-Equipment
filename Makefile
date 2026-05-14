.PHONY: install data train serve test docker clean

install:
	pip install -r requirements.txt

data:
	python data/generate_synthetic_data.py --units 100

train:
	python -m src.train --units 100

train-fast:
	python -m src.train --units 60 --skip-cv

serve:
	python -m app.app

serve-prod:
	gunicorn -b 0.0.0.0:8000 -w 2 --threads 4 app.app:app

survival:
	Rscript src/survival_analysis.R

test:
	pytest -q

docker:
	docker build -t predictive-mx:latest .
	docker run --rm -p 8000:8000 predictive-mx:latest

clean:
	rm -rf models/*.joblib models/*.json models/demo_holdout.csv
	rm -f data/turbofan_synthetic.csv
	find . -name __pycache__ -type d -exec rm -rf {} +
