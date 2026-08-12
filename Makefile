.PHONY: install dev test lint smoke sft rm dpo grpo ablations report clean

install:
	pip install -e ".[train,judge]"

dev:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check src tests

smoke:
	bash scripts/smoke_test.sh

sft:
	python -m prefopt.train_sft --config configs/sft.yaml

rm:
	python -m prefopt.train_rm --config configs/rm.yaml

dpo:
	python -m prefopt.train_dpo --config configs/dpo.yaml

grpo:
	python -m prefopt.train_grpo --config configs/grpo.yaml

ablations:
	bash scripts/run_ablations.sh

report:
	python -m prefopt.report --runs runs --out RESULTS.md

clean:
	rm -rf runs/* RESULTS.md
