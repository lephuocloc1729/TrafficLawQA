.PHONY: help install test neuro-symbolic ui-facts ui-sign-template-review ui-neuro-symbolic-inspector

help:
	@echo "Neuro-symbolic traffic sign QA"
	@echo "  install                         Install dependencies"
	@echo "  test                            Run pipeline tests"
	@echo "                   Run symbolic regression examples"
	@echo "  neuro-symbolic                  Evaluate the neuro-symbolic pipeline"
	@echo "                      Run the single-image pipeline UI"
	@echo "  ui-facts                        Inspect extracted facts"
	@echo "  ui-sign-template-review         Review fact templates"
	@echo "  ui-neuro-symbolic-inspector     Inspect neuro-symbolic reasoning"

install:
	uv pip install -e ".[dev]"

test:
	pytest -q

neuro-symbolic:
	python -m src.eval.freeform_qa_metrics --config configs/neuro_symbolic_eval.json

ui-facts:
	PYTHONPATH=. streamlit run src/ui/facts_viewer.py

ui-sign-template-review:
	PYTHONPATH=. streamlit run src/ui/sign_template_review.py --server.port 8502

ui-neuro-symbolic-inspector:
	PYTHONPATH=. streamlit run src/ui/neuro_symbolic_inspector.py
