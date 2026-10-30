# LexiSignVQA: Neuro-symbolic traffic sign QA

The retained pipeline detects traffic signs with YOLO26, extracts sign and OCR facts, parses the question into an intent, evaluates traffic rules and sign meanings, and generates a Vietnamese answer with supporting legal context.

## Run

1. Install dependencies: `make install`.
2. Configure the LLM in `.env` using `.env.example` and place the YOLO26 checkpoint at `models/yolo26s_vtsdb100_best.pt` (or choose another checkpoint in the UI).
3. Run the local facts viewer: `make ui-facts`.
4. Evaluate the pipeline on the locked sample set: `make neuro-symbolic`.
5. Run the tests: `make test`.

## Pipeline layout

- `src/deps/detection/yolo26s.py`: detector and class-to-sign mapping.
- `src/core/extract_signs.py`: image detection and sign crops.
- `src/core/symbolic/`: sign normalization, facts, intent parsing, catalogs, and reasoning.
- `src/core/sub_task_2.py`: fact assembly, legal context, and answer generation.
- `src/core/freeform_qa.py`: single-image pipeline orchestration.
- `src/ui/facts_viewer.py`: inspect detection, OCR, and facts.
- `src/eval/freeform_qa_metrics.py`: locked neuro-symbolic evaluation.

The reference legal data and model weights live under `data/`, `models/`, and `artifacts/`. The evaluation configuration is `configs/neuro_symbolic_eval.json`.
