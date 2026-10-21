# Traffic Sign Legal QA

Traffic Sign Legal QA is a multimodal question-answering project for Vietnamese
traffic law and road signs. The system takes a road-scene image and a natural
language question, detects and classifies traffic signs with a fine-tuned YOLO26s
model, applies symbolic traffic rules, and generates a Vietnamese free-form
answer with citations and visual evidence.

The current QA pipeline returns structured JSON instead of forcing the model to
choose only `A/B/C/D` or `Đúng/Sai`.

## Features

- Detect traffic signs from uploaded road-scene images.
- Classify detected signs directly into the application's traffic-fact schema.
- Keep every detector result for symbolic reasoning; the default QA path does not
  call a VLM relevance filter or retrieve sign identities from Qdrant.
- Load full legal article context by `law_id` and `article_id`.
- Generate Vietnamese free-form answers with legal citations.
- Attach evidence from legal articles and directly detected signs.
- Distinguish trained sign variants such as `P.131a`, `P.131b`, and `P.131c`.
- Run an interactive Streamlit demo locally or on a rented GPU.
- Use local GPU LLMs through an OpenAI-compatible endpoint.
- Evaluate correctness, citation quality, evidence faithfulness, grounding,
  retrieval recall, JSON validity, hallucination, and latency.

## Architecture

The project has two main stages.

### 1. LawDB Processing

1. Preprocess the raw legal database.
2. Extract sign crops from clean legal-document images.
3. Parse sign names and descriptions using a VLM.
4. Embed sign crops with the configured vision embedding model.
5. Ingest embeddings and metadata into Qdrant.

### 2. Image QA

1. Run fine-tuned YOLO26s on the uploaded road-scene image.
2. Map detections into sign observations, preserving unresolved classes as
   ambiguous so the symbolic engine can fail closed.
3. Evaluate the question against sign facts and traffic rules.
4. Load legal article text from LawDB using the direct sign/rule references.
5. Ask the LLM to produce strict JSON with answer, citations, evidence,
   reasoning, and confidence.

## Project Structure

```text
.
├── data/
│   ├── law_db/                 # Legal corpus, article images, extracted signs
│   ├── train_data/             # Training samples and images
│   ├── public_test/            # Public test samples and images
│   └── private_test/           # Private test samples and images
├── docs/                       # Diagrams and notes
├── notebook/                   # Exploration notebooks
├── outputs/                    # Evaluation reports and generated artifacts
├── src/
│   ├── core/
│   │   ├── lawdb/              # LawDB preprocess, parse, lookup, ingest
│   │   ├── freeform_qa.py      # Single-image QA pipeline used by Streamlit
│   │   ├── sub_task_1.py       # Article retrieval pipeline
│   │   ├── sub_task_2.py       # Free-form cited QA pipeline
│   │   ├── extract_signs.py    # Query-image sign detection
│   │   ├── filter_signs.py     # VLM sign relevance filtering
│   │   └── query_signs.py      # Qdrant retrieval
│   ├── deps/                   # Detector, embedding, LLM, Qdrant wrappers
│   ├── eval/                   # Evaluation scripts
│   ├── prompts/                # Prompt templates
│   ├── ui/                     # Streamlit apps
│   └── settings.py             # Environment-based configuration
├── docker-compose.yaml         # Qdrant service
├── Makefile                    # Common commands
├── pyproject.toml              # Python dependencies
└── README.md
```

## Requirements

- Python `>=3.10`
- GPU recommended for YOLO26s and local LLM inference
- OpenAI-compatible chat completion endpoint

Recommended default components:

- Qdrant: `http://localhost:6333`
- LLM endpoint: `http://localhost:18989/v1`
- LLM model: `gemma3:12b` or another multimodal instruction model
- LawDB sign extraction: `ImageProcessing`
- Query sign detection: `Yolo26s`
- Fine-tuned checkpoint: `models/yolo26s_vtsdb100_best.pt`

## Installation

```bash
git clone <your-repository-url>
cd traffic-sign-legal-qa

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -e ".[dev]"
```

Enable traffic-sign text recognition (time ranges, distances, and numeric
limits) with PaddleOCR:

```bash
pip install -e ".[dev,ocr]"
```

PaddleOCR uses the GPU when the installed PaddlePaddle build supports the
machine's CUDA version; otherwise it selects CPU automatically. For GPU use,
install the matching CUDA build of PaddlePaddle using the
[official PaddlePaddle installation guide](https://www.paddlepaddle.org.cn/install/quick).
OCR runs once on the full source image immediately after YOLO26. Its text
polygons are associated with detected signs by overlap or close alignment
below a sign, so text on an attached supplementary panel can be linked to the
sign above. The symbolic layer extracts every visible time window, below-sign
distance, expressway route code/name, and destination-plus-distance pair. The
highway profile also collects nearby YOLO maximum/minimum speed-limit signs.
These observations support `INFORMATION` questions; they do not by themselves
decide whether a restriction is legally active. Unclear distances retain an
unresolved S.501/S.502 meaning.
Configure it with `TRAFFIC_SIGN_OCR_ENABLED`,
`TRAFFIC_SIGN_OCR_DEVICE` (`auto`, `gpu`, or `cpu`), and
`TRAFFIC_SIGN_OCR_LANGUAGE` (default `vi`).

With `uv`:

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Configuration

Create `.env` in the project root. You can start from `.env.example`.

Recommended local GPU configuration:

```env
LLM_BASE_URL=http://localhost:18989/v1
LLM_MODEL=gemma3:12b
LLM_API_KEYS=local

IMAGE_LAWDB_DETECTION_MODEL=ImageProcessing
IMAGE_QUERY_DETECTION_MODEL=Yolo26s
SIGN_MODEL_PATH=models/yolo26s_vtsdb100_best.pt
ENABLE_SIGN_RETRIEVAL_FALLBACK=false
LLM_TEMPERATURE=0.0
LLM_MAX_NEW_TOKENS=1200
```

Important variables:

| Variable | Description |
|---|---|
| `LLM_BASE_URL` | OpenAI-compatible endpoint |
| `LLM_MODEL` | Model id served by the endpoint |
| `LLM_API_KEYS` | API key string; use `local` for local servers |
| `IMAGE_LAWDB_DETECTION_MODEL` | Detector for clean LawDB images |
| `IMAGE_QUERY_DETECTION_MODEL` | Detector for uploaded/query images |
| `SIGN_MODEL_PATH` | Fine-tuned YOLO26s checkpoint |
| `ENABLE_SIGN_RETRIEVAL_FALLBACK` | Optional legacy retrieval fallback; disabled for YOLO26s QA |
| `LLM_TEMPERATURE` | LLM sampling temperature |
| `LLM_MAX_NEW_TOKENS` | Maximum generated tokens for JSON answers |

## Start Qdrant

```bash
make docker-up
```

Qdrant should be available at:

```text
http://localhost:6333
```

Stop Qdrant:

```bash
make docker-down
```

## Start a Local GPU LLM

The project uses an OpenAI-compatible API client. Any server that supports
`/v1/chat/completions` with image input can be used.

### Ollama

```bash
ollama pull gemma3:12b
OLLAMA_HOST=0.0.0.0:18989 OLLAMA_ORIGINS="*" ollama serve
```

`.env`:

```env
LLM_BASE_URL=http://localhost:18989/v1
LLM_MODEL=gemma3:12b
LLM_API_KEYS=local
```

### llama.cpp server

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=OFF
cmake --build build --config Release -j -t llama-server

CUDA_VISIBLE_DEVICES=0 ./build/bin/llama-server \
  --model weights/gemma-3-12b-it-Q4_K_M.gguf \
  --mmproj weights/mmproj-BF16.gguf \
  --gpu-layers 99 \
  --ctx-size 5000 \
  --port 18989 \
  --flash-attn
```

`.env`:

```env
LLM_BASE_URL=http://localhost:18989/v1
LLM_MODEL=gemma-3-12b-it
LLM_API_KEYS=local
```

## Prepare LawDB

Run the complete LawDB pipeline:

```bash
make lawdb
```

Equivalent commands:

```bash
python -m src.core.lawdb.preprocess_lawdb_infos
python -m src.core.lawdb.extract_lawdb_sign_images
python -m src.core.lawdb.parse_lawdb_sign_infos
python -m src.core.lawdb.ingest_lawdb_signs
```

For full VLM sign parsing with controlled chunk size:

```bash
python -m src.core.lawdb.parse_lawdb_sign_infos \
  --chunk-size 8 \
  --sleep-sec 0.1 \
  --max-tokens 2500
```

The ingest step stores one Qdrant point per extracted LawDB sign with:

- `law_id`
- `article_id`
- `image_path`
- `sign_name`
- `sign_description`

The Qdrant collection name is generated from the selected LawDB detector and
embedding model, for example:

```text
lawdb_processed_by_imageprocessing_and_clip-gmp-vit-l-14
```

## Run Pipelines

### Article Retrieval

```bash
python -m src.core.sub_task_1 --set_name train
python -m src.core.sub_task_1 --set_name public_test
python -m src.core.sub_task_1 --set_name private_test
```

### Free-form Legal QA

```bash
python -m src.core.sub_task_2 --set_name train
python -m src.core.sub_task_2 --set_name public_test
python -m src.core.sub_task_2 --set_name private_test
```

### Baseline 1: BM25 + LLM

This separate baseline uses the configured fixed sign detector to crop sign
regions, reads text from those crops with Tesseract OCR, searches structured
LawDB article/clause/point text with BM25, and asks the LLM to answer from only
the question, OCR output, and retrieved legal units. It does not call Qdrant,
image embeddings, VLM sign filtering, or the symbolic reasoner.

Install the optional Python OCR wrapper and the system Tesseract binary with
Vietnamese language data (`vie`), then run:

```bash
pip install -e '.[baseline1]'
python -m src.core.baseline1_bm25_llm \
  --image path/to/road-scene.jpg \
  --question "Xe máy có được dừng tại đây không?" \
  --choices $'A. Có\nB. Không'
```

The existing YoloE detector localizes broad sign shapes; it does not identify
sign codes or meanings. The baseline reports that limitation and uses the
question plus OCR text for BM25 when sign identity is unavailable. Tesseract
is optional at runtime: without the OCR dependency or language data, the
pipeline continues with the question alone and reports the OCR status. Set
`LLM_API_KEYS` and the usual LLM settings in `.env` before running.

The QA pipeline writes these fields to each processed sample:

- `freeform_answer`
- `citations`
- `visual_evidence`
- `reasoning`
- `confidence`
- `raw_answer_response`
- `answer_prompt_parts`
- `article_contexts`

For public/private test splits, the pipeline also writes `submission_task2.json`
with the free-form answer payload.

## Answer JSON Format

The model is prompted to return strict JSON:

```json
{
  "answer": "Câu trả lời tự do bằng tiếng Việt, ngắn gọn nhưng đủ căn cứ.",
  "citations": [
    {
      "law_id": "QCVN 41:2024/BGTVT",
      "article_id": "B.31",
      "title": "B.31 Biển số P.131 (a,b,c) \"Cấm đỗ xe\"",
      "evidence": "Biển số P.131b có hiệu lực cấm các loại xe cơ giới đỗ ở phía đường có đặt biển vào những ngày lẻ."
    }
  ],
  "visual_evidence": [
    {
      "sign_name": "Biển số P.131b - Cấm đỗ xe (ngày lẻ)",
      "description": "Biển cấm đỗ xe vào ngày lẻ.",
      "image_name": "demo_sign_0.png"
    }
  ],
  "reasoning": "Ảnh cho thấy biển P.131b; điều B.31 quy định P.131b cấm đỗ xe vào ngày lẻ.",
  "confidence": "high"
}
```

If the raw model output is not valid JSON, the parser keeps
`raw_answer_response` and returns a low-confidence fallback answer.

## Streamlit Demo

Run:

```bash
make ui-freeform-demo
```

or:

```bash
PYTHONPATH=. streamlit run src/ui/freeform_qa_demo.py
```

The demo supports:

- image upload;
- free-form Vietnamese question input;
- selection between the RAG baseline, YOLO26s neuro-symbolic pipeline,
  BM25 + LLM, and VLM-only;
- optional choices for pipelines that use them;
- pipeline-specific answers, citations, OCR/retrieval details, and visual
  evidence where those outputs apply.

### Capture all four pipelines on the shared 20-image set

Run:

```bash
make test-four-pipelines
```

The runner saves each case's question, pipeline name, answer, and per-case
latency to `outputs/four_pipeline_test_results.json`. Pipeline setup time is
excluded from latency. It does not calculate answer correctness or other
evaluation scores; use the `gold_answer` values in
[`examples/traffic_sign_vqa_testcases_20/manifest.json`](examples/traffic_sign_vqa_testcases_20/manifest.json)
for a later independent ChatGPT review. Resume an interrupted capture with
`python -m src.eval.four_pipeline_test --resume`.

## Compare YOLOv8m with Image Retrieval

Start Qdrant and make sure the configured LawDB image collection has been
ingested. Then run:

```bash
PYTHONPATH=. streamlit run src/ui/model_comparison.py
```

Upload an image to view YOLOv8m's boxes and class labels beside the original
YoloE → CLIP/Qdrant retrieval results. Each branch processes the uploaded image
independently; retrieval uses YoloE regions rather than YOLOv8m detections. The
YoloE checkpoint is downloaded separately to `models/yoloe-v8l-seg.pt`.

The QA pipeline uses the fine-tuned YOLO26s VTSDB100 checkpoint directly. Put
`yolo26s_vtsdb100_best.pt` at `models/yolo26s_vtsdb100_best.pt` or set
`SIGN_MODEL_PATH` to its location. Its detections use the same metadata fields
as the existing detector adapter. The adapter maps classes with a clear QCVN
correspondence; other classes remain unidentified/ambiguous and do not trigger
retrieval or an LLM identity-selection call, so the symbolic pipeline fails
closed for those signs. The checkpoint training and download steps are in
[`docs/yolo26_vtsdb100_gpu_runbook.md`](docs/yolo26_vtsdb100_gpu_runbook.md).

## Run Demo on a GPU Machine

Start Streamlit with public binding:

```bash
PYTHONPATH=. streamlit run src/ui/freeform_qa_demo.py \
  --server.address 0.0.0.0 \
  --server.port 8501 \
  --server.headless true
```

If the GPU provider maps public port `20019` to internal port `8501`, the demo
URL will be:

```text
http://<gpu-public-ip>:20019
```

Required services on the GPU machine:

- Qdrant on internal port `6333`
- LLM OpenAI-compatible endpoint on internal port `18989`
- Streamlit on internal port `8501`

## Evaluation

The baseline is locked by `configs/baseline_eval.json`. It fixes the 20 train
sample ids, dataset paths, LLM model, retrieval `LIMIT`, detectors, embedding
model, Qdrant collection, judge, temperature, and token limit. Run it with:

```bash
make baseline
```

The evaluator writes aggregate metrics and detailed per-sample inputs, outputs,
and metrics separately:

```text
outputs/baseline_metrics.json
outputs/baseline_rows.json
```

Interrupted runs can continue without recomputing completed samples:

```bash
python -m src.eval.freeform_qa_metrics \
  --config configs/baseline_eval.json \
  --resume
```

The locked legacy RAG baseline keeps its historical config fingerprint. The
evaluator runs it with retrieval enabled and symbolic reasoning disabled.
The neuro-symbolic pipeline uses a separate config and output files:

```bash
make neuro-symbolic
python -m src.eval.freeform_qa_metrics --config configs/neuro_symbolic_eval.json --resume
```

Both use the same locked 20 sample IDs. Neuro-symbolic metrics and rows are
written to `outputs/neuro_symbolic_metrics.json` and
`outputs/neuro_symbolic_rows.json`.

To compare a later run against the baseline:

```bash
make baseline-compare CANDIDATE=outputs/candidate_metrics.json
```

The required baseline metrics are:

- `answer_correctness`
- `citation_precision`
- `citation_recall`
- `evidence_faithfulness`
- `retrieval_recall_at_1`
- `retrieval_recall_at_3`
- `retrieval_recall_at_5`
- `json_validity`
- `latency`

The committed baseline was promoted from the existing judged sign-centric run
associated with repository baseline `f3eb048`. Its provenance and source hash
are recorded in the output files. That legacy evaluator retained generated
answers and per-row metrics but not the full retrieval/citation payload; every
new run produced by the current evaluator retains the full pipeline output.

Committed baseline metrics:

| Metric | Value |
|---|---:|
| Answer correctness | 0.80 |
| Citation precision | 0.65 |
| Citation recall | 0.50 |
| Evidence faithfulness | 1.00 |
| Retrieval Recall@1 | 0.475 |
| Retrieval Recall@3 | 0.525 |
| Retrieval Recall@5 | 0.55 |
| JSON validity | 1.00 |
| Average latency | 17.197s |

The first-20 slice contains many general legal questions, so citation and
retrieval scores are less representative for sign retrieval. The sign-centric
slice is more useful for judging visual grounding and LawDB sign retrieval.

### Baseline 2: VLM-only

This baseline sends the original image and natural-language question directly
to one multimodal model in a single request. It does not run sign detection,
OCR, LawDB lookup, BM25 or vector retrieval, or symbolic reasoning. It uses the
same locked `train_40`–`train_59` sample IDs as the other baseline.

The evaluator uses `gemma-3-12b-it` through the OpenAI-compatible endpoint set
by `LLM_BASE_URL`, with credentials from `LLM_API_KEYS`. It asks for a concise
Vietnamese answer and an uncertainty statement when the evidence is
insufficient. It does not request legal citations; any law mentioned is
model-generated and unverified.

Run it independently with:

```bash
make baseline-vlm-only
```

To resume an interrupted run:

```bash
python -m src.eval.vlm_only --config configs/baseline2_vlm_only.json --resume
```

The evaluator writes `outputs/baseline2_vlm_only_metrics.json` and
`outputs/baseline2_vlm_only_rows.json`. Answer correctness uses a token-overlap
heuristic; citation metrics are not applicable.

## Debug and Utility UIs

```bash
make ui-freeform-demo      # Free-form cited QA demo
make ui-inspect            # Inspect processed LawDB
make ui-inspect-task1      # Inspect retrieval output
make ui-label              # Annotation helper
```

## Development Checks

Compile check:

```bash
python -m compileall src
```

Project status:

```bash
make status
```

Clean caches:

```bash
make clean
```

## Data Layout

Expected layout:

```text
data/
├── law_db/
│   ├── vlsp2025_law_new.json
│   └── images.fld/
├── train_data/
│   ├── vlsp_2025_train.json
│   └── train_images/
├── public_test/
│   ├── vlsp_2025_public_test_task1.json
│   └── public_test_images/
└── private_test/
    ├── vlsp2025_submission_task1.json
    └── private_test_images/
```

Processed files and extracted sign folders are generated automatically from the
selected settings.

## Sign Variant Disambiguation

Some signs are visually similar and share the same legal article. For example,
article `B.31` contains:

- `P.131a`: cấm đỗ xe nói chung.
- `P.131b`: cấm đỗ xe vào ngày lẻ.
- `P.131c`: cấm đỗ xe vào ngày chẵn.

The QA pipeline retrieves top-k candidates, then resolves ambiguous variants
before generating the final answer. The selected candidate is placed in
`primary_visual_evidence`, and the prompt requires legal evidence to match that
specific variant.

## Dataset Note

This project expects the Vietnamese traffic-law QA dataset and LawDB files to be
available under `data/`. Follow the dataset provider's usage terms when using
or redistributing the data.
