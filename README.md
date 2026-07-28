# Traffic Sign Legal QA

Traffic Sign Legal QA is a multimodal question-answering project for Vietnamese
traffic law and road signs. The system takes a road-scene image and a natural
language question, detects relevant traffic signs, retrieves legal evidence from
a local LawDB index, and generates a Vietnamese free-form answer with citations
and visual evidence.

The current QA pipeline returns structured JSON instead of forcing the model to
choose only `A/B/C/D` or `Đúng/Sai`.

## Features

- Detect traffic signs from uploaded road-scene images.
- Parse each detected crop at query time for OCR text, numbers, times,
  vehicles, arrows, colors, shapes, and symbols.
- Group main signs with supplementary signs when they appear together.
- Filter detected signs according to the user question.
- Use question intent metadata during sign filtering.
- Retrieve top-k matching traffic-sign entries from Qdrant.
- Load full legal article context by `law_id` and `article_id`.
- Generate Vietnamese free-form answers with legal citations.
- Attach evidence from legal articles and detected/retrieved signs.
- Force the answer prompt to reason from `primary_sign_group`, OCR text,
  numbers, time ranges, vehicles, arrows, and legal evidence.
- Disambiguate similar sign variants such as `P.131a`, `P.131b`, and `P.131c`.
- Retry once with a JSON repair prompt when the model output is malformed.
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

1. Detect signs in the uploaded road-scene image.
2. Parse each crop with the VLM for OCR and visual attributes.
3. Build sign groups from main and supplementary signs.
4. Infer question intent and use it to select relevant signs.
5. Retrieve similar LawDB signs from Qdrant.
6. Resolve ambiguous variants among top-k retrieved signs.
7. Load legal article text from LawDB.
8. Ask the LLM to produce strict JSON with answer, citations, evidence,
   reasoning, and confidence.
9. If the JSON is malformed, retry once with a JSON repair prompt.

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
- Docker or another Qdrant runtime
- GPU recommended for sign detection, embeddings, and local VLM/LLM inference
- OpenAI-compatible chat completion endpoint

Recommended default components:

- Qdrant: `http://localhost:6333`
- LLM endpoint: `http://localhost:18989/v1`
- LLM model: `gemma3:12b` or another multimodal instruction model
- LawDB sign extraction: `ImageProcessing`
- Query sign detection: `YoloE`
- Embedding model: `CLIP-GmP-ViT-L-14`
- Retrieval limit: `5`

## Installation

```bash
git clone <your-repository-url>
cd traffic-sign-legal-qa

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -e .
```

With `uv`:

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Configuration

Create `.env` in the project root. You can start from `.env.example`.

Recommended local GPU configuration:

```env
LLM_BASE_URL=http://localhost:18989/v1
LLM_MODEL=gemma3:12b
LLM_API_KEYS=local

IMAGE_LAWDB_DETECTION_MODEL=ImageProcessing
IMAGE_QUERY_DETECTION_MODEL=YoloE
EMBEDDING_NAME=CLIP-GmP-ViT-L-14

QDRANT_URI=http://localhost:6333
LIMIT=5
LLM_TEMPERATURE=0.0
LLM_MAX_NEW_TOKENS=1200
```

Important variables:

| Variable | Description |
|---|---|
| `LLM_BASE_URL` | OpenAI-compatible endpoint |
| `LLM_MODEL` | Model id served by the endpoint |
| `LLM_API_KEYS` | API key string; use `local` for local servers |
| `QDRANT_URI` | Qdrant URL |
| `EMBEDDING_NAME` | Vision embedding model |
| `IMAGE_LAWDB_DETECTION_MODEL` | Detector for clean LawDB images |
| `IMAGE_QUERY_DETECTION_MODEL` | Detector for uploaded/query images |
| `LIMIT` | Top-k retrieval candidates |
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
- optional choices/reference text;
- resource warm-up before presentation;
- persistent result cache for repeated demo samples;
- detected sign crops;
- selected relevant signs;
- free-form answer;
- citations and legal evidence;
- visual evidence;
- raw JSON/debug information.

### Presentation Mode

For a smoother live demo:

1. Start Qdrant and the local LLM endpoint.
2. Open the Streamlit demo.
3. Click `Warm up` in the sidebar to load the detector, embedding model,
   Qdrant collection, and LLM.
4. Keep `Use cached results` enabled.
5. Run each planned demo image/question once before presenting.
6. During the presentation, upload the same image and ask the same question.
   The app will reuse the cached answer and sign crops from `outputs/demo_cache`.

The cache key includes the uploaded image hash, question, optional choices,
retrieval limit, detector names, embedding model, collection name, and LLM
model. If any of these change, the app will run the full pipeline again.

To clear cached demo answers:

```bash
make clean-demo-cache
```

You can also clear the cache from the sidebar with `Clear cache`.

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

Run a small free-form QA evaluation:

```bash
PYTHONPATH=. python -m src.eval.freeform_qa_metrics \
  --num-samples 20 \
  --judge \
  --output outputs/freeform_qa_metrics_20_judge.json
```

Run a sign-centric slice:

```bash
PYTHONPATH=. python -m src.eval.freeform_qa_metrics \
  --begin-idx 39 \
  --num-samples 20 \
  --judge \
  --output outputs/freeform_qa_metrics_signcentric_20_judge.json
```

Metrics:

- `answer_correctness`
- `citation_accuracy`
- `citation_precision`
- `citation_recall`
- `evidence_faithfulness`
- `evidence_faithfulness_proxy`
- `visual_grounding_accuracy_proxy`
- `retrieval_recall_at_1`
- `retrieval_recall_at_3`
- `retrieval_recall_at_5`
- `json_validity`
- `hallucination`
- `latency`

Recent smoke-test metrics:

| Metric | First 20 train | Sign-centric 20 |
|---|---:|---:|
| Answer correctness | 0.85 | 0.75 |
| Citation accuracy | 0.00 | 0.60 |
| Evidence faithfulness | 1.00 | 1.00 |
| Visual grounding accuracy proxy | 0.00 | 0.55 |
| Retrieval Recall@1 | 0.00 | 0.45 |
| Retrieval Recall@3 | 0.00 | 0.525 |
| Retrieval Recall@5 | 0.00 | 0.55 |
| JSON validity | 1.00 | 1.00 |
| Hallucination | 0.15 | 0.05 |
| Average latency | 14.15s | 14.48s |

The first-20 slice contains many general legal questions, so citation and
retrieval scores are less representative for sign retrieval. The sign-centric
slice is more useful for judging visual grounding and LawDB sign retrieval.

## Debug and Utility UIs

```bash
make ui-freeform-demo      # Free-form cited QA demo
make ui-inspect            # Inspect processed LawDB
make ui-inspect-task1      # Inspect retrieval output
make ui-edit-lawdb-signs   # Review and edit parsed sign descriptions
make ui-label              # Annotation helper
```

### Edit Parsed Sign Descriptions

Use the sign description editor to manually review LawDB sign crops and update
the parsed `name` / `description` fields directly in the selected JSON file:

```bash
make ui-edit-lawdb-signs
```

The editor shows one cropped sign at a time, its source `law_id`,
`article_id`, article title/text, current parsed metadata, and editable fields.
When `Save changes` is clicked, the JSON file is updated atomically and a
one-time `.bak` backup is created next to the original file.

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
