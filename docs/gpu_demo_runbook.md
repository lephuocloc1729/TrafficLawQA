# GPU Streamlit Demo Runbook

Use this runbook to prepare the rented GPU machine, start the four-pipeline
Streamlit demo, and check its dependencies before testing. Run commands from
the repository root on the GPU machine. Never put SSH passwords, API keys, or
provider access tokens in this file or in shell history.

## Deployment layout

The current demo uses these services on the GPU host:

| Service | Internal endpoint | Purpose |
| --- | --- | --- |
| Streamlit | `0.0.0.0:8501` | Browser UI |
| Qdrant | `127.0.0.1:6333` | LawDB sign-vector search |
| Ollama OpenAI-compatible API | `127.0.0.1:18989` | Gemma 3 12B inference |

The GPU provider maps internal ports to public ports. Use the provider's current
port mapping when opening Streamlit or Qdrant; do not assume the public port is
the same between rented instances.

## 1. Prepare the repository and environment

```bash
cd /root/TrafficLawQA
git status --short --branch
test -x .venv/bin/python
test -f .env
mkdir -p outputs models
```

Create `.env` from `.env.example` only if this machine does not already have a
working `.env`. Preserve provider-specific settings locally and never commit
`.env`.

Confirm the active configuration without printing credentials:

```bash
grep -E '^(LLM_BASE_URL|LLM_MODEL|IMAGE_LAWDB_DETECTION_MODEL|IMAGE_QUERY_DETECTION_MODEL|SIGN_MODEL_PATH|TRAFFIC_SIGN_OCR_DEVICE|TRAFFIC_SIGN_OCR_ENABLED)=' .env
```

For this demo, `LLM_MODEL` must refer to the Gemma 3 12B model served by Ollama,
`IMAGE_QUERY_DETECTION_MODEL` must be `Yolo26s`, OCR should use the configured
GPU device, and `SIGN_MODEL_PATH` must resolve to the fine-tuned checkpoint.
The OCR extra includes VietOCR plus the PP-OCRv6 text detector. VietOCR downloads
its pretrained `vgg_transformer` weights on first initialization, so allow the
GPU host to reach the model download source before starting the first run.

## 2. Check the GPU and Python runtime

```bash
nvidia-smi
.venv/bin/python -c 'import torch; print("torch CUDA:", torch.cuda.is_available()); print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")'
.venv/bin/python -c 'import paddle, torch, vietocr; from paddleocr import TextDetection; print("Paddle CUDA:", paddle.is_compiled_with_cuda(), "Torch CUDA:", torch.cuda.is_available(), "VietOCR:", vietocr.__file__)'
```

Stop here if CUDA is unavailable or the installed PaddlePaddle build has no
CUDA support. VietOCR uses PyTorch CUDA, while PaddleOCR is retained only for
text-region detection. Do not try to fix a runtime import error by changing
application code; install the GPU-compatible dependencies for this machine
first.

## 3. Check the YOLO26s checkpoint

The app's default path is
`models/yolo26s_vtsdb100_best.pt`. The trained checkpoint may be stored in
`artifacts/yolo26s_checkpoints/best.pt`; create the expected path if it is not
already present:

```bash
if [ ! -s models/yolo26s_vtsdb100_best.pt ]; then
  test -s artifacts/yolo26s_checkpoints/best.pt
  ln -s ../artifacts/yolo26s_checkpoints/best.pt models/yolo26s_vtsdb100_best.pt
fi
ls -lh models/yolo26s_vtsdb100_best.pt
```

If `SIGN_MODEL_PATH` in `.env` points somewhere else, verify that exact file
instead. The path must be readable from the repository root. Do not start the
app while the configured checkpoint is missing or is an empty/broken symlink.

## 4. Check LawDB files and Qdrant

Verify the processed LawDB files used by this deployment exist:

```bash
test -s data/law_db/vlsp2025_law_new_preprocessed_with_imageprocessing.json
test -d data/law_db/signs_imageprocessing
```

On the current rented GPU image, Docker's daemon cannot start because the
container does not have the required networking privileges. Qdrant v1.9.5 is
installed as a standalone binary. If Qdrant is stopped, start that binary with
its persistent storage configuration:

```bash
nohup /root/qdrant/qdrant --config-path /root/qdrant/config.yaml \
  >> /root/qdrant/qdrant.log 2>&1 < /dev/null &
```

Check Qdrant's readiness and the collection configured for this deployment:

```bash
curl --fail --silent --show-error http://127.0.0.1:6333/readyz
curl --fail --silent --show-error \
  http://127.0.0.1:6333/collections/lawdb_processed_by_imageprocessing_and_clip-gmp-vit-l-14
```

The collection response must report status `green` and a nonzero
`points_count` (the current loaded LawDB has 489 points). If the collection is
missing, start Qdrant first, then run the project's LawDB ingest command from
the repository root:

```bash
PYTHONPATH=. .venv/bin/python -m src.core.lawdb.ingest_lawdb_signs
```

Do not rerun preprocessing or ingestion just because Streamlit restarted. Only
ingest when the collection is absent or the source LawDB/index was deliberately
changed.

## 5. Check Ollama and Gemma 3 12B

```bash
curl --fail --silent --show-error http://127.0.0.1:18989/api/tags
curl --fail --silent --show-error http://127.0.0.1:18989/v1/models
```

Confirm the model configured by `LLM_MODEL` appears in the served model list and
that the endpoint is the GPU host's local Ollama service. This demo must use
Gemma 3 12B, not a Gemini API endpoint.

## 6. Start or restart Streamlit

Start from the repository root and set `PYTHONPATH=.`. The latter is required
because the app imports modules from the top-level `src` package. Omitting it
can produce `ModuleNotFoundError: No module named 'src'` even when dependencies
are installed.

```bash
cd /root/TrafficLawQA
mkdir -p outputs
nohup env PYTHONPATH=. PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
  .venv/bin/streamlit run src/ui/freeform_qa_demo.py \
  --server.address 0.0.0.0 \
  --server.port 8501 \
  --server.headless true \
  >> outputs/streamlit.log 2>&1 < /dev/null &
echo "Streamlit PID: $!"
```

For a restart, stop the existing Streamlit process for this app first, then run
the command above. Avoid broad `pkill -f` patterns that could stop the current
shell or another service.

## 7. Pre-test acceptance checks

All checks below must pass before testing the pipelines:

```bash
curl --fail --silent --show-error http://127.0.0.1:8501/_stcore/health
curl --fail --silent --show-error http://127.0.0.1:6333/readyz
curl --fail --silent --show-error http://127.0.0.1:18989/v1/models
test -s models/yolo26s_vtsdb100_best.pt
tail -n 80 outputs/streamlit.log
```

Open the public Streamlit URL from the provider's current port mapping. Test one
representative question in each pipeline before a full dataset run. For a
pipeline that uses LawDB retrieval, confirm the Qdrant collection has points.

## Troubleshooting

| Error | Likely cause | Fix |
| --- | --- | --- |
| `FileNotFoundError: YOLO26s checkpoint not found` | `SIGN_MODEL_PATH` resolves to a missing file | Complete section 3 and restart Streamlit |
| `ModuleNotFoundError: No module named 'src'` | App started outside the repo root or without `PYTHONPATH=.` | Complete section 6 exactly |
| Qdrant connection refused | Qdrant is stopped or `QDRANT_URI` points to the wrong host/port | Start Qdrant and verify section 4 |
| LLM connection refused | Ollama is stopped or `LLM_BASE_URL` is wrong | Start Ollama and verify section 5 |
| Ollama reports model not found | Configured Gemma alias is not installed/served | Pull or configure Gemma 3 12B, then verify `/v1/models` |
| `libGL.so.1` or OpenCV import error | Missing system OpenGL runtime library | Install the OS package providing `libGL.so.1`, then restart the app |
| VietOCR or text detector cannot initialize GPU | CPU-only/mismatched PyTorch or Paddle build, or CUDA runtime issue | Recheck section 2 and install GPU-compatible PyTorch and Paddle builds |

After every restart, check `outputs/streamlit.log` and the Streamlit health
endpoint before retrying a request in the browser. A healthy HTTP endpoint alone
does not prove that the selected pipeline's checkpoint, Qdrant collection, OCR,
and LLM are ready; use the component checks above as well.
