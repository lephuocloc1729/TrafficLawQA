# Neuro-Symbolic TrafficLawVQA Capstone Task Plan

This plan assumes the current `LexiSignVQA-main` project is the starting
codebase. The goal is to turn the existing image-based RAG/VLM pipeline into a
neuro-symbolic Vietnamese traffic-law VQA system with structured facts,
deterministic rules, reasoning traces, and short-video evidence.

## Target Outcome

By the end of the capstone, the system should support:

- Image input and short-video input.
- Traffic-sign detection and LawDB retrieval from the existing pipeline.
- Structured sign facts, question intent, symbolic rule reasoning, and trace.
- Vietnamese final answers with citations, visual evidence, and optional
  timestamp evidence.
- Evaluation comparing the current baseline against the neuro-symbolic version.

## System Variants To Compare

Use these variants throughout development and evaluation.

| Variant | Name | Description |
|---|---|---|
| V0 | Current LexiSignVQA | Detection + retrieval + LLM answer. |
| V1 | RAG baseline | Current pipeline with stable prompts and metrics. |
| V2 | Structured facts | V1 + normalized traffic-sign facts. |
| V3 | Neuro-symbolic image QA | V2 + question intent + rule engine + trace. |
| V4 | Neuro-symbolic video QA | V3 + video frame sampling and timestamp evidence. |

## Month 1: Baseline Stabilization

### Week 1: Confirm Existing Project Baseline

- [ ] Create a clean `.env` for local or GPU-server execution.
  - Output: working model, Qdrant, detector, and embedding configuration.
  - Check: `python -m src.core.freeform_qa` or Streamlit demo can load resources.

- [ ] Start Qdrant and verify the collection name.
  - Output: Qdrant running locally.
  - Check: `make docker-up`, then confirm `settings.collection_name`.

- [ ] Run the LawDB preprocessing pipeline if the index is missing.
  - Output: processed LawDB JSON, extracted sign crops, Qdrant collection.
  - Check: `make lawdb` finishes or each LawDB step finishes individually.

- [ ] Run the current Streamlit image demo.
  - Output: image upload, question input, answer, citations, visual evidence.
  - Check: demo can answer at least 3 train/public examples end to end.

- [ ] Save screenshots and notes for the current system.
  - Output: baseline demo evidence for your report.
  - Check: screenshots show detection, answer, citations, and limitations.

### Week 2: Baseline Evaluation

- [ ] Run the current free-form QA metrics on 20 samples.
  - Output: `outputs/baseline_v1_20.json`.
  - Suggested command:

```bash
python -m src.eval.freeform_qa_metrics --num-samples 20 --output outputs/baseline_v1_20.json
```

- [ ] Run the same evaluation with LLM judge if feasible.
  - Output: `outputs/baseline_v1_20_judge.json`.
  - Suggested command:

```bash
python -m src.eval.freeform_qa_metrics --num-samples 20 --judge --output outputs/baseline_v1_20_judge.json
```

- [ ] Create an error table with 20 evaluated samples.
  - Output: spreadsheet or markdown table with failure type per sample.
  - Failure labels:
    - sign detection failure
    - wrong sign selected
    - wrong retrieved article
    - right article but wrong reasoning
    - answer correct but citation wrong
    - JSON invalid
    - hallucinated legal detail

- [ ] Pick the first rule-scope sign groups.
  - Output: final list of 5-8 sign families for symbolic rules.
  - Recommended first scope:
    - `P.130`: no stopping and no parking
    - `P.131a`: no parking
    - `P.131b`: no parking on odd days
    - `P.131c`: no parking on even days
    - `P.127`: speed limit variants
    - `P.123a/b`: no left/right turn
    - `P.102`: no entry
    - supplementary signs for time, distance, and direction

## Month 2: Symbolic Core

### Week 3: Structured Sign Fact Schema

- [x] Create `src/core/symbolic/sign_schema.py`.
  - Output: dataclasses or Pydantic models for symbolic data.
  - Minimum models:
    - `TrafficSignFact`
    - `QuestionIntent`
    - `RuleMatch`
    - `SymbolicDecision`
    - `ReasoningTrace`

- [x] Define the minimum `TrafficSignFact` fields.
  - Required fields:
    - `sign_code`
    - `sign_family`
    - `sign_name`
    - `restriction_type`
    - `target_vehicle`
    - `time_condition`
    - `direction_scope`
    - `distance_scope`
    - `related_articles`
    - `visual_evidence`
    - `confidence`
    - `source`

- [x] Add JSON serialization helpers.
  - Output: facts and decisions can be dumped into the final result JSON.
  - Check: unit test or small script converts one fact to dict cleanly.

### Week 4: Sign Normalization

- [x] Create `src/core/symbolic/sign_normalizer.py`.
  - Output: retrieved LawDB payloads become structured sign facts.

- [x] Implement sign-code extraction.
  - Examples:
    - `Biển số P.131a Cấm đỗ xe` -> `P.131a`
    - `P.131b` -> family `P.131`
    - `P.127` variants -> family `P.127`

- [x] Implement rule-scope mapping for known signs.
  - Output: static mapping table from sign code to symbolic meaning.
  - Example:

```python
P131B = {
    "restriction_type": "no_parking",
    "time_condition": "odd_days",
    "target_vehicle": "motor_vehicle",
}
```

- [x] Add fallback behavior.
  - If sign code is unknown, create a fact with `restriction_type="unknown"` and
    preserve the retrieved article metadata.

- [x] Integrate facts into the current pipeline without changing answers yet.
  - Output: V2 pipeline returns `symbolic_facts` in debug/result JSON.
  - Check: current answer quality should not regress, because reasoning is not
    active yet.

### Week 5: Question Intent Parser

- [x] Create `src/core/symbolic/question_intent.py`.
  - Output: Vietnamese question becomes small structured intent JSON.

- [x] Start with deterministic keyword parsing.
  - Intents:
    - `CHECK_PERMISSION`
    - `QUERY_SIGN`
    - `INFORMATION`
    - `PENALTY`
    - `UNKNOWN`

- [x] Extract action.
  - Actions:
    - `parking`
    - `stopping`
    - `turn_left`
    - `turn_right`
    - `entering`
    - `speed`
    - `circulation`

- [x] Extract vehicle subject when possible.
  - Subjects:
    - `car`
    - `motorbike`
    - `truck`
    - `bus`
    - `bicycle`
    - `pedestrian`
    - `motor_vehicle`
    - `unknown`

- [x] Add optional LLM parser fallback.
  - Output: if keyword parser is unsure, ask the LLM to return strict JSON.
  - Keep the schema tiny and validate it before use.

### Week 6: Manual Rule Engine

- [x] Create `src/core/symbolic/traffic_rules.py`.
  - Output: manually encoded rules for the first sign scope.

- [x] Create `src/core/symbolic/reasoner.py`.
  - Output: applies rules to facts and question intent.

- [x] Implement permission rules.
  - Required first rules:
    - `P130_NO_STOPPING_NO_PARKING`
    - `P131A_NO_PARKING`
    - `P131B_ODD_DAY_NO_PARKING`
    - `P131C_EVEN_DAY_NO_PARKING`
    - `P123A_NO_LEFT_TURN`
    - `P123B_NO_RIGHT_TURN`
    - `P102_NO_ENTRY`
    - `P127_SPEED_LIMIT`

- [x] Implement rule output.
  - Each rule should return:
    - `rule_id`
    - `decision`
    - `matched_facts`
    - `legal_basis`
    - `reason`
    - `confidence`

- [x] Add no-rule fallback.
  - If no rule fires, return `decision="UNKNOWN"` and let the existing RAG/LLM
    path answer with low or medium confidence.

### Week 7: Integrate Symbolic Decision Into Answer Generation

- [x] Modify answer generation prompt inputs to include:
  - `symbolic_facts`
  - `question_intent`
  - `rules_fired`
  - `symbolic_decision`
  - `reasoning_trace`

- [x] Update the final answer schema.
  - Add:
    - `symbolic_decision`
    - `rules_fired`
    - `symbolic_facts`
    - `reasoning_trace`

- [x] Update the prompt rule.
  - The LLM should explain the symbolic decision, not override it unless the
    decision is `UNKNOWN`.

- [x] Test 10 examples manually.
  - Output: table comparing V1 answer vs V3 answer.
  - Check: rule-based cases should become more consistent and more explainable.

### Week 8: Symbolic Unit Tests

- [x] Add tests for sign-code extraction.
  - Cases: `P.131a`, `P.131b`, `P.131c`, `P.130`, `P.127`, malformed text.

- [x] Add tests for question intent parsing.
  - Vietnamese questions for parking, stopping, turning, entry, speed.

- [x] Add tests for rule firing.
  - Example: `P.131b` + parking + odd day -> prohibited.

- [x] Add tests for unknown or unsupported cases.
  - Check: unsupported signs do not produce fake confident decisions.

## Month 3: Video Evidence And UI

### Week 9: Video Frame Sampling

- [ ] Add a video dependency if needed.
  - Suggested: OpenCV if compatible with your environment.

- [ ] Create `src/core/video/frame_sampler.py`.
  - Output: sampled frames from short video.
  - Parameters:
    - `sample_fps`
    - `max_frames`
    - `min_interval_seconds`

- [ ] Save sampled frames to a temp run directory.
  - Output format:
    - `frame_000001_t0.50.jpg`
    - `frame_000002_t1.00.jpg`

- [ ] Add metadata for each frame.
  - Required:
    - `frame_index`
    - `timestamp_sec`
    - `image_path`

### Week 10: Video Sign Detection And Deduplication

- [ ] Create `src/core/video/video_signs.py`.
  - Output: detected signs across sampled frames.

- [ ] Reuse the existing image detector for each sampled frame.
  - Check: no separate detector is needed at first.

- [ ] Group repeated detections.
  - Simple first approach:
    - retrieve top sign code per crop
    - group by normalized `sign_code`
    - keep all timestamps

- [ ] Select the best crop per group.
  - Heuristics:
    - highest detector confidence if available
    - largest crop area
    - sharpest crop
    - best retrieval score if available

- [ ] Return video evidence.
  - Required fields:
    - `sign_code`
    - `best_crop_path`
    - `frame_index`
    - `timestamp_sec`
    - `all_timestamps`
    - `visual_evidence`

### Week 11: Video QA Pipeline

- [ ] Create `src/core/video_qa.py`.
  - Output: `answer_video_question(video_path, question, ...)`.

- [ ] Reuse the image QA symbolic components.
  - Do not create a separate reasoning system for video.

- [ ] Convert video sign groups into symbolic facts.
  - Add timestamp evidence to each fact.

- [ ] Generate final answer with timestamp evidence.
  - Example output sentence:
    - `Biển P.130 xuất hiện rõ nhất ở giây 2.5 của video.`

- [ ] Test 3-5 short videos.
  - If you lack real videos, create quick phone-recorded examples or simple
    screen recordings from existing images as a development-only set.

### Week 12: Streamlit UI Upgrade

- [ ] Update `src/ui/freeform_qa_demo.py`.
  - Support image or video upload.

- [ ] Add input type routing.
  - Image -> `answer_image_question`
  - Video -> `answer_video_question`

- [ ] Add UI sections.
  - Detected signs
  - Structured facts
  - Rules fired
  - Reasoning trace
  - Citations
  - Visual/frame evidence
  - Debug JSON

- [ ] Keep the UI demo-friendly.
  - Show the final answer first.
  - Put symbolic internals in expanders.
  - Do not overwhelm the main screen.

## Month 4: Evaluation, Report, And Defense

### Week 13: Neuro-Symbolic Metrics

- [ ] Extend `src/eval/freeform_qa_metrics.py`.
  - Add:
    - `rule_coverage`
    - `rule_accuracy`
    - `symbolic_decision_accuracy`
    - `trace_validity`
    - `supported_decision_rate`

- [ ] Define rule coverage.
  - Formula:
    - samples where at least one rule fires / total samples.

- [ ] Define rule accuracy.
  - Formula:
    - rule decision correct / samples where rule fires.

- [ ] Define supported decision rate.
  - Formula:
    - symbolic decision has both legal citation and visual evidence / samples
      with symbolic decision.

- [ ] Save evaluation outputs per variant.
  - Suggested files:
    - `outputs/eval_v1_rag_baseline.json`
    - `outputs/eval_v2_structured_facts.json`
    - `outputs/eval_v3_neuro_symbolic_image.json`
    - `outputs/eval_v4_neuro_symbolic_video.json`

### Week 14: Ablation Experiments

- [ ] Run baseline V1 on the same sample set.
- [ ] Run V3 neuro-symbolic image QA on the same sample set.
- [ ] Compare metrics.
  - Required table:
    - answer correctness
    - citation accuracy
    - retrieval recall@5
    - evidence faithfulness
    - hallucination rate
    - JSON validity
    - latency
    - rule coverage
    - rule accuracy

- [ ] Create qualitative case studies.
  - Include at least:
    - one success case
    - one retrieval failure
    - one visual ambiguity case
    - one symbolic rule improvement case
    - one unsupported/unknown case

- [ ] Decide final claims honestly.
  - Good claim:
    - symbolic rules improve interpretability and consistency for covered signs.
  - Risky claim:
    - symbolic rules solve all traffic-law VQA.

### Week 15: Final Report

- [ ] Write the method section.
  - Sections:
    - current LexiSignVQA baseline
    - structured sign representation
    - question intent parsing
    - symbolic legal rule engine
    - short-video evidence processing
    - answer generation with citations

- [ ] Write the experiment section.
  - Include exact sample sizes, model settings, metrics, and limitations.

- [ ] Add architecture diagrams.
  - Required diagrams:
    - current baseline pipeline
    - proposed neuro-symbolic pipeline
    - video evidence pipeline
    - symbolic reasoning trace example

- [ ] Add limitations.
  - Mention:
    - only selected sign families covered
    - detector errors propagate
    - supplementary signs are hard
    - video support is frame-based, not full temporal scene understanding
    - outputs are educational, not official legal advice

### Week 16: Demo And Defense Preparation

- [ ] Prepare 5 demo scenarios.
  - Suggested:
    - `P.130`: stopping/parking prohibited
    - `P.131b`: odd-day parking
    - `P.131c`: even-day parking
    - `P.123a/b`: turn restriction
    - short video with repeated sign detection

- [ ] Prepare failure demo gracefully.
  - Show a case where the system says confidence is low or unsupported.
  - This makes the system look more honest and engineered.

- [ ] Prepare defense slides.
  - Slides:
    - problem
    - current baseline
    - why neuro-symbolic
    - architecture
    - rule engine example
    - video evidence example
    - evaluation results
    - limitations
    - future work

- [ ] Freeze final code.
  - Tag or branch the final submission state.
  - Save final demo command and environment settings.

## Suggested File Structure

Add these files gradually:

```text
src/core/symbolic/
├── __init__.py
├── sign_schema.py
├── sign_normalizer.py
├── question_intent.py
├── traffic_rules.py
└── reasoner.py

src/core/video/
├── __init__.py
├── frame_sampler.py
└── video_signs.py

src/core/video_qa.py
```

Optional tests:

```text
tests/
├── test_sign_normalizer.py
├── test_question_intent.py
├── test_traffic_rules.py
└── test_reasoner.py
```

## Minimum Viable Neuro-Symbolic Output

Your final result JSON should include at least:

```json
{
  "answer": "...",
  "citations": [],
  "visual_evidence": [],
  "symbolic_facts": [],
  "question_intent": {},
  "rules_fired": [],
  "symbolic_decision": {
    "decision": "PROHIBITED",
    "confidence": "high"
  },
  "reasoning_trace": []
}
```

## Weekly Working Routine

Use this routine every week.

- [ ] Pick 3-5 tasks from the current week.
- [ ] Create or update code.
- [ ] Run one small manual example.
- [ ] Run or add one test.
- [ ] Write down one failure case.
- [ ] Commit or save a stable checkpoint.
- [ ] Update the report with what changed.

## Definition Of Done

The project is done when:

- [ ] Current baseline runs end to end.
- [ ] Neuro-symbolic image QA runs end to end.
- [ ] At least 5 sign families have symbolic rules.
- [ ] Final output includes facts, intent, rules fired, decision, trace,
  citations, and answer.
- [ ] Short-video input produces timestamp evidence.
- [ ] Evaluation compares baseline and neuro-symbolic variants.
- [ ] Report includes limitations and failure analysis.
- [ ] Demo can be run from clear commands.
