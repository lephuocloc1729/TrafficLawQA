# Traffic-sign image-embedding benchmark protocol

## Objective

Compare the project's current CLIP-GmP image embedding with DINOv2 and SigLIP 2
for image-to-image sign retrieval. This is an identity/retrieval benchmark, not
a detector benchmark and not a test of legal reasoning.

## Current executable benchmark

Run `scripts/benchmark_sign_embeddings.py`. By default it compares:

- `CLIP-GmP-ViT-L-14` (`zer0int/CLIP-GmP-ViT-L-14`), the project baseline;
- `dinov2-with-registers-large` (`facebook/dinov2-with-registers-large`);
- `siglip2-base-patch16-224` (`google/siglip2-base-patch16-224`).

The script uses labeled sign crops in
`data/raw/law_db/signs_imageprocessing` and label metadata from the corresponding
parsed LawDB JSON. It excludes records without an unambiguous code label. All
crops from a source figure are held out together, and are excluded from that
fold's gallery. Gallery similarities are max-aggregated by code before ranking.
It reports exact-code Recall@1/3/5, macro Recall@1, MRR, embedding cost, and
per-query top-five candidates. Artifacts go under
`outputs/embedding_benchmark_20261007/`.

## Training decision for the current snapshot

The parsed LawDB contains 489 sign records. After excluding missing images,
ambiguous labels, and duplicate image/code rows, the executable benchmark has
370 labeled crops across 281 labels. Only nine labels have examples from at
least two independent source figures, and only three have three. That is
enough for a small, noisy frozen-embedding pilot, but not for a credible
three-model fine-tuning comparison with separate train, validation, and test
groups. Therefore the script deliberately evaluates frozen encoders and marks
fine-tuning as not run. It does not count augmented copies of one diagram as
independent training examples.

## Training protocol once enough labeled road imagery is collected

1. Build a reviewed manifest with one row per crop: canonical QCVN code,
   family, image path, physical sign-instance ID, capture sequence/location,
   split, and annotation provenance. Keep uncertain labels out of exact-code
   training; retain them for family-level evaluation.
2. Split by physical sign instance and capture sequence—not by individual
   crops—so adjacent frames or alternate crops of one source sign cannot leak
   across train/validation/test. Preserve a fixed unseen-location test set.
3. Train the same metric-learning objective for each encoder, starting from
   pretrained weights. Use positives from different real instances of the
   same exact QCVN code and hard negatives from visually confusable variants.
   Start with a small projection head or parameter-efficient adapters; unfreeze
   the full backbone only if validation data justifies it.
4. Tune checkpoints and confidence thresholds using validation only. Compare
   exact-code and family Recall@K, MRR, per-class recall, unknown false-accept
   rate, accuracy at fixed coverage, and latency. Freeze the final test set
   until all settings are selected.
5. Re-embed the complete LawDB gallery into a separate vector collection for
   each model/checkpoint. Never mix vectors from different encoders or model
   revisions in one collection.

Images of signs from other countries can help train broad sign localization or
visual invariances, but should not be assigned Vietnamese QCVN labels unless an
expert verifies the equivalence. Check the source dataset's license and terms.
