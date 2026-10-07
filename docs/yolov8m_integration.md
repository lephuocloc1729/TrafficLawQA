# YOLOv8m detector integration

The freeform image-QA path now uses the YOLOv8m checkpoint from the local
`Traffic-sign-VietNam-recognition` project as its detector. The checkpoint is
kept at `models/yolov8m_final.pt`; `models/` is ignored by Git. Set
`SIGN_MODEL_PATH` if the file is stored elsewhere. The application requires
the existing pinned `ultralytics` dependency.

The adapter preserves each detection's crop, box, confidence, class id, and
class label. Its class-label crosswalk is in
`src/core/symbolic/catalogs/yolov8m_qcvn_crosswalk.json`. Exact class labels
map to canonical codes; family-level labels remain ambiguous with candidate
variants, and the generic `other` classes stay unresolved. P.127 classes also
preserve their speed values. The mapping describes sign identity only; the
project's QCVN catalog is still an unreviewed extraction, so this is not legal
validation and must not alone authorize a rule decision.

Legacy LawDB image retrieval remains available in `src/core/query_signs.py`,
but is not in the default freeform pipeline. Set
`ENABLE_SIGN_RETRIEVAL_FALLBACK=true` to invoke it only for selected signs that
YOLOv8m did not resolve to a canonical code. Known detections are not sent to
image retrieval. The existing retrieval index, embeddings, and payload code
are retained for explicit fallback and offline evaluation.

Checkpoint SHA-256:
`d68d8310a0f18c4f029e29fb3a6be20c3d4d92776e60de272decafe56901e43a`

The source repository did not provide a clear checkpoint redistribution
license. The binary is therefore local-only and should not be committed or
redistributed until its license/provenance is confirmed.
