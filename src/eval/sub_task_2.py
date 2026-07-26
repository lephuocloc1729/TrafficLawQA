import unicodedata as ud
from collections import defaultdict, Counter
from typing import Any, Dict, List, Mapping, MutableMapping

from src.settings import settings
from src.core.utils import load_json


def normalize_text(value: Any) -> str:
    # Preserve legacy/int-special-case behavior
    if isinstance(value, int) and value == 40:
        return "A"

    if value is None:
        return ""

    text = str(value)
    return ud.normalize("NFC", text)


def evaluate(samples: List[Mapping[str, Any]]) -> Dict[str, Any]:
    """
    Evaluate predictions from legalqa.py output.

    Each sample is expected to contain:
      - 'predict': model prediction (any type convertible to str)
      - 'answer' : ground truth (any type convertible to str)
      - optional 'question_type': str

    Args:
        samples: List of sample dictionaries.

    Returns:
        Dict with aggregated metrics.
    """
    valid_samples = 0
    correct_predictions = 0

    # Stats by question type and by answer value
    question_type_stats: MutableMapping[str, Dict[str, int]] = defaultdict(
        lambda: {"correct": 0, "total": 0}
    )
    # Ground-truth answer distribution & per-answer accuracy
    answer_distribution: MutableMapping[str, Dict[str, int]] = defaultdict(
        lambda: {"correct": 0, "total": 0}
    )
    # Predicted answer histogram
    predict_distribution: Counter[str] = Counter()

    for idx, sample in enumerate(samples):
        if "predict" not in sample or "answer" not in sample:
            print(
                f"Warning: sample {sample.get('id', idx)} missing 'predict' or 'answer'; skipping."
            )
            continue

        # Normalize both sides to be consistent
        predict = normalize_text(sample["predict"])
        answer = normalize_text(sample["answer"])
        question_type = (
            normalize_text(sample.get("question_type", "Unknown")) or "Unknown"
        )

        valid_samples += 1
        is_correct = predict == answer
        if is_correct:
            correct_predictions += 1

        question_type_stats[question_type]["total"] += 1
        answer_distribution[answer]["total"] += 1
        if is_correct:
            question_type_stats[question_type]["correct"] += 1
            answer_distribution[answer]["correct"] += 1

        predict_distribution[predict] += 1

    accuracy = (correct_predictions / valid_samples) if valid_samples else 0.0

    # Build per-question-type breakdown
    question_type_breakdown: Dict[str, Dict[str, Any]] = {}
    for qtype, stats in question_type_stats.items():
        total = stats["total"]
        q_acc = (stats["correct"] / total) if total else 0.0
        question_type_breakdown[qtype] = {
            "accuracy": round(q_acc, 4),
            "correct": stats["correct"],
            "total": total,
        }

    # Build per-answer breakdown (ground truth)
    answer_breakdown: Dict[str, Dict[str, Any]] = {}
    for ans, stats in answer_distribution.items():
        total = stats["total"]
        ans_acc = (stats["correct"] / total) if total else 0.0
        answer_breakdown[ans] = {
            "accuracy": round(ans_acc, 4),
            "correct": stats["correct"],
            "total": total,
        }

    return {
        "total_samples": len(samples),
        "valid_samples": valid_samples,
        "correct_predictions": correct_predictions,
        "accuracy": round(accuracy, 4),
        "question_type_breakdown": question_type_breakdown,
        "answer_distribution": dict(predict_distribution),
        "answer_breakdown": answer_breakdown,
    }


def pretty_print(results: Mapping[str, Any]) -> None:
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Total Samples: {results['total_samples']}")
    print(f"Valid Samples: {results['valid_samples']}")
    print(f"Correct Predictions: {results['correct_predictions']}")
    print(f"Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print()

    print("Question Type Breakdown:")
    for qtype, stats in sorted(
        results["question_type_breakdown"].items(), key=lambda kv: str(kv[0])
    ):
        print(f"  {qtype}:")
        print(f"    Accuracy: {stats['accuracy']:.4f} ({stats['accuracy']*100:.2f}%)")
        print(f"    Correct: {stats['correct']}/{stats['total']}")
    print()

    print("Answer Distribution (Predictions):")
    for answer, count in sorted(
        results["answer_distribution"].items(), key=lambda kv: str(kv[0])
    ):
        print(f"  {answer}: {count}")
    print()

    print("Answer Breakdown (Ground Truth):")
    for answer, stats in sorted(
        results["answer_breakdown"].items(), key=lambda kv: str(kv[0])
    ):
        print(f"  {answer}:")
        print(f"    Accuracy: {stats['accuracy']:.4f} ({stats['accuracy']*100:.2f}%)")
        print(f"    Correct: {stats['correct']}/{stats['total']}")


def main():
    samples = load_json(settings.train_extracted_sign_json_path)
    if not isinstance(samples, list):
        raise ValueError("Input JSON must be a list of sample objects.")

    results = evaluate(samples)
    pretty_print(results)


if __name__ == "__main__":
    main()
