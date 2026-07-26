from src.core.utils import load_json
from src.settings import settings


def calculate_score_each_sample(
    relevant_articles: list[dict], predicted_articles: list[dict]
) -> float:

    relevant_article_uids = [
        relevant_article["law_id"] + "#" + relevant_article["article_id"]
        for relevant_article in relevant_articles
    ]
    predicted_article_uids = [
        predicted_article["law_id"] + "#" + predicted_article["article_id"]
        for predicted_article in predicted_articles
    ]

    precision = len(set(predicted_article_uids) & set(relevant_article_uids)) / len(
        predicted_article_uids
    )
    recall = len(set(predicted_article_uids) & set(relevant_article_uids)) / len(
        relevant_article_uids
    )
    f2 = (
        5 * precision * recall / (4 * precision + recall)
        if precision + recall > 0
        else 0
    )
    return precision, recall, f2


def calculate_score(samples: list[dict]) -> float:
    f2_scores = []
    precision_scores = []
    recall_scores = []
    for i, sample in enumerate(samples):
        if "predicted_articles" not in sample:
            continue

        precision, recall, f2 = calculate_score_each_sample(
            sample["relevant_articles"],
            sample["predicted_articles"],
        )

        precision_scores.append(precision)
        recall_scores.append(recall)
        f2_scores.append(f2)

    return {
        "samples": len(samples),
        "precision": sum(precision_scores) / len(precision_scores),
        "recall": sum(recall_scores) / len(recall_scores),
        "f2": sum(f2_scores) / len(f2_scores),
    }


def main():
    train_samples = load_json(settings.train_extracted_sign_json_path)
    print(
        f"File Path: {settings.train_extracted_sign_json_path}\n",
        calculate_score(train_samples),
    )


if __name__ == "__main__":
    main()
