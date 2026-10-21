from enum import Enum


class DetectionName(Enum):
    GroundingDINO = "GroundingDINO"
    ImageProcessing = "ImageProcessing"
    YoloE = "YoloE"
    YoloESegment = "YoloESegment"
    YoloV8m = "YoloV8m"
    Yolo26s = "Yolo26s"


class QuestionType(Enum):
    MultipleChoice = "Multiple choice"
    YesNo = "Yes/No"


IGNORED_ARTICLE_ID = ("A", "G", "I", "K", "M", "O", "P")

DEFAULT_ARTICLES = {
    "B.": {"law_id": "QCVN 41:2024/BGTVT", "article_id": "22"},
    "C.": {"law_id": "QCVN 41:2024/BGTVT", "article_id": "28"},
    "D.": {"law_id": "QCVN 41:2024/BGTVT", "article_id": "32"},
    "E.": {"law_id": "QCVN 41:2024/BGTVT", "article_id": "36"},
}


ANSWER_MAPPING = {
    "Yes": "Đúng",
    "No": "Sai",
    "Không": "Sai",
    "False": "Sai",
    "Fail": "Sai",
    "True": "Đúng",
}
