import base64
import streamlit as st

from enum import Enum
from PIL import Image
from io import BytesIO
from src.settings import settings
from src.core.utils import load_json, save_json


class PageName(Enum):
    TRAIN_SAMPLES = "Train Samples"
    LAWDB_SAMPLES = "LawDB Samples"


st.set_page_config(layout="wide")


def load_law_data(path: str):
    raw_law_data = load_json(path)
    return {e["id"]: e["articles"] for e in raw_law_data}


def save_law_data(law_data: dict, path: str):
    raw_law_data = [{"id": k, "articles": v} for k, v in law_data.items()]
    save_json(raw_law_data, path)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def create_image_url(img_path: str) -> str:
    with open(img_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    mime_type = "image/jpeg"

    return f"data:{mime_type};base64,{encoded}"


def create_image_url_resized(img_path: str) -> str:
    with Image.open(img_path) as img:
        img = img.convert("RGB")
        img = img.resize((500, int(img.height / img.width * 500)))

        # Save to in-memory buffer
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return f"data:image/jpeg;base64,{encoded}"


def load_law_data_and_train_data():
    law_data = load_law_data(settings.lawdb_json_preprocessing_path)
    train_data = load_json(settings.train_extracted_sign_json_path)
    return law_data, train_data


def find_articles_in_law_db(
    law_data: dict,
    law_id: str,
    article_id: str,
) -> list[dict[str, any]]:
    articles = []
    for article in law_data[law_id]:
        if article["id"] == article_id:
            articles.append(article)
    return articles


def show_article(article: dict):
    st.write("---")
    st.write(f'Article ID: {article["id"]}')
    st.write(f'Article Title: {article["title"]}')
    st.write(f'Article Content: {article["text"]}')
    st.write("---")
    st.write("Images")
    if len(article["images"]) == 0:
        st.write("No images")

    for image in article["images"]:
        st.write(image)
        st.image(f"{settings.lawdb_image_dir}/{image}")

    st.write("---")


# Load data
law_data, train_data = load_law_data_and_train_data()

# Select sample
with st.sidebar:
    page = st.selectbox(
        "Page",
        [PageName.TRAIN_SAMPLES.value, PageName.LAWDB_SAMPLES.value],
    )

    if page == PageName.TRAIN_SAMPLES.value:
        top_k = st.slider("Top-k", 1, 100, 10)
        sample = st.selectbox(
            "Select a sample",
            train_data,
            format_func=lambda x: f'{x["id"]}: {x["image_id"]}',
        )
        image_path = f'{settings.train_image_dir}/{sample["image_id"]}.jpg'
        image = Image.open(image_path).convert("RGB")
        st.image(image)
        st.json(sample)

    elif page == PageName.LAWDB_SAMPLES.value:
        law_id = st.selectbox("Select a law", law_data.keys())
        article_ids = [e["id"] for e in law_data[law_id]]
        article_id = st.selectbox(
            "Select an article",
            list(range(len(article_ids))),
            format_func=lambda x: article_ids[x],
        )
        article = law_data[law_id][article_id]


if page == PageName.TRAIN_SAMPLES.value:
    left_column, right_column = st.columns(2)
    with left_column:
        for detected_sign in sample["detected_signs"]:
            with st.expander(str(detected_sign["is_chosen"]), expanded=True):
                st.image(
                    f'{settings.train_extracted_sign_dir}/{detected_sign["image_name"]}'
                )
    with right_column:
        st.write("Relevant articles")
        for article in sample["predicted_articles"]:
            law_id = article["law_id"]
            article_id = article["article_id"]
            title = f"{law_id}: {article_id}"
            articles = find_articles_in_law_db(law_data, law_id, article_id)
            with st.expander(title):
                for article in articles:
                    show_article(article)
elif page == PageName.LAWDB_SAMPLES.value:
    show_article(article)
