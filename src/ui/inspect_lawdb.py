import json
import base64
import streamlit as st

from PIL import Image
from io import BytesIO

from src.settings import settings


st.set_page_config(layout="wide")


# Helper functions
def save_txt(text: str, path: str):
    with open(path, "w") as f:
        f.write(text)


def save_json(data: dict, path: str):
    with open(path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


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
    law_data = load_law_data(settings.lawdb_json_sign_parsing_path)
    return law_data


def show_article(article: dict):
    left_column, right_column = st.columns(2)

    with left_column:
        st.write("---")
        st.header(f'ID: {article["id"]}')
        st.header(f'TITLE: {article["title"]}')
        st.markdown(f'TEXT: {article["text"]}')

        with st.expander(f"{len(article['images'])} original images", expanded=True):
            if len(article["images"]) == 0:
                st.write("No images")
            for image in article["images"]:
                st.write(image)
                st.image(f"data/law_db/images.fld/{image}")
            st.write("---")

        with st.expander(f"{len(article['signs'])} cropped images", expanded=True):
            if len(article["signs"]) == 0:
                st.write("No images")
            for image in article["signs"]:
                st.write(image)
                st.image(f"data/law_db/signs_imageprocessing/{image}")
            st.write("---")

    with right_column:
        if article.get("__is_sucessfully_parsing_sign", False):
            st.success("Successfully parsed signs")
        else:
            st.error("Failed to parse signs")
            with st.expander("__debug__", expanded=True):
                st.write(article.get("__is_sucessfully_parsing_sign", False))
                st.write(article.get("__error", ""))
                st.write(article.get("__detailed_signs_for_debug_only", []))
        st.header("Detailed cropped images")
        if len(article.get("detailed_signs", [])) == 0:
            st.write("No images")
        for detailed_sign in article.get("detailed_signs", []):
            with st.expander(detailed_sign.get("name", ""), expanded=True):
                st.write(detailed_sign.get("description", ""))
                st.image(
                    f"data/law_db/signs_imageprocessing/{detailed_sign.get('image', '')}"
                )
                st.json(detailed_sign)


# Load data
law_data = load_law_data_and_train_data()

# Select sample
with st.sidebar:
    page = "Law data"
    mode = st.selectbox(
        "Mode",
        ["Index", "Article id"],
    )
    filter_images = st.checkbox("Filter images", value=True)
    failed_cases = st.checkbox("Failed cases", value=False)

    if page == "Law data":
        law_id = st.selectbox("Select a law", law_data.keys())
        articles = law_data[law_id]

        if filter_images:
            articles = [e for e in articles if len(e.get("signs", [])) > 0]
        if failed_cases:
            articles = [
                e for e in articles if not e.get("__is_sucessfully_parsing_sign", False)
            ]

        st.write(f"Total articles: {len(articles)}")

        if len(articles) > 0:
            if mode == "Index":
                article_index = st.number_input(
                    "Select an article index", 0, max(len(articles) - 1, 0)
                )
                article = articles[article_index]
            elif mode == "Article id":
                article_ids = [e["id"] for e in articles]
                article_id = st.selectbox(
                    "Select an article",
                    list(range(len(article_ids))),
                    format_func=lambda x: article_ids[x],
                )
                article = articles[article_id]

        st.json(article)


if page == "Law data":
    show_article(article)

# run on server 0.0.0.0
# streamlit run visualize_lawdb.py --server.address 0.0.0.0
