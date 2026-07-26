import os
import json
import streamlit as st

from PIL import Image
from src.settings import settings

# Configuration
JSON_FILE = "data/private_test/vlsp2025_submission_task2.json"
IMAGES_DIR = "data/private_test/private_test_images"
OUTPUT_FILE = "data/private_test/vlsp2025_submission_task2_corrected.json"

IMAGE_WIDTH = 600

st.set_page_config(page_title="VLSP2025 Legal QA Review", layout="wide")


def load_data():
    """Load the JSON data from file - prefer corrected file if it exists"""
    # First try to load the corrected file if it exists and has answers
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Check if the file has any answers
            answered_count = sum(1 for q in data if "select_answer" in q)
            if answered_count > 0:
                print(f"Debug: Loaded corrected file with {answered_count} answers")
                return data
        except Exception as e:
            print(f"Debug: Error loading corrected file: {e}")

    # Fallback to original file
    try:
        with open(JSON_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Debug: Loaded original file with {len(data)} questions")
        return data
    except FileNotFoundError:
        st.error(f"File {JSON_FILE} not found!")
        return []
    except json.JSONDecodeError:
        st.error("Error reading JSON file!")
        return []


def save_data(data, show_message=True):
    """Save the corrected data to a new file"""
    try:
        # Debug: Check if any questions have select_answer
        answered_count = sum(1 for q in data if "select_answer" in q)
        print(
            f"Debug: Saving {len(data)} questions, {answered_count} have select_answer"
        )

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        if show_message:
            st.success(f"Data saved to {OUTPUT_FILE} ({answered_count} answers)")
    except Exception as e:
        st.error(f"Error saving file: {e}")
        print(f"Debug: Error saving file: {e}")


def display_question(question_data, index, data):
    """Display a single question with all its information"""
    st.header(f"Question {index + 1}")
    st.subheader(f"ID: {question_data.get('id', 'N/A')}")

    # Display image if available
    image_id = question_data.get("image_id", "")
    if image_id:
        image_path = os.path.join(IMAGES_DIR, f"{image_id}.jpg")
        if os.path.exists(image_path):
            try:
                image = Image.open(image_path)
                # Set max width to half of screen
                st.image(image, caption=f"Image ID: {image_id}", width=IMAGE_WIDTH)
            except Exception as e:
                st.error(f"Error loading image: {e}")
        else:
            st.warning(f"Image file {image_path} not found")

    # Display question
    st.markdown("**Question:**")
    st.write(question_data.get("question", "N/A"))

    # Display question type
    question_type = question_data.get("question_type", "N/A")
    st.markdown(f"**Question Type:** {question_type}")

    # Answer selection
    col1, col2 = st.columns([3, 1])

    with col1:
        if question_type == "Multiple choice":
            st.markdown("**Select Answer:**")
            current_answer = question_data.get("select_answer", "")
            choices = question_data.get("choices", {})

            if st.button(
                f"A: {choices.get('A', 'N/A')}",
                key=f"A_{index}",
                use_container_width=True,
                type="primary" if current_answer == "A" else "secondary",
            ):
                question_data["select_answer"] = "A"
                data[index] = question_data  # Ensure the data list is updated
                save_data(data, show_message=False)  # Auto-save after selection
                st.session_state.current_index = min(
                    len(data) - 1, st.session_state.current_index + 1
                )
                st.rerun()

            if st.button(
                f"B: {choices.get('B', 'N/A')}",
                key=f"B_{index}",
                use_container_width=True,
                type="primary" if current_answer == "B" else "secondary",
            ):
                question_data["select_answer"] = "B"
                data[index] = question_data  # Ensure the data list is updated
                save_data(data, show_message=False)  # Auto-save after selection
                st.session_state.current_index = min(
                    len(data) - 1, st.session_state.current_index + 1
                )
                st.rerun()

            if st.button(
                f"C: {choices.get('C', 'N/A')}",
                key=f"C_{index}",
                use_container_width=True,
                type="primary" if current_answer == "C" else "secondary",
            ):
                question_data["select_answer"] = "C"
                data[index] = question_data  # Ensure the data list is updated
                save_data(data, show_message=False)  # Auto-save after selection
                st.session_state.current_index = min(
                    len(data) - 1, st.session_state.current_index + 1
                )
                st.rerun()

            if st.button(
                f"D: {choices.get('D', 'N/A')}",
                key=f"D_{index}",
                use_container_width=True,
                type="primary" if current_answer == "D" else "secondary",
            ):
                question_data["select_answer"] = "D"
                data[index] = question_data  # Ensure the data list is updated
                save_data(data, show_message=False)  # Auto-save after selection
                st.session_state.current_index = min(
                    len(data) - 1, st.session_state.current_index + 1
                )
                st.rerun()

        elif question_type == "Yes/No":
            st.markdown("**Select Answer:**")
            current_answer = question_data.get("select_answer", "")

            col_true, col_false = st.columns(2)

            with col_true:
                if st.button(
                    "Đúng",
                    key=f"True_{index}",
                    use_container_width=True,
                    type="primary" if current_answer == "Đúng" else "secondary",
                ):
                    question_data["select_answer"] = "Đúng"
                    data[index] = question_data  # Ensure the data list is updated
                    save_data(data, show_message=False)  # Auto-save after selection
                    st.session_state.current_index = min(
                        len(data) - 1, st.session_state.current_index + 1
                    )
                    st.rerun()

            with col_false:
                if st.button(
                    "Sai",
                    key=f"False_{index}",
                    use_container_width=True,
                    type="primary" if current_answer == "Sai" else "secondary",
                ):
                    question_data["select_answer"] = "Sai"
                    data[index] = question_data  # Ensure the data list is updated
                    save_data(data, show_message=False)  # Auto-save after selection
                    st.session_state.current_index = min(
                        len(data) - 1, st.session_state.current_index + 1
                    )
                    st.rerun()

    with col2:
        if "select_answer" in question_data:
            st.success(f"Selected: {question_data['select_answer']}")
        else:
            st.info("No answer selected")

        # Clear answer button
        if "select_answer" in question_data:
            if st.button(
                "🗑️ Clear Answer", key=f"clear_{index}", use_container_width=True
            ):
                # Remove the select_answer field
                if "select_answer" in question_data:
                    del question_data["select_answer"]
                data[index] = question_data  # Ensure the data list is updated
                save_data(data, show_message=False)  # Auto-save after clearing
                st.rerun()

    # Display relevant articles
    st.markdown("**Relevant Articles:**")
    relevant_articles = question_data.get("relevant_articles", [])
    if relevant_articles:
        for article in relevant_articles:
            st.write(
                f"- Law ID: {article.get('law_id', 'N/A')}, Article ID: {article.get('article_id', 'N/A')}"
            )
    else:
        st.write("None")

    # Display choices for multiple choice questions
    if question_type == "Multiple choice":
        st.markdown("**Choices:**")
        choices = question_data.get("choices", {})
        for choice_key, choice_text in choices.items():
            st.write(f"{choice_key}: {choice_text}")

    st.divider()


def main():
    st.title("VLSP2025 Legal QA Review System")

    # Load data
    data = load_data()

    if not data:
        st.error("No data loaded. Please check the JSON file.")
        return

    st.write(f"Total questions: {len(data)}")

    # Navigation
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if st.button("⬅️ Previous"):
            if "current_index" in st.session_state:
                st.session_state.current_index = max(
                    0, st.session_state.current_index - 1
                )

    with col2:
        if "current_index" not in st.session_state:
            st.session_state.current_index = 0

        total_questions = len(data)
        current_index = st.session_state.current_index

        # Question selector
        selected_question = st.selectbox(
            "Go to question:",
            range(1, total_questions + 1),
            index=current_index,
            key="question_selector",
        )
        st.session_state.current_index = selected_question - 1

    with col3:
        if st.button("Next ➡️"):
            if "current_index" in st.session_state:
                st.session_state.current_index = min(
                    len(data) - 1, st.session_state.current_index + 1
                )

    # Progress bar
    progress = (st.session_state.current_index + 1) / len(data)
    st.progress(progress)
    st.write(f"Question {st.session_state.current_index + 1} of {len(data)}")

    # Display current question
    if 0 <= st.session_state.current_index < len(data):
        display_question(
            data[st.session_state.current_index], st.session_state.current_index, data
        )

    # Save button
    col_save1, col_save2, col_save3 = st.columns([1, 1, 1])

    with col_save2:
        if st.button("💾 Save All Answers", use_container_width=True):
            save_data(data)
            st.rerun()

    # Statistics
    st.sidebar.header("Statistics")
    answered_questions = sum(1 for q in data if "select_answer" in q)
    st.sidebar.write(f"Answered: {answered_questions}/{len(data)}")
    st.sidebar.write(".1f")

    # Question type breakdown
    question_types = {}
    for q in data:
        qt = q.get("question_type", "Unknown")
        question_types[qt] = question_types.get(qt, 0) + 1

    st.sidebar.header("Question Types")
    for qt, count in question_types.items():
        answered = sum(
            1 for q in data if q.get("question_type") == qt and "select_answer" in q
        )
        st.sidebar.write(f"{qt}: {answered}/{count}")


if __name__ == "__main__":
    main()
