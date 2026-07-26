import os
import re
import html2text
from src.settings import settings
from src.core.utils import load_json, save_txt, save_json


def html_table_to_markdown(html):
    markdown = html2text.html2text(html).replace("**", "")

    pattern = re.compile(
        r"!\[\]\(Quy%20chuẩn%20Việt%20Nam-QCVN%2041_2024-BGTVT\.fld/(image\d+\.(?:png|jpg))\)",
        re.DOTALL,
    )

    def replacer(match):
        return f"<<IMAGE: {match.group(1)} /IMAGE>>"

    return pattern.sub(replacer, markdown)


def replace_tables_with_markdown(text):
    pattern = re.compile(r"<<TABLE:\s*(<table.*?</table>)\s*/TABLE>>", re.DOTALL)

    def table_replacer(match):
        html_table = match.group(1)
        markdown = html_table_to_markdown(html_table)
        return f"""```markdown
{markdown}
```
"""

    matches = pattern.findall(text)
    markdowns = [html_table_to_markdown(match) for match in matches]

    new_text = pattern.sub(table_replacer, text)

    pattern = re.compile(r"<<IMAGE:\s*(image\d{2,5}\.jpg)\s*/IMAGE>>", re.DOTALL)
    images = pattern.findall(new_text)

    def make_image_replacer():
        index = 0

        def replacer(match):
            nonlocal index
            result = f"<<IMAGE_{index}>>"
            index += 1
            return result

        return replacer

    image_replacer = make_image_replacer()
    new_text = pattern.sub(image_replacer, new_text)

    return matches, markdowns, images, new_text


def load_law_data(path: str):
    raw_law_data = load_json(path)
    return {e["id"]: e["articles"] for e in raw_law_data}


if __name__ == "__main__":
    law_data = load_law_data(settings.lawdb_json_path)

    i = 0
    debug_dir = "debug"
    for law_id, articles in law_data.items():
        for article in articles:
            text: str = article["text"]
            (
                matches,
                markdowns,
                images,
                processed_text,
            ) = replace_tables_with_markdown(text)
            count = len(matches)
            for match, markdown in zip(matches, markdowns):
                dirpath = f"{debug_dir}/{i}"
                os.makedirs(dirpath, exist_ok=True)
                save_txt(match, os.path.join(dirpath, f"file.html"))
                save_txt(markdown, os.path.join(dirpath, f"file.markdown"))
                i += 1

            article["images"] = images
            article["num_images"] = len(images)
            article["text"] = processed_text

    processed_law_data = [
        {"id": law_id, "articles": articles} for law_id, articles in law_data.items()
    ]

    save_json(processed_law_data, settings.lawdb_json_preprocessing_path)
