import json
import base64
from PIL import Image
from io import BytesIO
from typing import Any, Iterable, Sequence, Tuple


def save_txt(text: str, path: str):
    with open(path, "w") as f:
        f.write(text)


def save_json(data: dict, path: str):
    with open(path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def pil_to_base64_png(
    image: Image.Image,
    new_width: int | None = None,
    png_mime: str = "image/png",
) -> Tuple[str, str]:
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGB")

    if new_width:
        ratio = new_width / image.width
        new_height = max(1, int(image.height * ratio))
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    buf = BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8"), png_mime


def chunked(seq: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def normalize_bool_tokens(s: str) -> str:
    return (
        s.replace("true", "True")
        .replace("false", "False")
        .replace("TRUE", "True")
        .replace("FALSE", "False")
    )
