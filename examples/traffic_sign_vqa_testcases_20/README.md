# Bộ 20 test case ảnh, câu hỏi và đáp án chuẩn

Bộ gồm 20 ảnh từ test split của `data/archive.zip` (VTSDB100), mỗi ảnh đi kèm một câu hỏi tiếng Việt, đáp án chuẩn và nhãn/hộp tham chiếu. Seed lấy mẫu: `20261020`.

- Mở `index.html` để xem ảnh, câu hỏi và đáp án chuẩn.
- `manifest.json` chứa dữ liệu đầy đủ; `gold_answer` là đáp án để evaluator chấm.
- `testcases.csv` có thể dùng để nhập hoặc rà soát các test case.
- Khi chạy bốn pipeline, chỉ truyền ảnh và câu hỏi; giữ `gold_answer` ở phía evaluator để tránh lộ đáp án cho mô hình.
- Chạy `make test-four-pipelines` để lưu câu trả lời và latency của từng pipeline vào `outputs/four_pipeline_test_results.json`; có thể tiếp tục lần chạy dở bằng `python -m src.eval.four_pipeline_test --resume`.
- Chấm đúng/sai theo ý nghĩa và thông tin cốt lõi, chấp nhận cách diễn đạt tương đương. Không yêu cầu trùng khớp từng chữ.

Đáp án được soạn từ nhãn tham chiếu của bộ ảnh và QCVN 41:2024/BGTVT ban hành kèm Thông tư 51/2024/TT-BGTVT. Đây là đáp án chuẩn cho benchmark; các tình huống thực tế vẫn cần xét đầy đủ biển phụ, vạch kẻ đường và điều kiện giao thông tại hiện trường. [Văn bản trên Công báo Chính phủ](https://congbao.chinhphu.vn/van-ban/thong-tu-so-51-2024-tt-bgtvt-43387/53158.htm).
