# 20 test case ảnh + câu hỏi

Bộ gồm 20 ảnh ngẫu nhiên từ test split của `data/archive.zip` (VTSDB100), mỗi ảnh đi kèm một câu hỏi tiếng Việt về biển báo nổi bật. Seed lấy mẫu: `20261020`.

- Mở `index.html` để xem ảnh và câu hỏi.
- `manifest.json` chứa dữ liệu đầy đủ cùng nhãn/hộp tham chiếu để đối chiếu.
- `testcases.csv` dùng để nhập vào công cụ đánh giá.
- Bộ test có thể dùng chung trên mọi Git branch; ảnh và câu hỏi không phụ thuộc implementation.

Câu hỏi không chứa nhãn tham chiếu. Không có đáp án chuẩn được tạo vì yêu cầu chỉ gồm ảnh và câu hỏi.
