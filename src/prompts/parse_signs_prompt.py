PARSE_SIGNS_PROMPT = """
Bạn đang được cung cấp tiêu đề, nội dung một điều trong Quy chuẩn kỹ thuật quốc gia Việt Nam về báo hiệu đường bộ, và các ảnh crop biển báo theo đúng thứ tự.

TITLE: <<TITLE>>
CONTENT: 
```
<<CONTENT>>
```

Nhiệm vụ: mô tả chính xác từng biển báo/hình crop từ <<IMAGE_<<FROM_INDEX>>>> đến <<IMAGE_<<TO_INDEX>>>> dựa trên ảnh và nội dung điều luật.

Output bắt buộc là JSON array hợp lệ, có đúng <<NUM_SIGNS>> object theo đúng thứ tự ảnh được gửi. Không markdown, không code fence, không giải thích ngoài JSON.

Schema:
[
  {
    "image_tag": "<<IMAGE_0>>",
    "sign_code": "P.131b",
    "sign_family": "P.131",
    "name": "<Tên biển báo giao thông bằng tiếng Việt>",
    "description": "<Mô tả ngắn bằng tiếng Việt gồm đặc điểm thị giác và ý nghĩa pháp lý>",
    "legal_conditions": {
      "time_condition": "<Điều kiện thời gian hoặc chuỗi rỗng>",
      "vehicle_scope": "<Đối tượng phương tiện hoặc chuỗi rỗng>",
      "direction_scope": "<Phạm vi hướng di chuyển hoặc chuỗi rỗng>"
    }
  }
  ...
]

Yêu cầu:
- Object thứ nhất tương ứng ảnh crop thứ nhất được gửi trong request này, object thứ hai tương ứng ảnh crop thứ hai, v.v.
- image_tag dùng chỉ số tuyệt đối trong article, bắt đầu từ <<IMAGE_<<FROM_INDEX>>>>.
- sign_code chỉ được lấy nguyên văn từ TITLE hoặc CONTENT, sau đó chuẩn hóa dấu phẩy thành dấu chấm, ví dụ P,131b thành P.131b.
- Tuyệt đối không suy đoán hoặc tự tạo sign_code từ hình ảnh. Nếu TITLE/CONTENT không nêu rõ mã của đúng biển thì sign_code phải là chuỗi rỗng "".
- sign_family là mã bỏ hậu tố biến thể chữ cái, ví dụ P.131b thành P.131. Nếu sign_code rỗng thì sign_family cũng phải rỗng.
- name phải ưu tiên mã/tên biển trong điều luật nếu có, ví dụ "Biển số P.131a - Cấm đỗ xe".
- description phải nêu cả dấu hiệu nhìn thấy trong ảnh crop và ý nghĩa/quy định pháp lý nếu context có.
- legal_conditions chỉ ghi điều kiện được nêu trực tiếp trong TITLE/CONTENT; trường nào không có bằng chứng thì trả chuỗi rỗng.
- Nếu không xác định được tên riêng, đặt name là "Biển báo chưa xác định từ context" và mô tả những gì nhìn thấy; không bịa mã biển.
- Chỉ dùng tiếng Việt cho name, description và legal_conditions.
"""
