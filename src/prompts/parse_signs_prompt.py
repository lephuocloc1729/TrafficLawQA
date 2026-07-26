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
    "name": "<Tên biển báo giao thông bằng tiếng Việt>",
    "description": "<Mô tả ngắn bằng tiếng Việt gồm: đặc điểm thị giác chính và ý nghĩa/quy định theo nội dung điều luật>"
  },
  {
    "image_tag": "<<IMAGE_1>>",
    "name": "<Tên biển báo giao thông bằng tiếng Việt>",
    "description": "<Mô tả ngắn bằng tiếng Việt gồm: đặc điểm thị giác chính và ý nghĩa/quy định theo nội dung điều luật>"
  }
  ...
]

Yêu cầu:
- Object thứ nhất tương ứng ảnh crop thứ nhất được gửi trong request này, object thứ hai tương ứng ảnh crop thứ hai, v.v.
- image_tag dùng chỉ số tuyệt đối trong article, bắt đầu từ <<IMAGE_<<FROM_INDEX>>>>.
- name phải ưu tiên mã/tên biển trong điều luật nếu có, ví dụ "Biển số P.131a - Cấm đỗ xe".
- description phải nêu cả dấu hiệu nhìn thấy trong ảnh crop và ý nghĩa/quy định pháp lý nếu context có.
- Nếu không xác định được tên riêng, đặt name là "Biển báo chưa xác định từ context" và mô tả những gì nhìn thấy; không bịa mã biển.
- Chỉ dùng tiếng Việt cho name và description.
"""
