# Retrieval Error Analysis

Samples: **20** · decision threshold: **20%**

## Error frequency

| Error type | Count | Rate |
|---|---:|---:|
| NO_DETECTION | 0 | 0.0% |
| WRONG_SIGN_VARIANT | 0 | 0.0% |
| WRONG_ARTICLE | 5 | 25.0% |
| RIGHT_ARTICLE_LOW_RANK | 3 | 15.0% |
| WRONG_SYMBOLIC_RULE | 0 | 0.0% |
| WRONG_CITATION | 13 | 65.0% |
| WRONG_FINAL_ANSWER | 4 | 20.0% |
| INVALID_JSON | 0 | 0.0% |

## Decision guidance

| Signal | Count | Rate | Triggered | Recommendation |
|---|---:|---:|:---:|---|
| RIGHT_ARTICLE_LOW_RANK | 3 | 15.0% | false | Prioritize reranking or hybrid retrieval; gold articles are retrieved but not near the top. |
| WRONG_SIGN_VARIANT | 0 | 0.0% | false | Improve sign-code extraction and variant resolution before changing article retrieval. |
| WRONG_ARTICLE | 5 | 25.0% | true | Audit detection, embeddings and LawDB coverage; the relevant article is absent from observed top-5 results. |
| RETRIEVAL_CORRECT_BUT_ANSWER_WRONG | 4 | 20.0% | true | For these specific cases, focus on symbolic reasoning or the answer prompt rather than retrieval. |

## Per-sample analysis

| ID | Question | Gold articles | Retrieved top-1 | top-3 | top-5 | Sign code | Symbolic decision | Final answer | Error type |
|---|---|---|---|---|---|---|---|---|---|
| "train_40" | "Trong khoảng thời gian từ 6:00 đến 9:00 và từ 16:30 đến 19:30, xe nào không được phép lưu thông trên đoạn đường này khi gặp biển hình tròn có hình xe máy?" | ["QCVN 41:2024/BGTVT#11", "QCVN 41:2024/BGTVT#B.3"] | unavailable | unavailable | unavailable | unavailable | unavailable | "Trong khoảng thời gian từ 6:00 đến 9:00 và từ 16:30 đến 19:30, xe gắn máy không được phép lưu thông trên đoạn đường này." | ["WRONG_ARTICLE", "WRONG_CITATION"] |
| "train_41" | "Biển báo \"10t\" có ý nghĩa gì và loại xe nào bị hạn chế lưu thông trên đoạn đường này?" | ["QCVN 41:2024/BGTVT#11", "QCVN 41:2024/BGTVT#B.16"] | unavailable | unavailable | unavailable | unavailable | unavailable | "Biển báo \"10t\" có nghĩa là hạn chế trọng tải toàn bộ xe. Theo đó, các phương tiện có tổng trọng tải vượt quá 10 tấn không được phép lưu thông trên đoạn đường này." | ["RIGHT_ARTICLE_LOW_RANK", "WRONG_CITATION"] |
| "train_42" | "Biển báo \"10t\" yêu cầu phương tiện nào không được phép lưu thông trên đoạn đường này?" | ["QCVN 41:2024/BGTVT#11", "QCVN 41:2024/BGTVT#B.15"] | unavailable | unavailable | unavailable | unavailable | unavailable | "Biển báo \"10t\" (P.115) yêu cầu hạn chế trọng tải toàn bộ xe, tức là các phương tiện có tổng trọng tải vượt quá 10 tấn không được phép lưu thông trên đoạn đường này." | ["WRONG_CITATION"] |
| "train_43" | "Biển báo \"người tham gia giao thông không được phép đi bộ\" yêu cầu gì?" | ["QCVN 41:2024/BGTVT#11", "QCVN 41:2024/BGTVT#B.12"] | unavailable | unavailable | unavailable | unavailable | unavailable | "Biển báo \"Cấm người đi bộ\" (P.112) yêu cầu người đi bộ không được phép đi trên đoạn đường có biển báo." | ["WRONG_CITATION"] |
| "train_44" | "Biển báo \"Cấm xe đạp\" yêu cầu cấm phương tiện nào lưu thông trên đoạn đường này?" | ["QCVN 41:2024/BGTVT#11", "QCVN 41:2024/BGTVT#B.10"] | unavailable | unavailable | unavailable | unavailable | unavailable | "Biển báo \"Cấm xe đạp\" (P.110a) cấm xe đạp lưu thông trên đoạn đường." | ["WRONG_CITATION"] |
| "train_45" | "Biển báo có số 3.3m. Câu nào đúng nói về biển báo này?" | ["QCVN 41:2024/BGTVT#11", "QCVN 41:2024/BGTVT#B.18"] | unavailable | unavailable | unavailable | unavailable | unavailable | "Biển báo P.117 \"Hạn chế chiều cao\" có trị số ghi là 3,5m. Theo QCVN 41:2024/BGTVT B.17(c), trị số trên biển là khoảng cách từ điểm cao nhất của mặt đường đến điểm thấp nhất của chướng ngại vật vượt trên đường (chiều cao tĩnh không) trừ đi 0,5 m. Do đó, xe có chiều cao lớn hơn 3,5m - 0,5m = 3m sẽ bị cấm." | ["RIGHT_ARTICLE_LOW_RANK", "WRONG_CITATION"] |
| "train_46" | "Biển báo có hình xe 3 bánh và xe khách yêu cầu phương tiện nào không được phép lưu thông trên đoạn đường này?" | ["QCVN 41:2024/BGTVT#11", "QCVN 41:2024/BGTVT#B.7"] | unavailable | unavailable | unavailable | unavailable | unavailable | "Biển báo này cấm xe ô tô khách và xe ô tô tải." | ["WRONG_ARTICLE", "WRONG_CITATION"] |
| "train_47" | "Xe máy có được phép lưu thông trên đoạn đường này vào các khung giờ từ 6:00 đến 9:00 và từ 16:30 đến 19:30 đúng hay sai?" | ["QCVN 41:2024/BGTVT#11", "QCVN 41:2024/BGTVT#B.11"] | unavailable | unavailable | unavailable | unavailable | unavailable | "Không, xe máy không được phép lưu thông trên đoạn đường này vào các khung giờ từ 6:00 đến 9:00 và từ 16:30 đến 19:30. Biển báo P.111a cấm xe gắn máy đi qua khu vực." | ["WRONG_CITATION"] |
| "train_48" | "Biển báo có hình người đi bộ này yêu cầu người đi bộ không được phép lưu thông trên đoạn đường này. Câu này đúng hay sai?" | ["QCVN 41:2024/BGTVT#11", "QCVN 41:2024/BGTVT#B.12"] | unavailable | unavailable | unavailable | unavailable | unavailable | "Đúng. Biển báo P.112 \"Cấm người đi bộ\" có nghĩa là người đi bộ không được phép lưu thông trên đoạn đường này." | ["WRONG_CITATION"] |
| "train_49" | "Biển báo này yêu cầu các phương tiện giao thông có chiều cao vượt quá 3,3m không được phép lưu thông. Câu này đúng hay sai?" | ["QCVN 41:2024/BGTVT#11", "QCVN 41:2024/BGTVT#B.18"] | unavailable | unavailable | unavailable | unavailable | unavailable | "Câu nói này đúng. Biển P.117 \"Hạn chế chiều cao\" báo hiệu các xe có chiều cao vượt quá trị số ghi trên biển (3,5m) không được phép lưu thông." | ["RIGHT_ARTICLE_LOW_RANK", "WRONG_CITATION"] |
| "train_50" | "Biển báo \"cấm đi ngược chiều\" yêu cầu phương tiện nào không được phép lưu thông trên đoạn đường này?" | ["QCVN 41:2024/BGTVT#11", "QCVN 41:2024/BGTVT#B.1"] | unavailable | unavailable | unavailable | unavailable | unavailable | "Biển \"cấm đi ngược chiều\" (P.102) cấm các loại xe cơ giới và thô sơ đi vào theo chiều đặt biển, trừ các xe được ưu tiên." | ["WRONG_ARTICLE", "WRONG_CITATION"] |
| "train_51" | "Biển báo này chỉ dẫn đường đi đến những điểm nào?" | ["QCVN 41:2024/BGTVT#E.14"] | unavailable | unavailable | unavailable | unavailable | unavailable | "Biển báo này chỉ dẫn đường đi đến Sân bay Trà Nóc (12km), Trung tâm TP. Cần Thơ (2km) và Sóc Trăng (67km)." | [] |
| "train_52" | "Biển báo này chỉ dẫn các điểm đến nào và theo hướng nào?" | ["QCVN 41:2024/BGTVT#E.14"] | unavailable | unavailable | unavailable | unavailable | unavailable | "Biển báo chỉ dẫn đến Hải Dương, Mỹ Hào và Hưng Yên theo các hướng khác nhau." | [] |
| "train_53" | "Biển báo này chỉ dẫn điều gì về khu vực phía trước?" | ["QCVN 41:2024/BGTVT#D.17"] | unavailable | unavailable | unavailable | unavailable | unavailable | "Biển báo này chỉ dẫn rằng khu vực phía trước là bắt đầu khu đông dân cư." | [] |
| "train_54" | "Khi gặp biển báo này, giới hạn tốc độ cho ô tô là bao nhiêu khi có dải phân cách?" | ["QCVN 41:2024/BGTVT#D.17"] | unavailable | unavailable | unavailable | unavailable | unavailable | "Khi gặp biển báo \"Bắt đầu khu đông dân cư\" (R.420), tốc độ tối đa cho phép phụ thuộc vào các biển báo hiệu lệnh khác đặt trong phạm vi khu vực này." | ["WRONG_FINAL_ANSWER"] |
| "train_55" | "Khi gặp biển báo này, giới hạn tốc độ cho ô tô là bao nhiêu khi không có dải phân cách?" | ["QCVN 41:2024/BGTVT#D.17"] | unavailable | unavailable | unavailable | unavailable | unavailable | "Khi gặp biển báo \"Bắt đầu khu đông dân cư\" (R.420), tốc độ tối đa cho phép sẽ được quy định bởi các biển báo hiệu tốc độ trong khu vực đó." | ["WRONG_FINAL_ANSWER"] |
| "train_56" | "Khi gặp biển báo này, giới hạn tốc độ cho xe máy là bao nhiêu khi có dải phân cách ?" | ["QCVN 41:2024/BGTVT#D.17"] | unavailable | unavailable | unavailable | unavailable | unavailable | "Khi gặp biển báo \"Bắt đầu khu đông dân cư\" (R.420), tốc độ tối đa cho phép sẽ được quy định bởi biển số P.127." | ["WRONG_FINAL_ANSWER"] |
| "train_57" | "Biển báo này yêu cầu giới hạn tốc độ cho ô tô là 100 km/h khi có dải phân cách và 80 km/h khi không có dải phân cách. Câu này đúng hay sai?" | ["QCVN 41:2024/BGTVT#22", "QCVN 41:2024/BGTVT#D.18"] | unavailable | unavailable | unavailable | unavailable | unavailable | "Không đủ thông tin để xác định giới hạn tốc độ cụ thể cho ô tô dựa trên hình ảnh và các điều luật được cung cấp." | ["WRONG_ARTICLE", "WRONG_CITATION"] |
| "train_58" | "Khi gặp biển báo này, giới hạn tốc độ cho xe máy là bao nhiêu khi không có dải phân cách?" | ["QCVN 41:2024/BGTVT#D.17"] | unavailable | unavailable | unavailable | unavailable | unavailable | "Khi gặp biển báo \"Bắt đầu khu đông dân cư\", tốc độ tối đa cho phép sẽ được quy định bởi biển số P.127 (biển báo hiệu tốc độ tối đa cho phép)." | ["WRONG_FINAL_ANSWER"] |
| "train_59" | "Biển báo này chỉ dẫn đường đi đến các điểm nào \n" | ["QCVN 41:2024/BGTVT#46", "QCVN 41:2024/BGTVT#P.17"] | unavailable | unavailable | unavailable | unavailable | unavailable | "Biển báo chỉ dẫn đường đi đến Hà Nội cách 145km, Vĩnh Yên cách 91km, Việt Trì cách 75km và TX Phú Thọ cách 53km." | ["WRONG_ARTICLE", "WRONG_CITATION"] |

## Limitations

- Unavailable retrieval or detection details are null; they are not treated as empty results.
- WRONG_SIGN_VARIANT requires gold_sign_code(s) in the input sample.
- WRONG_SYMBOLIC_RULE requires gold_symbolic_decision in the input sample.
- When only aggregate per-sample recall is available, low-rank and wrong-article labels are inferred from those metrics; exact article lists remain unavailable.
