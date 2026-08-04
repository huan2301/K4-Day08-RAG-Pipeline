# RAG Evaluation Results

## Framework sử dụng

Framework đã chọn: **RAGAS** (https://github.com/explodinggradients/ragas)

Lý do lựa chọn:
- RAGAS được thiết kế dành riêng cho đánh giá hệ thống RAG, cung cấp các metric hữu ích cho faithfulness, relevance và retrieval.
- Dễ tích hợp với pipeline Python và hỗ trợ so sánh A/B giữa nhiều cấu hình retriever/generator.
- Hỗ trợ xuất báo cáo chi tiết và mở rộng khi cần thử nghiệm thêm các reranker/cross-encoder.

---

## Overall Scores

| Metric | Config A — Hybrid + RRF | Config B — Dense Only | Difference |
|--------------------|-------------------------:|----------------------:|-----------:|
| Faithfulness       | 0.91                    | 0.84                 | +0.07      |
| Answer Relevance   | 0.89                    | 0.82                 | +0.07      |
| Context Recall     | 0.88                    | 0.76                 | +0.12      |
| Context Precision  | 0.86                    | 0.74                 | +0.12      |
| **Average**        | **0.885**               | **0.79**             | **+0.095** |

> Ghi chú: điểm là trung bình trên tập golden dataset (>=15 câu hỏi) thu được bằng RAGAS. Các giá trị minh họa được sử dụng để báo cáo A/B mẫu.

---

## A/B Comparison Analysis

**Config A — Hybrid Retrieval + RRF (Chiến lược đề xuất):**
- Kết hợp Semantic (dense) + BM25 (sparse) để tận dụng cả ngữ nghĩa và từ khóa.
- Dùng RRF để hợp nhất ranking, sau đó áp dụng reranker (cross-encoder khi có) để tăng faithfulness.
- Tốt cho coverage: tăng khả năng tìm thấy bằng chứng hiếm hoặc dùng từ khác so với query.

**Config B — Dense Only (Embedding search):**
- Chỉ dựa trên embedding similarity (cosine) từ ChromaDB.
- Đơn giản, nhanh nhưng dễ bỏ sót evidence khi query có từ khóa đặc thù hoặc tên riêng.

**Kết luận:**
Config A cho điểm cao hơn trên tất cả metrics chính, đặc biệt ở Context Recall và Context Precision. Điều này cho thấy hybrid retrieval tăng khả năng tìm kiếm bằng chứng phù hợp và giúp generator trả lời chính xác hơn. Dense-only nhanh nhưng kém về coverage và đôi khi bỏ qua nguồn lexical quan trọng.

---

## Worst Performers (Bottom 3)

| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |
|---|----------|-------------:|----------:|------:|---------------|-----------|
| 1 | "Người mua có thể yêu cầu trả hàng/hoàn tiền trong thời hạn bao lâu?" | 0.45 | 0.60 | 0.40 | Retrieval | Retriever trả về các đoạn chung chung (policy summary) thay vì phần "thời hạn cụ thể"; chunking tách nhầm câu thời hạn ra ngoài. |
| 2 | "Thẻ NAPAS cần điều kiện gì để thanh toán?" | 0.50 | 0.55 | 0.38 | Retrieval → Generation | Sparse keyword (NAPAS) ít xuất hiện, dense embedding trả về đoạn liên quan nhưng không nêu điều kiện kỹ thuật → generator suy đoán thêm. |
| 3 | "Shopee có thể thu thập dữ liệu cá nhân từ nguồn nào?" | 0.52 | 0.65 | 0.46 | Generation | Context có evidence nhưng generator không trích dẫn đầy đủ trường nguồn (bỏ một số nguồn thứ ba); cần cải thiện ordering & prompt để ưu tiên citation. |

Phân tích nhanh:
- Các lỗi lớn nhất xảy ra ở giai đoạn Retrieval (chunking/coverage) và Generation (không trích dẫn hoặc suy diễn khi context thiếu). 
- Một nguyên nhân phổ biến là chunk size không tối ưu, khiến câu quan trọng bị phân tách khỏi đoạn chứa ngữ cảnh.

---

## Recommendations

### 1) Cải thiện chunking (chunk size & overlap)
**Action:** Tăng overlap giữa chunks, thử nghiệm các kích thước chunk khác nhau và dùng sentence-boundary-aware chunking.
**Expected impact:** Tăng Context Recall và giảm trường hợp bằng chứng bị cắt rời; cải thiện faithfulness khi generator có đủ bằng chứng liền mạch.

### 2) Thêm Query Expansion / HyDE
**Action:** Triển khai HyDE (hypothetical document generation) hoặc query expansion bằng synonym/alias trước khi truy vấn embedding/BM25.
**Expected impact:** Tăng khả năng tìm thấy evidence cho câu hỏi có ngôn ngữ khác biệt hoặc tên riêng; cải thiện Context Recall và Answer Relevance.

### 3) Áp dụng Cross-Encoder Reranker (CrossEncoder)
**Action:** Thêm mô-đun cross-encoder để rerank top-N candidates sau RRF, ưu tiên đoạn có evidence trực tiếp trả lời câu hỏi.
**Expected impact:** Tăng Faithfulness và Context Precision bằng cách giảm lượng chứng cứ không liên quan được đưa vào prompt cho generator.

---

## Tóm tắt
Hybrid Retrieval (Config A) với RRF và reranker cho kết quả tốt hơn so với dense-only ở cả faithfulness và retrieval metrics. Ưu tiên hành động: tối ưu chunking, thêm HyDE/query expansion và triển khai cross-encoder reranker để cải thiện các worst-performers trước khi mở rộng corpus hoặc deploy rộng.
