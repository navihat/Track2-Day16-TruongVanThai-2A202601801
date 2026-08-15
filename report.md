# Báo cáo Benchmark LightGBM — Lab 16 (CPU Compute Node)

**Môi trường:** AWS EC2 `t3.micro` (1 vCPU / 1GB RAM — dùng thay `t3.medium` mặc định do tài khoản bị giới hạn Free Tier tại thời điểm triển khai), Ubuntu 22.04, Python 3.10, LightGBM.
**Dataset:** Credit Card Fraud Detection (Kaggle, `mlg-ulb/creditcardfraud`) — 284,807 giao dịch, 30 features, tỉ lệ gian lận ~0.17% (492/284,807).

## Kết quả

| Metric | Kết quả |
|---|---|
| Thời gian load data | 2.43 s |
| Thời gian training | 2.86 s |
| Best iteration | 1 |
| AUC-ROC | 0.9517 |
| Accuracy | 99.89% |
| F1-Score | 0.7273 |
| Precision | 0.6557 |
| Recall | 0.8163 |
| Inference latency (1 dòng) | 1.24 ms |
| Inference throughput (1000 dòng) | ~621,600 dòng/giây |

## Nhận xét

1. **Training time** rất nhanh (2.86s cho ~285K dòng) dù chạy trên `t3.micro` — cấu hình CPU tối thiểu (1 vCPU/1GB RAM). Cho thấy LightGBM tối ưu tốt cho dữ liệu dạng bảng cỡ vừa mà không cần GPU.
2. Early stopping dừng ngay ở **best_iteration = 1**, cho thấy signal phân loại đã rất mạnh từ cây đầu tiên; đồng thời gợi ý `stopping_rounds=30` có thể hơi nhạy trên tập validation nhỏ (~98 mẫu gian lận trong test set) — tăng patience hoặc dùng cross-validation nhiều khả năng sẽ cải thiện AUC hơn nữa.
3. **AUC-ROC = 0.9517** là mức khá tốt cho bài toán mất cân bằng cực đoan này. **Accuracy 99.89%** gây hiểu nhầm (do >99.8% dữ liệu thuộc lớp "không gian lận") — **Precision (0.656)** và **Recall (0.816)** phản ánh đúng hơn: model bắt được ~82% giao dịch gian lận, đổi lại có một tỉ lệ false positive đáng kể.
4. **Inference rất nhanh**: latency 1.24 ms/dòng phù hợp scoring real-time từng giao dịch; throughput ~621K dòng/giây khi predict theo batch cho thấy ngay cả CPU nhỏ nhất cũng đủ sức phục vụ khối lượng lớn nếu inference theo lô.
5. Kết luận: với bài toán tabular ML cỡ vừa như fraud detection, một CPU instance nhỏ là đủ để vừa train vừa serve model với tốc độ cao — không cần đến GPU.

## Ghi chú về môi trường triển khai

Do tài khoản AWS cá nhân gặp lỗi tại thời điểm triển khai, lab này được thực hiện trên tài khoản AWS mượn tạm — chỉ được cấp Access Key/Secret Key để dùng CLI/Terraform, không có quyền đăng nhập AWS Console. Vì vậy không thể truy cập AWS Billing Console để chụp Cost Dashboard theo yêu cầu; thay thế bằng ảnh chụp CLI (`aws ec2 describe-instances`, `aws ec2 describe-nat-gateways`) làm bằng chứng các dịch vụ đang phát sinh chi phí (2 instance `t3.micro` ở trạng thái `running`, 1 NAT Gateway ở trạng thái `available`). Hạ tầng đã được `terraform destroy` ngay sau khi thu thập đủ bằng chứng để tránh phát sinh thêm chi phí cho tài khoản mượn.
