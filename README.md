# **Nhận diện bệnh lúa**

Dự án này tập trung vào việc phát hiện và phân loại các bệnh trên lá lúa bằng cách sử dụng kỹ thuật học sâu. Mô hình được huấn luyện để nhận diện bảy loại bệnh trên lá lúa: **bacterial leaf blight**, **healthy**, **leaf blast**, **leaf scald**, **narrow brown spot**, **rice hispa**, và **sheath blight**.

## **Dữ liệu**

Dữ liệu được sử dụng trong dự án này bao gồm các ảnh lá lúa đã được gán nhãn, mỗi ảnh tương ứng với một trong bảy loại bệnh. Dữ liệu được chia thành hai phần: **huấn luyện** và **kiểm tra**. Lớp **Lua_dataset** chịu trách nhiệm tải và xử lý các ảnh sử dụng API `Dataset` của PyTorch.

## **Yêu cầu**

Để chạy dự án này, bạn cần cài đặt các thư viện sau:

- **PyTorch**
- **torchvision**
- **OpenCV**
- **scikit-learn**
- **matplotlib**
- **TensorBoard**
- **tqdm**

Bạn có thể cài đặt các thư viện này bằng cách sử dụng pip:

```bash
pip install torch torchvision opencv-python scikit-learn matplotlib tensorboard tqdm
```
## **Cách huấn luyện mô hình**
Để huấn luyện mô hình, bạn chạy tệp train.py. Tệp này nhận các tham số từ dòng lệnh như đường dẫn đến dữ liệu, số epoch, learning rate và batch size.

Ví dụ lệnh chạy:
```bash
- **python train.py --data_path "đường/dẫn/đến/dữ liệu" --epochs 100 --batch_size 32 --lr 0.01**
```
## **Kiến trúc mô hình**
Mô hình dựa trên ResNet-50 (mạng nơ-ron tích chập sâu) và đã được điều chỉnh để đầu ra 7 lớp tương ứng với 7 loại bệnh trên lá lúa. Mô hình được huấn luyện với hàm mất mát cross-entropy và tối ưu hóa bằng phương pháp SGD.

## **Quá trình huấn luyện**
Tăng cường dữ liệu: Ảnh được thay đổi kích thước thành 224x224 pixel và chuẩn hóa theo các giá trị trung bình và độ lệch chuẩn của ImageNet.

- **Đo lường hiệu suất:** Trong quá trình huấn luyện, hiệu suất của mô hình được đánh giá thông qua các chỉ số như độ chính xác, precision, recall, F1 score và mAP (mean average precision). Ma trận nhầm lẫn cũng được ghi vào TensorBoard để phân tích thêm.

- **Lưu điểm kiểm tra (checkpoint):** Trạng thái của mô hình sẽ được lưu sau mỗi epoch, với mô hình tốt nhất (dựa trên độ chính xác trên bộ kiểm tra) được lưu riêng.

## **TensorBoard**
Bạn có thể theo dõi quá trình huấn luyện bằng TensorBoard. Sau khi huấn luyện xong, bạn chạy lệnh sau để khởi động TensorBoard:
- **tensorboard --logdir "đường/dẫn/đến/thư/mục/tensorboard/logs"**
Truy cập vào **http://localhost:6006** trên trình duyệt của bạn để theo dõi quá trình huấn luyện.

## **Kết quả**
Sau khi huấn luyện mô hình, bạn sẽ có một mô hình có thể phân loại các bệnh trên lá lúa. Hiệu suất của mô hình có thể được đo lường qua độ chính xác và các chỉ số khác như precision, recall và F1 score.
