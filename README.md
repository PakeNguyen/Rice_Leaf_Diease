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
