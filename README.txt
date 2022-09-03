|| 
||    BKAI-NAVER CHALLENGE 2022
||
=========================================================
||    Vietnamese Scene Text Detection and Recognition  ||
=========================================================

Source code of UIT AIClub CS.AI20

=========================================================

TÓM TẮT:
    Đội thi sử dụng kiến trúc Detecter là Bezier-Detection + Bezier Align được tham khảo từ nguồn [https://github.com/VinAIResearch/dict-guided]
    và kiến trúc Recognize là VGG19 + Transfomer được tham khảo từ nguồn [https://github.com/pbcquoc/vietocr]
    Đội thi cũng huấn luyện thêm một mô hình detection là YoloR


=========== CẤU TRÚC FOLDER ===========
(chỉ liệt kê và ghi chú các phần cần thiết kèm theo để BTC có thể reproduce)

    scene_text_pipeline
    |__ Data :                      (-)thư mục lưu data gốc và các data dùng để huấn luyện sau khi xử lý.
        |__ script  :               (0)thư mục chứa các code convert, crop, xử lý data.
        |__ image_private_process : (-)folder ảnh private được xử lý rõ nét
    |__ Pre-processing : 
        |__ straug/augment_data.py: (1) module tăng cường dữ liệu
    |__ Detecter
        |__ dict-guided :           (2) thư mục chứa model Detecter "Bezier-Detection + Bezier Align"
            |__ predict.sh :        (3) fide chạy dự đoán
            |__weights/*_final.pth  (-) file trọng số của model
        |__ yolor
            |__ predict.sh :
            |__weights/*_final.pth
    |__ Recongnize
        |__ vietocr                 (4) thư mục chứa model Recongnize "VGG19 + Transformer"
            |__ predict.sh :        (5) fide chạy dự đoán
            |__weights/*_final.pth  (-) file trọng số của model
    |__ Results
        |__ merge.py :              (6) file merge kết quả cuối cùng
        |__ submission :            (7) thư mục lưu kết quả FINAL sau khi Post-Process


=========== CHUẨN BỊ MÔI TRƯỜNG ===========
**Ghi chú: Đội thi sẽ kí hiệu dấu ! trước các câu lệnh thực hiện trên terminal

Để không bị xung đột trong quá trình cài đặt, 
đội thi đã chuẩn bị 2 Docker Image tương ứng với 2 môi trường của dict-guided và vietocr

BTC sử dụng docker pull để download Docker Image về từ Docker Hub

- Với dict-guided và vietocr (I)
    !docker pull 20011911/bkaichallenge_detecter_abcnet_in_dict-guided 
- Với vietocr: (II)
    !docker pull 20011911/bkaichallenge_detecter_yolor

============ REPRODUCE -  KIỂM THỬ KẾT QUẢ ===========



==> Phần I: Detecter
    (**) Với ABCNet
    - Chuẩn bị Docker Image:
        !docker pull 20011911/bkaichallenge_detecter_abcnet_in_dict-guided 
    - Khởi tạo và login vào container bằng lệnh:
        !docker run -it -v ./scene_text_pipeline:/scene_text_pipeline -w /scene_text_pipeline  --gpus "all" 20011911/bkaichallenge_detecter_abcnet_in_dict-guided /bin/bash
    - Thực hiện lệnh:
        !sh run_1.sh

    (**) Với YOLOR:
    - Chuẩn bị Docker Image:
        !docker pull 20011911/bkaichallenge_detecter_yolor-guided 
    - Khởi tạo và login vào container bằng lệnh:
        !docker run -it -v ./scene_text_pipeline:/scene_text_pipeline -w /scene_text_pipeline  --gpus "all" 20011911/bkaichallenge_detecter_yolor-guided /bin/bash
    - Thực hiện lệnh:
        !sh run_2.sh

==> Phần II: Recognize và merge file

    - Chuẩn bị Docker Image
        !docker pull 20011911/bkaichallenge_detecter_abcnet_in_dict 
    - Khởi tạo và login vào container bằng lệnh:
        !docker run -it -v ./scene_text_pipeline:/scene_text_pipeline -w /scene_text_pipeline  --gpus "all" 20011911/bkaichallenge_detecter_abcnet_in_dict  /bin/bash
    -  Thực hiện lệnh:
        !sh run_3.sh

==> Phần III: Zip file
    - Fule submission lưu tại:
        /scene_text_pipeline/Results/submission/prediction.zip [DONE]

============ HUẤN LUYỆN LẠI TỪ ĐẦU ===========


==> Phần I: Huấn luyện Detecter

    (**) Huấn luyện ABCNet
    *Môi trường
    - Chuẩn bị Docker Image:
        !docker pull 20011911/bkaichallenge_detecter_abcnet_in_dict-guided 
    - Khởi tạo và login vào container bằng lệnh:
        !docker run -it -v ./scene_text_pipeline:/scene_text_pipeline -w /scene_text_pipeline  --gpus "all" 20011911/bkaichallenge_detecter_abcnet_in_dict-guided /bin/bash
    
    *Chuẩn bị dữ liệu:
    - Chỉ sử dụng 2500 ảnh dữ liệu do BTC cung cấp, lưu tại:
        ./Data/trainset
    - Thực hiện training tại:
        !cd /scene_text_pipeline/Detecter/dict_guided
        !python3 train.py
    *Huấn luyện
    - Thực hiện file train.sh
        !sh train.sh
    - Final weight của đội thi được train sau 200K iter

    (**) Huấn luyện YOLOR
    - Chuẩn bị Docker Image:
        !docker pull 20011911/bkaichallenge_detecter_yolor-guided 
    - Khởi tạo và login vào container bằng lệnh:
        !docker run -it -v ./scene_text_pipeline:/scene_text_pipeline -w /scene_text_pipeline  --gpus "all" 20011911/bkaichallenge_detecter_yolor-guided /bin/bash
    - Thực hiện training tại:
        !cd /scene_text_pipeline/Detecter/yolor
        !sh train.sh
        
==> Phần II: Huấn luyện Recognize
    *Môi trường
    - Chuẩn bị Docker Image:
        !docker pull 20011911/bkaichallenge_detecter_abcnet_in_dict-guided 
    - Khởi tạo và login vào container bằng lệnh:
        !docker run -it -v ./scene_text_pipeline:/scene_text_pipeline -w /scene_text_pipeline  --gpus "all" 20011911/bkaichallenge_recognize_vietocr /bin/bash
    
    *Chuẩn bị dữ liệu:
    - Sử dụng 2500 ảnh dữ liệu do BTC cung cấp, lưu tại:
        ./Data/public_test_img
    - Cắt ảnh:
        ./Data/script/crop.py
    - Áp dụng tăng cường dữ liệu thêm 40K case image crop
        ./Pre-processing/straug/augment_data.py

    *Huấn luyện
    - Thực hiện file train.py
        !python3 train.py
    - Final weight của đội thi được train sau 50K iter

