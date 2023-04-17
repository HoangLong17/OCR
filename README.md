# Cài Đặt
Download pretrained weight của [VGG 19](https://download.pytorch.org/models/vgg19_bn-c79401a0.pth).

Download file .whl của [opencv-python==4.5.5.64](https://files.pythonhosted.org/packages/67/50/665a503167396ad347957bea0bd8d5c08c865030b2d1565ff06eba613780/opencv_python-4.5.5.64-cp36-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl).

Sau đó, chạy lệnh sau.
```
sh setup.sh
```

# Train và Evaluate
Chỉnh sửa config, vào file <mark>config\/\_\_init\_\_.py</mark>.
```
class TrainConfig:
    MODEL_CONFIG = "config/vgg-transformer.yml"     # config của model
    BATCH_SIZE = 64                                 
    PRINT_EVERY = 200                             
    VALID_EVERY = 10000
    ITERS =  100000                                   # số lượng epochs
    METRICS = 10000                                 # số lượng max samples dùng để validate
    EXPORT = "weights/weight.pt"                    # đường dẫn để lưu weight sau khi train model
    
    DATA_ROOT = "dataset/OCR"                       # thư mục gốc lưu data trên s3
    TRAIN_ANNOTATION = "train_line_annotation.txt"
    VALID_ANNOTATION = "test_line_annotation.txt"
    TRAIN_LABEL_STUDIO_ANNOTATION = "train_label_path.txt"
    VALID_LABEL_STUDIO_ANNOTAION = "test_label_path.txt"
    
    CHECKPOINT = "./checkpoint/checkpoint.pth"      
    DEVICE = "cuda:0"
    AUGMENT = True
    PRETRAINED = "weights/new_v1.pth"               # đường dẫn của pretrained weight
    CHECK_FIELD = None                              # trường dữ liệu được validate (dùng khi evaluate model)
    
    
class EvalConfig:
    MODEL_CONFIG = "config/vgg-transformer.yml"
    BATCH_SIZE = 8
    PRINT_EVERY = None
    VALID_EVERY = None
    ITERS =  None
    METRICS = 10000
    EXPORT = None
    
    DATA_ROOT = "dataset/OCR"
    TRAIN_ANNOTATION = None
    TEST_ANNOTATION = None
    TRAIN_LABEL_STUDIO_ANNOTATION = None
    TEST_LABEL_STUDIO_ANNOTAION = "test_label_path.txt"
    
    CHECKPOINT = None
    DEVICE = "cuda:0"
    AUGMENT = False
    PRETRAINED = "weights/new_v1.pth"
    CHECK_FIELD = None
    
```

TRAIN_ANNOTATION và VALID_ANNOTATION là các file.txt được đánh nhãn bằng tay có dạng:
```
Đường_dẫn_đến_file_ảnh<\t>label_của_ảnh<\n>
```

TRAIN_LABEL_STUDIO_ANNOTATION và VALID_LABEL_STUDIO_ANNOTAION là các file .txt được đánh nhãn bằng label studio có dạng:
```
Đường_dẫn_đến_file_label_của_labelstudio<\n>
```

Train model:
```
python train.py
```

Evaluate model:
```
python evaluate.py
```
Test model:
```
python predict.py --imgs Images' folder or image 's path
```

**Lưu ý:** Nên train model với batch size bằng 64 và iters bằng 100 000, nếu giảm batch size thì tăng số lượng iter tương ứng và ngược lại.

# Convert Onnx và Inference
Chỉnh sửa config, vào file <mark>config\/\_\_init\_\_.py</mark>.
```
class OnnxConfig:
    MODEL_CONFIG = "config/vgg-transformer.yml"    # config của model
    DEVICE = "cuda:0"
    PT_WEIGHT = "weights/new_v1.pth"               # đường dẫn đến pytorch weight của model
    
    ENCODER_WEIGHT = "weights/encoder.onnx"        # đường dẫn đến onnx weight của model encoder
    DECODER_WEIGHT = "weights/decoder.onnx"        # đường dẫn đến onnx weight của model decoder
```
Convert model sang onnx:
```
python convert2onnx.py
```
Inference onnx model:
```
python onnx_inference.py --imgs Images' folder or image 's path
```
Inference onnx model với hàm translate được tối ưu:
```
python onnx_inference_v2.py --imgs Images' folder or image 's path
```
