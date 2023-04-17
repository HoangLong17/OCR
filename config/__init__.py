class TrainConfig:
    MODEL_CONFIG = "config/vgg-transformer.yml"
    BATCH_SIZE = 64
    PRINT_EVERY = 200
    VALID_EVERY = 10000
    ITERS =  100000
    METRICS = 10000
    EXPORT = "weights/weight.pt"
    
    DATA_ROOT = "dataset/OCR"
    TRAIN_ANNOTATION = "train_line_annotation.txt"
    VALID_ANNOTATION = "test_line_annotation.txt"
    TRAIN_LABEL_STUDIO_ANNOTATION = "train_label_path.txt"
    VALID_LABEL_STUDIO_ANNOTAION = "test_label_path.txt"
    
    CHECKPOINT = "./checkpoint/checkpoint.pth"
    DEVICE = "cuda:0"
    AUGMENT = True
    PRETRAINED = "weights/new_v1.pth"
    CHECK_FIELD = None
    
    
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
    
    
class OnnxConfig:
    MODEL_CONFIG = "config/vgg-transformer.yml"
    DEVICE = "cuda:0"
    PT_WEIGHT = "weights/new_v1.pth"
    
    ENCODER_WEIGHT = "weights/encoder.onnx"
    DECODER_WEIGHT = "weights/decoder.onnx"
    