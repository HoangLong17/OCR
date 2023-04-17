import argparse
from PIL import Image
import time
from glob import glob

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from vietocr.loader.s3_download import S3StorageService, ConfigS3, b64_2_img
from config import EvalConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgs', type=str, required=True, help="Images' folder or image 's path")
    
    args = parser.parse_args()
    base_config = Cfg.load_config_from_file("config/base.yml")
    update_config = Cfg.load_config_from_file(EvalConfig.MODEL_CONFIG)
    base_config.update(update_config)
    config = Cfg(base_config)

    config['device'] = EvalConfig.DEVICE
    config['weights'] = EvalConfig.PRETRAINED

    detector = Predictor(config)

    imgs_path = args.imgs
    if imgs_path.endswith(".jpg") or imgs_path.endswith(".png"):
        imgs = [Image.open(imgs_path)]
    else:
        imgs = [Image.open(img) for img in glob(args.imgs+"/*")]
    c_time = time.time()
    sents = detector.predict(imgs)
    print(time.time()-c_time)

    for s in sents:
        print(s)

if __name__ == '__main__':
    main()
