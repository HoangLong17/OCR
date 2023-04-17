import onnxruntime
import numpy as np
import torch
from torch.nn.functional import log_softmax, softmax
import math
from PIL import Image
import cv2
from glob import glob
import time
import argparse

from config import OnnxConfig
from vietocr.model.vocab import Vocab
from vietocr.tool.config import Cfg
from vietocr.tool.translate import process_input

class OcrOnnxInference:
    def __init__(self, config, sessions: list):
        self.encoder_session, self.decoder_session = sessions 
        self.config = config
        self.vocab = Vocab(self.config['vocab'])

    def predict(self, img):
        img = process_input(img, self.config['dataset']['image_height'], \
            self.config['dataset']['image_min_width'], self.config['dataset']['image_max_width'])   

        np_img = np.array(img, dtype=np.float32)
        sents, probs = self.translate_onnx(np_img)

        res = []
        for s in sents.tolist():
            r = self.vocab.decode(s)
            res.append(r)

        return res


    def translate_onnx(self, img, max_seq_length=128, sos_token=1, eos_token=2):
        img = np.array(img, dtype=np.float32)
        encoder_input = {self.encoder_session.get_inputs()[0].name: img}
        memory = self.encoder_session.run(None, encoder_input)[0]
        memory = np.array(memory)
        

        translated_sentence = [[sos_token]*img.shape[0]]
        char_probs = [[1]*img.shape[0]]

        max_length = 0

        while max_length <= max_seq_length and not all(np.any(np.asarray(translated_sentence).T==eos_token, axis=1)):
            tgt_inp = translated_sentence
            tgt_inp = np.array(tgt_inp, dtype=np.int64)
            tgt_inp = tgt_inp.transpose(1,0)
         

            decoder_input = {self.decoder_session.get_inputs()[0].name: tgt_inp,
                    self.decoder_session.get_inputs()[1].name: memory}

            output = self.decoder_session.run(None, decoder_input)[0]
            output = torch.Tensor(output)
            output = softmax(output, dim=-1)

            values, indices  = torch.topk(output, 5)
                
            indices = indices[:, -1, 0]
            indices = indices.tolist()
                
            values = values[:, -1, 0]
            values = values.tolist()
            char_probs.append(values)

            translated_sentence.append(indices)   
            max_length += 1

            del output

        translated_sentence = np.asarray(translated_sentence).T
            
        char_probs = np.asarray(char_probs).T
        char_probs = np.multiply(char_probs, translated_sentence>3)
        char_probs = np.sum(char_probs, axis=-1)/(char_probs>0).sum(-1)

        return translated_sentence, char_probs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgs', type=str, required=True, help="Images' folder or image 's path")
    args = parser.parse_args()
                    
    base_cfg = Cfg.load_config_from_file('config/base.yml')
    update_cfg = Cfg.load_config_from_file(OnnxConfig.MODEL_CONFIG)
    base_cfg.update(update_cfg)
    config = Cfg(base_cfg)

    config['device'] = OnnxConfig.DEVICE
    config['weights'] = OnnxConfig.PT_WEIGHT

    encoder = onnxruntime.InferenceSession(OnnxConfig.ENCODER_WEIGHT, providers=['CPUExecutionProvider', 'CUDAExecutionProvider'])
    decoder = onnxruntime.InferenceSession(OnnxConfig.DECODER_WEIGHT, providers=['CPUExecutionProvider', 'CUDAExecutionProvider'])
    predictor = OcrOnnxInference(config=config, sessions=[encoder, decoder])

    imgs_path = args.imgs
    if imgs_path.endswith(".jpg") or imgs_path.endswith(".png"):
        imgs = [Image.open(imgs_path)]
    else:
        imgs = [Image.open(img) for img in glob(args.imgs+"/*")]
    c_time = time.time()
    res = predictor.predict(imgs)
    print(time.time()-c_time)

    for r in res:
        print(r)