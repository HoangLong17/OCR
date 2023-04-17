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
import copy
from collections import defaultdict

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
        for s in sents:
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
        results = defaultdict(list)
        scores = defaultdict(list)
        order = []
        
        while max_length <= max_seq_length:
            have_eos = np.any(np.asarray(
                translated_sentence).T == eos_token, axis=1)
            copied_order = copy.deepcopy(order)
            # get orders of items which have eos_token
            for i, value in enumerate(have_eos):
                if value:
                    new_indice = i
                    for indice in sorted(copied_order):
                        if new_indice >= indice:
                            new_indice += 1
                    order.append(new_indice)

            # get sentences and probs of items which have eos_token
            for sent in np.array(translated_sentence).T[np.array(have_eos)]:
                results[sent.__len__()].append(list(sent))

            for prob in np.array(char_probs).T[np.array(have_eos)]:
                scores[prob.__len__()].append(list(prob))

            # if all items have eos_token then break
            if all(have_eos):
                break

            tgt_inp = translated_sentence
            tgt_inp = np.array(tgt_inp, dtype=np.int64)
            tgt_inp = tgt_inp.transpose(1, 0)

            # remove items which have eos_token
            tgt_inp = np.array(tgt_inp, dtype=np.int64)[~np.array(have_eos)]
            memory = np.array(memory, dtype=np.float32)[~np.array(have_eos)]
            char_probs = list(
                np.array(char_probs, dtype=np.float32).T[~np.array(have_eos)].T)

            decoder_input = {self.decoder_session.get_inputs()[0].name: tgt_inp,
                    self.decoder_session.get_inputs()[1].name: memory}

            output = self.decoder_session.run(None, decoder_input)[0]
            output = torch.Tensor(output)
            output = softmax(output, dim=-1)
            values, indices = torch.topk(output, 5)

            indices = indices[:, -1, 0]
            indices = indices.tolist()

            values = values[:, -1, 0]
            values = values.tolist()
            char_probs.append(values)

            translated_sentence = list(tgt_inp.T)
            translated_sentence.append(indices)
            max_length += 1

            del output

        sents = []
        probs = []
        # decoding sentences and calculating probs
        for key in results:
            sent_indices = np.array(results[key])

            prob = np.array(scores[key])
            prob = np.multiply(prob, sent_indices > 3)
            np.seterr(divide='ignore', invalid='ignore')
            prob = np.sum(prob, axis=-1)/(prob > 0).sum(-1)

            sents.extend(sent_indices.tolist())
            probs.extend(prob)

        # sort sents and probs in the right order
        sents = [x[0] for x in sorted(zip(sents, order), key=lambda x:x[1])]
        probs = [x[0] for x in sorted(zip(probs, order), key=lambda x:x[1])]


        return sents, probs


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
