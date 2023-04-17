import torch
import numpy as np
import math
from PIL import Image
from collections import defaultdict
import copy
from torch.nn.functional import log_softmax, softmax

from vietocr.model.finetuned_transformerocr import VietOCR
from vietocr.model.vocab import Vocab
from vietocr.model.beam import Beam

def batch_translate_beam_search(img, model, beam_size=4, candidates=1, max_seq_length=128, sos_token=1, eos_token=2):
    # img: NxCxHxW
    model.eval()
    device = img.device
    sents = []

    with torch.no_grad():
        src = model.cnn(img)
        print(src.shap)
        memories = model.transformer.forward_encoder(src)
        for i in range(src.size(0)):
#            memory = memories[:,i,:].repeat(1, beam_size, 1) # TxNxE
            memory = model.transformer.get_memory(memories, i)
            sent = beamsearch(memory, model, device, beam_size, candidates, max_seq_length, sos_token, eos_token)
            sents.append(sent)

    sents = np.asarray(sents)

    return sents
   
def translate_beam_search(img, model, beam_size=4, candidates=1, max_seq_length=128, sos_token=1, eos_token=2):
    # img: 1xCxHxW
    model.eval()
    device = img.device

    with torch.no_grad():
        src = model.cnn(img)
        memory = model.transformer.forward_encoder(src) #TxNxE
        sent = beamsearch(memory, model, device, beam_size, candidates, max_seq_length, sos_token, eos_token)

    return sent
        
def beamsearch(memory, model, device, beam_size=4, candidates=1, max_seq_length=128, sos_token=1, eos_token=2):    
    # memory: Tx1xE
    model.eval()

    beam = Beam(beam_size=beam_size, min_length=0, n_top=candidates, ranker=None, start_token_id=sos_token, end_token_id=eos_token)

    with torch.no_grad():
#        memory = memory.repeat(1, beam_size, 1) # TxNxE
        memory = model.transformer.expand_memory(memory, beam_size)

        for _ in range(max_seq_length):
            
            tgt_inp = beam.get_current_state().transpose(0,1).to(device)  # TxN
            decoder_outputs, memory = model.transformer.forward_decoder(tgt_inp, memory)

            log_prob = log_softmax(decoder_outputs[:,-1, :].squeeze(0), dim=-1)
            beam.advance(log_prob.cpu())
            
            if beam.done():
                break
                
        scores, ks = beam.sort_finished(minimum=1)

        hypothesises = []
        for i, (times, k) in enumerate(ks[:candidates]):
            hypothesis = beam.get_hypothesis(times, k)
            hypothesises.append(hypothesis)
    
    return [1] + [int(i) for i in hypothesises[0][:-1]]

def translate(img, model, max_seq_length=128, sos_token=1, eos_token=2):
    "data: BxCXHxW"
    model.eval()
    device = img.device

    with torch.no_grad():
        src = model.cnn(img)
        memory = model.transformer.forward_encoder(src)

        translated_sentence = [[sos_token]*len(img)]
        char_probs = [[1]*len(img)]

        max_length = 0

        while max_length <= max_seq_length and not all(np.any(np.asarray(translated_sentence).T==eos_token, axis=1)):
            tgt_inp = torch.LongTensor(translated_sentence).to(device)
            output, memory = model.transformer.forward_decoder(tgt_inp, memory)
                
            output = softmax(output, dim=-1)
            output = output.to('cpu')

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

def translate_v2(img, model, max_seq_length=128, sos_token=1, eos_token=2):
    "data: BxCXHxW"
    model.eval()
    device = img.device

    with torch.no_grad():
        memory = model.encoder(img)

        translated_sentence = [[sos_token]*len(img)]
        char_probs = [[1]*len(img)]

        max_length = 0

        while max_length <= max_seq_length and not all(np.any(np.asarray(translated_sentence).T==eos_token, axis=1)):
            tgt_inp = torch.LongTensor(translated_sentence).to(device)
            tgt_inp = tgt_inp.transpose(0, 1)
            output = model.decoder(tgt_inp, memory)
                
            output = softmax(output, dim=-1)
            output = output.to('cpu')

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


def build_model(config):
    vocab = Vocab(config['vocab'])
    device = config['device']
    
    model = VietOCR(len(vocab),
            config['backbone'],
            config['cnn'], 
            config['transformer'],
            config['seq_modeling'])
    
    model = model.to(device)

    return model, vocab

def resize(w, h, expected_height, image_min_width, image_max_width):
    new_w = int(expected_height * float(w) / float(h))
    round_to = 10
    new_w = math.ceil(new_w/round_to)*round_to
    new_w = max(new_w, image_min_width)
    new_w = min(new_w, image_max_width)

    return new_w, expected_height

def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

def process_image(image, image_height, image_min_width, image_max_width):
    img = image.convert('RGB')

    w, h = img.size
    new_w, image_height = resize(w, h, image_height, image_min_width, image_max_width)

    img = img.resize((new_w, image_height), Image.ANTIALIAS)
    if (image_max_width-new_w)%2 == 0:
        right = int((image_max_width-new_w)/2)
        left = int((image_max_width-new_w)/2)
    else:
        right = int((image_max_width-new_w)/2)
        left = int((image_max_width-new_w)/2) + 1
    img = add_margin(img, 0, left, 0, right, (255, 255, 255))
    # img.save("../image/test.png")

    img = np.asarray(img).transpose(2,0, 1)
    img = img/255
    return img

def process_input(images, image_height, image_min_width, image_max_width):
    imgs = []
    for image in images:
        img = process_image(image, image_height, image_min_width, image_max_width)
        imgs.append(img)
#     img = img[np.newaxis, ...]
    imgs = torch.FloatTensor(np.array(imgs))
    return imgs

def predict(filename, config):
    img = Image.open(filename)
    img = process_input(img)

    img = img.to(config['device'])

    model, vocab = build_model(config)
    s = translate(img, model)[0].tolist()
    s = vocab.decode(s)
    
    return s

