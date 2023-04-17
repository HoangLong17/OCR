from config import OnnxConfig
from vietocr.tool.config import Cfg
from vietocr.tool.translate import build_model
import torch
import onnxruntime
import numpy as np
import math
import onnx


def convert_encoder(model, img, save_path):
    with torch.no_grad():
        memory = model(img)
        torch.onnx.export(model,
            img,
            save_path,
            export_params=True,
            opset_version=14,
            verbose=True, 
            do_constant_folding=True,
            input_names=['images'],
            output_names=['memory'],
            dynamic_axes={'images':{0: "batch", 2: "height", 3: "width"},
                        'memory': {0: "batch", 1: "d_model_1", 2: "d_model_2"}})

    print("Convert encoder successfully")
    return memory

def convert_decoder(model, input, save_path):
    memory, tgt_input = input
    with torch.no_grad():
        torch.onnx.export(model,
            (tgt_input, memory),
            save_path,
            export_params=True,
            opset_version=14,
            verbose=True, 
            do_constant_folding=True,
            input_names=['tgt_inp', 'memory'],
            output_names=['outputs'],
            dynamic_axes={'tgt_inp':{0: "batch", 1: "length"},
                        'memory':{0: "batch", 1: "d_model_1", 2: "d_model_2"},
                        'outputs': {0: "batch", 1: "length"}})

    print("Convert decoder successfully")


if __name__ == "__main__":
    base_cfg = Cfg.load_config_from_file('config/base.yml')
    update_cfg = Cfg.load_config_from_file(OnnxConfig.MODEL_CONFIG)
    base_cfg.update(update_cfg)
    config = Cfg(base_cfg)

    config['device'] = OnnxConfig.DEVICE
    config['weights'] = OnnxConfig.PT_WEIGHT

    # build model
    model, vocab = build_model(config)

    # load weight
    model.load_state_dict(torch.load(config['weights'], map_location=torch.device(config['device'])))
    model = model.eval() 

    img = torch.rand(2, 3, config['dataset']['image_height'], config['dataset']['image_max_width']).to(config['device'])
    tgt_input = torch.randint(1,2, size=(2,1)).to(config['device'])
    
    memory = convert_encoder(model.encoder, img, OnnxConfig.ENCODER_WEIGHT)
    input = (memory, tgt_input)
    convert_decoder(model.decoder, input, OnnxConfig.DECODER_WEIGHT)