from vietocr.model.seqmodel.finetuned_transformer import FinetunedTransformer
from torch import nn

class VietOCR(FinetunedTransformer):
    def __init__(self, vocab_size,
                 backbone,
                 cnn_args, 
                 transformer_args, seq_modeling='transformer'):
        
        super(VietOCR, self).__init__(vocab_size, backbone, cnn_args, **transformer_args)

    def forward(self, img, tgt_input, tgt_key_padding_mask=None):
        """
        Shape:
            - img: (N, C, H, W)
            - tgt_input: (T, N)
            - tgt_key_padding_mask: (N, T)
            - output: b t v
        """
        tgt_input = tgt_input.transpose(0,1)
        memory = self.encoder(img)
        outputs = self.decoder(tgt_input, memory, tgt_key_padding_mask)

        return outputs
