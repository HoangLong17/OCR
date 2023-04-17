from vietocr.tool.config import Cfg
from vietocr.model.trainer import Trainer  
from config import EvalConfig

def load_config():
    base_config = Cfg.load_config_from_file("config/base.yml")
    update_config = Cfg.load_config_from_file(EvalConfig.MODEL_CONFIG)
    base_config.update(update_config)
    config =  Cfg(base_config)

    dataset_params = {
        'name':'hw',
        'data_root':EvalConfig.DATA_ROOT,
        'train_annotation': EvalConfig.TRAIN_ANNOTATION,
        'valid_annotation': EvalConfig.TEST_ANNOTATION,
        'train_label_studio_annotation': EvalConfig.TRAIN_LABEL_STUDIO_ANNOTATION,
        'valid_label_studio_annotation': EvalConfig.TEST_LABEL_STUDIO_ANNOTAION
    }

    params = {
            'batch_size': EvalConfig.BATCH_SIZE,
            'print_every': EvalConfig.PRINT_EVERY,
            'valid_every': EvalConfig.VALID_EVERY,
            'iters': EvalConfig.ITERS,
            'checkpoint': EvalConfig.CHECKPOINT,    
            'export': EvalConfig.EXPORT,
            'metrics': EvalConfig.METRICS
            }

    config['trainer'].update(params)
    config['dataset'].update(dataset_params)
    config['device'] = EvalConfig.DEVICE
    config['aug']['image_aug'] = EvalConfig.AUGMENT
    config['weights'] = EvalConfig.PRETRAINED
    config['check_field'] = EvalConfig.CHECK_FIELD

    return config

def main():
    config = load_config()
    trainer = Trainer(config, pretrained=True, mode='val')
    acc_full_seq, acc_per_char = trainer.precision()
    print("acc full seq: {:.4f} - acc per char: {:.4f}".format(acc_full_seq, acc_per_char))

if __name__ == "__main__":
    main()
