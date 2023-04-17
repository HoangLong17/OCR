from vietocr.tool.config import Cfg
from vietocr.model.trainer import Trainer  
from config import TrainConfig

def load_config():
    base_config = Cfg.load_config_from_file("config/base.yml")
    update_config = Cfg.load_config_from_file(TrainConfig.MODEL_CONFIG)
    base_config.update(update_config)
    config =  Cfg(base_config)

    dataset_params = {
        'name':'hw',
        'data_root': TrainConfig.DATA_ROOT,
        'train_annotation': TrainConfig.TRAIN_ANNOTATION,
        'valid_annotation': TrainConfig.VALID_ANNOTATION,
        'train_label_studio_annotation': TrainConfig.TRAIN_LABEL_STUDIO_ANNOTATION,
        'valid_label_studio_annotation': TrainConfig.VALID_LABEL_STUDIO_ANNOTAION
    }

    params = {
            'batch_size': TrainConfig.BATCH_SIZE,
            'print_every': TrainConfig.PRINT_EVERY,
            'valid_every': TrainConfig.VALID_EVERY,
            'iters': TrainConfig.ITERS,
            'checkpoint': TrainConfig.CHECKPOINT,    
            'export': TrainConfig.EXPORT,
            'metrics': TrainConfig.METRICS
            }

    config['trainer'].update(params)
    config['dataset'].update(dataset_params)
    config['device'] = TrainConfig.DEVICE
    config['aug']['image_aug'] = TrainConfig.AUGMENT
    config['weights'] = TrainConfig.PRETRAINED
    config['check_field'] = TrainConfig.CHECK_FIELD


    return config

def main():
    config = load_config()
    trainer = Trainer(config, pretrained=False, mode='train')
    trainer.config.save('config.yml')
    trainer.train()
    acc_full_seq, acc_per_char = trainer.precision()
    print("acc full seq: {:.4f} - acc per char: {:.4f}".format(acc_full_seq, acc_per_char))

if __name__ == "__main__":
    main()