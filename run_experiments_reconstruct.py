import wandb
from trainer_reconstruct import Trainer
from seg_pred_reconstruct import pred

labels = ['road','sidewalk','building','wall','fence','pole','traffic light','traffic sign','vegetation','terrain',
    'sky','person','rider','car','truck','bus','train','motorcycle','bicycle']

sweep_config = {
    'method': 'grid', #grid, random
    'metric': {
        'name': 'loss',
        'goal': 'maximize'   
    },
    'parameters': {
        'coordinate_w':{
            'values': [0]
            #'values': [0]
        },
        'rgb_w':{
            'values': [1,0]
            #'values': [0]
        }
    }
}



def model_try(nw,dw,cw,rw):
    config_defaults = dict(
        data_path = "/content/drive/MyDrive/kitti_data",
        split = "scene_09",
        log_dir = "/content/drive/MyDrive/monoseg_reconstruct_models/",
        output_dir = "/content/drive/MyDrive/monoseg_models_reconstruct/",
        input_path = "/content/drive/MyDrive/kitti_data/2011_09_26/2011_09_26_drive_0009_sync/image_02/data",
        gt_path = "/content/data_semantics/training/semantic",

        load_weights_folder = None,
        log_frequency = 444,
        save_frequency = 100,
        model_name = "scene_09",
        cls_num = 19,
        dataset = "kitti",
        height = 192,
        width = 640,
        scales = [0,1,2,3],
        batch_size = 1,
        learning_rate = 1e-4,
        num_epochs = 100,
        scheduler_step_size = 40,
        num_workers = 4,
        ncut_w = nw,
        depth_w = dw,
        coordinate_w = cw,
        rgb_w = rw,
        feature_w = 0
    )
    wandb.init(project="MonoSeg_reconstruct", config=config_defaults)
    config = wandb.config

    trainer = Trainer(config)
        
    trainer.train()
        
    seg_model = trainer.models["encoder"]
    reconstruct_decoder = trainer.models["reconstruct"]
        
    pred(seg_model, reconstruct_decoder, config)
    #iou_per_cls = dict(zip(labels, iou_per_cls))

    #wandb.log({"avg_iou": avg_iou})
    #wandb.log(iou_per_cls)

    return 

model_try(0.1,0,0,1)
#sweep_id = wandb.sweep(sweep_config, project="MonoSeg")
#wandb.agent(sweep_id, model_try)