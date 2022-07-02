import wandb
from trainer import Trainer
from seg_pred import pred
from seg_eval import model_eval

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
        'depth_w':{
            'values': [1,0]
            #'values': [0]
        },
        'rgb_w':{
            'values': [1,0]
            #'values': [0]
        }
    }
}



def model_try(ncut_w):
    config_defaults = dict(
        #data_path = "/content/kitti_data",
        data_path = None,
        split = "scene_09",
        log_dir = "/content/drive/MyDrive/monoseg_models/",
        output_dir = "/content/drive/MyDrive/monoseg_models_repeat/",
        input_path = "/content/data_semantics/training/image_2",
        semantic_path = "/content/data_semantics",
        #semantic_path = None,
        gt_path = "/content/data_semantics/training/semantic",
        #distance_path = "/content/distance_gt_sp_l/distance_gt_sp/",
        distance_path = None,
        train_output = "/content/drive/MyDrive/work_dirs/",
        #train_output = None,

        #load_weights_folder = "/content/drive/MyDrive/monoseg_models/byol_new_small_lr/models/weights_149",
        #load_weights_folder_t = "/content/drive/MyDrive/mono+stereo_640x192/",
        load_weights_folder = None,

        kd = False,
        h_match = False,
        #share = True,
        chamfer = False,
        architecture = "byol",
        ResX = 18,
        step = ["linear_eval"],
        models_to_load = ["encoder","decoder"],

        log_frequency =  100,
        save_frequency = 100,
        model_name = "byol_random",
        cls_num = 19,
        dataset = "kitti_semantic",
        height = 192,
        width = 640,
        depth_scales = [0,1,2,3],
        seg_scales = [0],
        batch_size = 1,
        learning_rate = 1e-3,
        num_epochs = 300,
        scheduler_step_size = 100,
        num_workers = 4,

        angle_w = 1,
        rgb_w = 0,
        coordinate_w = 0,
        depth_w = 0,
        ncut_w = ncut_w
    )
    wandb.init(project="seg_byol", config=config_defaults)
    config = wandb.config

    trainer = Trainer(config)
        
    trainer.train()
        
    #seg_model = trainer.models["encoder"]
    #depth_decoder = trainer.models["depth"]
        
    #pred(trainer.models["seg_encoder"], trainer.models["seg"], trainer.models["depth_encoder"], trainer.models["depth"], config)
    metrics = model_eval(config, trainer.models)
    print(metrics)
    wandb.log(metrics)
    wandb.finish()
    #iou_per_cls = dict(zip(labels, iou_per_cls))

    #wandb.log({"avg_iou": avg_iou})
    #wandb.log(iou_per_cls)

    return 

model_try(0)
#sweep_id = wandb.sweep(sweep_config, project="MonoSeg")
#wandb.agent(sweep_id, model_try)