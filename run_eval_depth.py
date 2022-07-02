import wandb
from trainer import Trainer
from seg_eval import model_eval
from evaluate_depth import evaluate

labels = ['road','sidewalk','building','wall','fence','pole','traffic light','traffic sign','vegetation','terrain',
    'sky','person','rider','car','truck','bus','train','motorcycle','bicycle']



def eval_try(model_name):

    config_defaults = dict(
        eval_split = "scene_09",
        load_weights_folder = "/content/drive/MyDrive/monoseg_models_1/"+model_name+"/models/weights_99",
        data_path = "/content/drive/MyDrive/kitti_data",
        #eval_mono = True,
        #post_process = True,
        cls_num = 19,
        architecture = "UNet_ResNet",
        num_workers = 4
    )
    wandb.init(project="MonoSeg_test_depth", config=config_defaults)
    config = wandb.config
    
    metrics = evaluate(config)
    print(metrics)
    wandb.log(metrics)
    
    wandb.finish()
    return 

#eval_try("eigen_zhou_double_UNet")
eval_try("UNet_ResNet")