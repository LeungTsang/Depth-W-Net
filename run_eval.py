import wandb
from trainer import Trainer
from seg_eval import model_eval

labels = ['road','sidewalk','building','wall','fence','pole','traffic light','traffic sign','vegetation','terrain',
    'sky','person','rider','car','truck','bus','train','motorcycle','bicycle']



def eval_try(model_name):

    config_defaults = dict(
        output_dir = "/content/drive/MyDrive/monoseg_models_eval/",
        input_path = "/content/data_semantics/training/image_2",
        #input_path = "/content/drive/MyDrive/kitti_data/2011_09_26/2011_09_26_drive_0009_sync/image_02/data",
        gt_path = "/content/data_semantics/training/semantic",
        #gt_path = None,

        load_weights_folder = "/content/drive/MyDrive/monoseg_models/"+model_name+"/models/weights_299",
        #load_weights_folder = "/content/drive/MyDrive/monoseg_reconstruct_models/"+model_name+"/models/weights_99",
        architecture = "byol",
        ResX = 18,

        h_match = False,

        model_name = model_name,
        cls_num = 19,
        height = 192,
        width = 640,
        scales = [0,1,2,3],
        seg_scales = [0],

        feature_w = 1,
        depth_w = 1,
        rgb_w = 1,
        coordinate_w = 1
        
    )
    wandb.init(project="MonoSeg_test", config=config_defaults)
    config = wandb.config
    
    metrics = model_eval(config)
    print(metrics)
    wandb.log(metrics)
    
    wandb.finish()
    return 

#eval_try("eigen_zhou_double_UNet")
eval_try("byol_linear_eval_ri")