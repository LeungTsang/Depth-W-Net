# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

from trainer import Trainer

config_fix = dict(
    data_path = "/content/drive/MyDrive/kitti_data"
    split = "eigen_zhou"
    log_dir = "/content/monoseg/models"
    load_weights_folder = None
    log_frequency = 100
    save_frequency = 5
    model_name = "monoseg"
    cls_num = 19
    dataset = "kitti"
    height = 192
    width = 640
    scales = [0,1,2,3]
    batch_size = 1
    learning_rate = 1e-4
    num_epochs = 50
    scheduler_step_size = 15
    num_workers = 12
    depth_w = 0
    coordinate_w = 0
    rgb_w = 0
    probability_w = 0
)

sweep_config = {
    'method': 'random', #grid, random
    'metric': {
      'name': 'loss',
      'goal': 'maximize'   
    },
}




if __name__ == "__main__":
    trainer = Trainer(config)
    trainer.train()
