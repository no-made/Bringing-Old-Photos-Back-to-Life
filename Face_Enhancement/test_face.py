# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from collections import OrderedDict

import torch

import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
import torchvision.utils as vutils
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def get_gpu():
    gpu = -1
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print('Number of GPUs available:', num_gpus)
        for i in range(num_gpus):
            cuda_properties = torch.cuda.get_device_properties(i)
            gpu_memory = cuda_properties.total_memory / 1024 ** 3  # Convert bytes to gigabytes

            if gpu_memory > 4:
                gpu = i
                print('GPU with more than 4 GB of RAM:', i, torch.cuda.mem_get_info(i))
                break
        # torch.cuda.memory_allocated()
        # torch.cuda.max_memory_allocated()
    if gpu == -1:
        print('No suitable GPU found. Setting gpu = -1')
    else:
        print('Selected GPU:', gpu, torch.cuda.get_device_name(gpu), torch.cuda.mem_get_info(gpu))
    return gpu


opt = TestOptions().parse()  # get test options
# opt.results_dir = result_dir
# opt.old_face_folder = old_face_folder
# opt.old_face_label_folder = old_face_label_folder
opt.name = "Setting_9_epoch_100"
opt.gpu_ids = get_gpu()
opt.dataroot = "./"
opt.preprocess_mode = 'resize'
opt.tensorboard_log = True
opt.load_size = 512
# opt.crop_size = 256
opt.batchSize = 4
opt.label_nc = 18
opt.no_instance = True
opt.no_parsing_map = True
dataloader = data.create_dataloader(opt)

model = Pix2PixModel(opt)
model.eval()

visualizer = Visualizer(opt)


#single_save_url = os.path.join(opt.checkpoints_dir, opt.name, opt.results_dir, "each_img")
single_save_url = os.path.join(opt.results_dir, "each_img")


if not os.path.exists(single_save_url):
    os.makedirs(single_save_url)


for i, data_i in enumerate(dataloader):
    print("Processing %d" % i)
    if i * opt.batchSize >= opt.how_many:
        break

    generated = model(data_i, mode="inference")

    img_path = data_i["path"]

    for b in range(generated.shape[0]):
        img_name = os.path.split(img_path[b])[-1]
        save_img_url = os.path.join(single_save_url, img_name)
        print("Processing image %s" % save_img_url)

        vutils.save_image((generated[b] + 1) / 2, save_img_url)

