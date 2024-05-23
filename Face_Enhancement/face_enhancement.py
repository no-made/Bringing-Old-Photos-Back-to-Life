# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import sys
from collections import OrderedDict

from .data.__init__ import create_dataloader
from .options.test_options import TestOptions
from .models.pix2pix_model import Pix2PixModel
from .util.visualizer import Visualizer
import torchvision.utils as vutils
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class Options:
    def __init__(self, **kwargs):
        self.how_many = None
        self.contain_dontcare_label = None
        for key, value in kwargs.items():
            setattr(self, key, value)


opt = Options(
    name="label2coco",
    gpu_ids=0,
    checkpoints_dir="./checkpoints",
    outputs_dir="./outputs",
    model="pix2pix",
    norm_G="spectralinstance",
    norm_D="spectralinstance",
    norm_E="spectralinstance",
    phase="test",
    batchSize=1,
    preprocess_mode="scale_width_and_crop",
    load_size=1024,
    crop_size=512,
    aspect_ratio=1.0,
    label_nc=182,
    contain_dontcare_label=False,
    output_nc=3,
    dataroot="./datasets/cityscapes/",
    dataset_mode="coco",
    serial_batches=True,
    no_flip=True,
    nThreads=0,
    max_dataset_size=float("inf"),
    load_from_opt_file=False,
    cache_filelist_write=False,
    cache_filelist_read=False,
    display_winsize=256,
    netG="spade",
    ngf=64,
    init_type="xavier",
    init_variance=0.02,
    z_dim=256,
    no_parsing_map=False,
    no_instance=False,
    semantic_nc=183,
    nef=16,
    use_vae=False,
    tensorboard_log=False,
    old_face_folder="",
    old_face_label_folder="",
    injection_layer="all",
    isTrain=False,
    how_many=float("inf"),
    which_epoch="latest",
    num_upsampling_layers="normal",
    no_pairing_check=True

)


def enhance_face(result_dir, old_face_folder, old_face_label_folder, name, gpu_ids):
    opt.results_dir = result_dir
    opt.old_face_folder = old_face_folder
    opt.old_face_label_folder = old_face_label_folder
    opt.name = name
    opt.gpu_ids = gpu_ids
    opt.dataroot = "./"
    opt.preprocess_mode = 'resize'
    opt.tensorboard_log = True
    opt.load_size = 256
    # opt.crop_size = 256
    opt.batchSize = 4
    opt.label_nc = 182
    opt.no_instance = True
    opt.no_parsing_map = True
    opt.semantic_nc = opt.label_nc + (1 if opt.contain_dontcare_label else 0) + (0 if opt.no_instance else 1)
    dataloader = create_dataloader(opt)
    print('dataloader', dataloader.dataset)
    model = Pix2PixModel(opt)
    model.eval()
    # visualizer = Visualizer(opt)

    # single_save_url = os.path.join(opt.checkpoints_dir, opt.name, opt.results_dir, "each_img")
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
