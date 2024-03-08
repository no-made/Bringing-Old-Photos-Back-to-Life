# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import importlib
import sys
import argparse
import os
# from Face_Enhancement.util.util import util
import torch
# from Face_Enhancement.models.__init__ import get_option_setter
# import Face_Enhancement.models.__init__
# import data
import pickle

class BaseOptions:
    def __init__(self):
        # self.initialized = False
        self.name = "label2coco"  # name of the experiment. It decides where to store samples and models
        self.gpu_ids = 0
        self.checkpoints_dir = "./checkpoints"  # models are saved here
        self.outputs_dir = "./outputs"  # models are saved here
        self.model = "pix2pix"  # selects model to use for netG (pix2pix | spade)
        self.norm_G = "spectralinstance"  # instance normalization or batch normalization
        self.norm_D = "spectralinstance"  # instance normalization or batch normalization
        self.norm_E = "spectralinstance"  # instance normalization or batch normalization
        self.phase = "train"  # train, val, test, etc
        self.batchSize = 1  # input batch size
        self.preprocess_mode = "scale_width_and_crop"  # scaling and cropping of images at load time.{resize_and_crop, crop, scale_width, scale_width_and_crop, scale_shortside, scale_shortside_and_crop, fixed, none, resize}
        self.load_size = 1024  # scale images to this size. The final image will be cropped to --crop_size.
        self.crop_size = 512  # Crop to the width of crop_size (after initially scaling the images to load_size.)
        self.aspect_ratio = 1.0
        self.label_nc = 182  # number of input label channels
        self.contain_dontcare_label = False  # if the label map contains dontcare label (dontcare=255)
        self.output_nc = 3  # number of output image channels
        # for setting inputs
        self.dataroot = "./datasets/cityscapes/"
        self.dataset_mode = "coco"
        self.serial_batches = True  # if true, takes images in order to make batches, otherwise takes them randomly
        self.no_flip = True  # if specified, do not flip the images for data augmentation
        self.nThreads = 0  # number of threads for data loading
        self.max_dataset_size = sys.maxsize  # Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.
        self.load_from_opt_file = False  # load the options from a saved .opt file
        self.cache_filelist_write = False  # saves the current filelist into a text file, so that it loads faster
        self.cache_filelist_read = False  # reads from the file list cache
        self.display_winsize = 400  # display window size for both visdom and HTML
        self.netG = "spade"  # selects model to use for netG (pix2pixhd | spade)
        self.ngf = 64  # # of gen filters in first conv layer
        self.init_type = "xavier"  # network initialization [normal|xavier|kaiming|orthogonal]
        self.init_variance = 0.02  # variance of the initialization distribution
        self.z_dim = 256  # dimension of the latent z vector
        self.no_parsing_map = False  # During training, we do not use the parsing map
        # for instance-wise features
        self.no_instance = False  # if specified, don't add instance map as input
        self.semantic_nc = self.label_nc + (1 if self.contain_dontcare_label else 0) + (0 if self.no_instance else 1)
        self.nef = 16  # # of encoder filters in the first conv layer
        self.use_vae = False  # enable training with an image encoder
        self.tensorboard_log = False  # use tensorboard to record the results
        self.old_face_folder = ""  # the folder to load the old face images
        self.old_face_label_folder = ""  # the folder to load the old face labels
        self.injection_layer = "all"
        self.isTrain = False
    @staticmethod
    def modify_options(opt, is_train):
        # Implement the modification of options as needed
        modified_options = {'modified_option': opt}  # Replace with your modifications
        return modified_options
    def gather_options(self):
        # Modify model-related options
        model_name = self.model
        model_option_setter = get_option_setter(model_name)
        for key, value in model_option_setter(self, self.isTrain).items():
            setattr(self, key, value)

        # ... (remove the lines related to dataset options)

        # Set semantic_nc based on the option.
        # self.semantic_nc = (
        #     self.label_nc + (1 if self.contain_dontcare_label else 0) + (0 if self.no_instance else 1)
        # )

        # Set GPU ids
        # str_ids = self.gpu_ids.split(",")
        # self.gpu_ids = [int(str_id) for str_id in str_ids if int(str_id) >= 0]

        if self.gpu_ids:
            print("The main GPU is ")
            print(self.gpu_ids[0])
            torch.cuda.set_device(self.gpu_ids[0])

        assert (
            not self.gpu_ids or self.batchSize % len(self.gpu_ids) == 0
        ), "Batch size %d is wrong. It must be a multiple of # GPUs %d." % (self.batchSize, len(self.gpu_ids))

        # self.initialized = True
        return self

    def print_options(self):
        message = ""
        message += "----------------- Options ---------------\n"
        for k, v in sorted(self.__dict__.items()):
            comment = ""
            message += "{:>25}: {:<30}{}\n".format(str(k), str(v), comment)
        message += "----------------- End -------------------"
        # print(message)

    def option_file_path(self, makedir=False):
        expr_dir = os.path.join(self.checkpoints_dir, self.name)
        if makedir:
            mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, "opt")
        return file_name

    def save_options(self):
        file_name = self.option_file_path(makedir=True)
        with open(file_name + ".txt", "wt") as opt_file:
            for k, v in sorted(self.__dict__.items()):
                comment = ""
                opt_file.write("{:>25}: {:<30}{}\n".format(str(k), str(v), comment))

        with open(file_name + ".pkl", "wb") as opt_file:
            pickle.dump(self, opt_file)

    def update_options_from_file(self):
        new_opt = self.load_options()
        for k, v in sorted(new_opt.__dict__.items()):
            setattr(self, k, v)

    def load_options(self):
        file_name = self.option_file_path(makedir=False)
        new_opt = pickle.load(open(file_name + ".pkl", "rb"))
        return new_opt

    def parse(self, save=False):
        opt = self.gather_options()
        opt.isTrain = self.isTrain  # train or test
        opt.contain_dontcare_label = False

        self.print_options(opt)
        if opt.isTrain:
            self.save_options(opt)

        # Set semantic_nc based on the option.
        # This will be convenient in many places
        opt.semantic_nc = (
            opt.label_nc + (1 if opt.contain_dontcare_label else 0) + (0 if opt.no_instance else 1)
        )
        self.opt = opt
        return self.opt

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def find_model_using_name(model_name):
    # Given the option --model [modelname],
    # the file "models/modelname_model.py"
    # will be imported.
    model_name += "_model"
    model_filename = "." + model_name
    modellib = importlib.import_module(model_filename, package='models')

    # In the file, the class called ModelNameModel() will
    # be instantiated. It has to be a subclass of torch.nn.Module,
    # and it is case-insensitive.
    model = None
    target_model_name = model_name.replace("_", "").lower()
    print('target_model_name:', target_model_name)
    for name, cls in modellib.__dict__.items():
        print('name:', name)
        if name.lower() == target_model_name and issubclass(cls, torch.nn.Module):
            model = cls

    if model is None:
        print(
            f"In {model_filename}.py, there should be a subclass of torch.nn.Module with class name that matches {target_model_name} in lowercase."
        )
        exit(0)

    return model


def get_option_setter(model_name):
    model_class = find_model_using_name(model_name)
    return model_class.modify_options


def create_model(opt):
    model = find_model_using_name(opt.model)
    instance = model(opt)
    print("model [%s] was created" % (type(instance).__name__))

    return instance