# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import os
from ..util import util
import torch

class BaseOptions:
    def __init__(self):
        # experiment specifics
        self.name = "label2city"  # name of the experiment. It decides where to store samples and models
        self.gpu_ids = "0"
        self.checkpoints_dir = "./checkpoints"  # models are saved here
        self.outputs_dir = "./outputs"  # models are saved here
        self.model = "pix2pixHD"  # selects model to use for netG
        self.norm = "instance"  # instance normalization or batch normalization
        self.use_dropout = False  # use dropout for the generator
        self.data_type = 32  # Supported data type i.e. 8, 16, 32 bit
        self.verbose = False  # toggles verbose

        # input/output sizes
        self.batchSize = 1  # input batch size
        self.loadSize = 1024  # scale images to this size
        self.fineSize = 512  # then crop to this size
        self.label_nc = 35  # # of input label channels
        self.input_nc = 3  # # of input image channels
        self.output_nc = 3  # # of output image channels

        # for setting inputs
        self.dataroot = "./datasets/cityscapes/"  # root directory of the dataset
        self.resize_or_crop = "scale_width"  # scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]
        self.serial_batches = True  # if true, takes images in order to make batches, otherwise takes them randomly
        self.no_flip = True  # if specified, do not flip the images for data argumentation
        self.nThreads = 2  # # threads for loading data
        self.max_dataset_size = float("inf")  # Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.

        # for displays
        self.display_winsize = 512  # display window size
        self.tf_log = False  # if specified, use tensorboard logging. Requires tensorflow installed

        # for generator
        self.netG = "global"  # selects model to use for netG
        self.ngf = 64  # # of gen filters in first conv layer
        self.k_size = 3  # # kernel size conv layer
        self.use_v2 = True  # use DCDCv2
        self.mc = 1024
        self.start_r = 3
        self.n_downsample_global = 4
        self.n_blocks_global = 9
        self.n_blocks_local = 3
        self.n_local_enhancers = 1
        self.niter_fix_global = 0
        self.load_pretrain = ""

        # for instance-wise features
        self.no_instance = False
        self.instance_feat = False
        self.label_feat = False
        self.feat_num = 3
        self.load_features = False
        self.n_downsample_E = 4
        self.nef = 16
        self.n_clusters = 10

        # diy
        self.self_gen = False
        self.mapping_n_block = 3
        self.map_mc = 64
        self.kl = 0
        self.load_pretrainA = ""
        self.load_pretrainB = ""
        self.feat_gan = False
        self.no_cgan = False
        self.map_unet = False
        self.map_densenet = False
        self.fcn = False
        self.is_image = False
        self.label_unpair = False
        self.mapping_unpair = False
        self.unpair_w = 1.0
        self.pair_num = -1
        self.Gan_w = 1
        self.feat_dim = -1
        self.abalation_vae_len = -1

        # useless, just to cooperate with docker
        self.gpu = ""
        self.dataDir = ""
        self.modelDir = ""
        self.logDir = ""
        self.data_dir = ""

        self.use_skip_model = False
        self.use_segmentation_model = False

        self.spatio_size = 64
        self.test_random_crop = False

        self.contain_scratch_L = False
        self.mask_dilation = 0
        self.irregular_mask = ""
        self.mapping_net_dilation = 1

        self.VOC = "VOC_RGB_JPEGImages.bigfile"

        self.non_local = ""
        self.NL_fusion_method = "add"
        self.NL_use_mask = False
        self.correlation_renormalize = False

        self.Smooth_L1 = False

        self.face_restore_setting = 1
        self.face_clean_url = ""
        self.syn_input_url = ""
        self.syn_gt_url = ""

        self.test_on_synthetic = False

        self.use_SN = False

        self.use_two_stage_mapping = False

        self.L1_weight = 10.0
        self.softmax_temperature = 1.0
        self.patch_similarity = False
        self.use_self = False

        self.use_own_dataset = False

        self.test_hole_two_folders = False

        self.no_hole = False
        self.random_hole = False

        self.NL_res = False

        self.image_L1 = False
        self.hole_image_no_mask = False

        self.down_sample_degradation = False

        self.norm_G = "spectralinstance"
        self.init_G = "xavier"

        self.use_new_G = False
        self.use_new_D = False

        self.only_voc = False

        self.cosin_similarity = False

        self.downsample_mode = "nearest"

        self.mapping_exp = 0
        self.inference_optimize = False

        self.initialized = True

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.isTrain = self.initialized  # train or test

        str_ids = self.gpu_ids.split(",")
        self.gpu_ids = []
        for str_id in str_ids:
            int_id = int(str_id)
            if int_id >= 0:
                self.gpu_ids.append(int_id)

        # set gpu ids
        if len(self.gpu_ids) > 0:
            torch.cuda.set_device(self.gpu_ids[0])

        args = vars(self)

        # save to the disk
        expr_dir = os.path.join(self.checkpoints_dir, self.name)
        if save and not self.continue_train:
            file_name = os.path.join(expr_dir, "opt.txt")
            with open(file_name, "wt") as opt_file:
                opt_file.write("------------ Options -------------\n")
                for k, v in sorted(args.items()):
                    opt_file.write("%s: %s\n" % (str(k), str(v)))
                opt_file.write("-------------- End ----------------\n")
        return self


# class BaseOptions:
#     def __init__(self):
#         self.parser = argparse.ArgumentParser()
#         self.initialized = False
#
#     def initialize(self):
#         # experiment specifics
#         self.parser.add_argument(
#             "--name",
#             type=str,
#             default="label2city",
#             help="name of the experiment. It decides where to store samples and models",
#         )
#         self.parser.add_argument(
#             "--gpu_ids", type=str, default="0", help="gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU"
#         )
#         self.parser.add_argument(
#             "--checkpoints_dir", type=str, default="./checkpoints", help="models are saved here"
#         )  ## note: to add this param when using philly
#         # self.parser.add_argument('--project_dir', type=str, default='./', help='the project is saved here')  ################### This is necessary for philly
#         self.parser.add_argument(
#             "--outputs_dir", type=str, default="./outputs", help="models are saved here"
#         )  ## note: to add this param when using philly  Please end with '/'
#         self.parser.add_argument("--model", type=str, default="pix2pixHD", help="which model to use")
#         self.parser.add_argument(
#             "--norm", type=str, default="instance", help="instance normalization or batch normalization"
#         )
#         self.parser.add_argument("--use_dropout", action="store_true", help="use dropout for the generator")
#         self.parser.add_argument(
#             "--data_type",
#             default=32,
#             type=int,
#             choices=[8, 16, 32],
#             help="Supported data type i.e. 8, 16, 32 bit",
#         )
#         self.parser.add_argument("--verbose", action="store_true", default=False, help="toggles verbose")
#
#         # input/output sizes
#         self.parser.add_argument("--batchSize", type=int, default=1, help="input batch size")
#         self.parser.add_argument("--loadSize", type=int, default=1024, help="scale images to this size")
#         self.parser.add_argument("--fineSize", type=int, default=512, help="then crop to this size")
#         self.parser.add_argument("--label_nc", type=int, default=35, help="# of input label channels")
#         self.parser.add_argument("--input_nc", type=int, default=3, help="# of input image channels")
#         self.parser.add_argument("--output_nc", type=int, default=3, help="# of output image channels")
#
#         # for setting inputs
#         self.parser.add_argument("--dataroot", type=str, default="./datasets/cityscapes/")
#         self.parser.add_argument(
#             "--resize_or_crop",
#             type=str,
#             default="scale_width",
#             help="scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]",
#         )
#         self.parser.add_argument(
#             "--serial_batches",
#             action="store_true",
#             help="if true, takes images in order to make batches, otherwise takes them randomly",
#         )
#         self.parser.add_argument(
#             "--no_flip",
#             action="store_true",
#             help="if specified, do not flip the images for data argumentation",
#         )
#         self.parser.add_argument("--nThreads", default=2, type=int, help="# threads for loading data")
#         self.parser.add_argument(
#             "--max_dataset_size",
#             type=int,
#             default=float("inf"),
#             help="Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.",
#         )
#
#         # for displays
#         self.parser.add_argument("--display_winsize", type=int, default=512, help="display window size")
#         self.parser.add_argument(
#             "--tf_log",
#             action="store_true",
#             help="if specified, use tensorboard logging. Requires tensorflow installed",
#         )
#
#         # for generator
#         self.parser.add_argument("--netG", type=str, default="global", help="selects model to use for netG")
#         self.parser.add_argument("--ngf", type=int, default=64, help="# of gen filters in first conv layer")
#         self.parser.add_argument("--k_size", type=int, default=3, help="# kernel size conv layer")
#         self.parser.add_argument("--use_v2", action="store_true", help="use DCDCv2")
#         self.parser.add_argument("--mc", type=int, default=1024, help="# max channel")
#         self.parser.add_argument("--start_r", type=int, default=3, help="start layer to use resblock")
#         self.parser.add_argument(
#             "--n_downsample_global", type=int, default=4, help="number of downsampling layers in netG"
#         )
#         self.parser.add_argument(
#             "--n_blocks_global",
#             type=int,
#             default=9,
#             help="number of residual blocks in the global generator network",
#         )
#         self.parser.add_argument(
#             "--n_blocks_local",
#             type=int,
#             default=3,
#             help="number of residual blocks in the local enhancer network",
#         )
#         self.parser.add_argument(
#             "--n_local_enhancers", type=int, default=1, help="number of local enhancers to use"
#         )
#         self.parser.add_argument(
#             "--niter_fix_global",
#             type=int,
#             default=0,
#             help="number of epochs that we only train the outmost local enhancer",
#         )
#
#         self.parser.add_argument(
#             "--load_pretrain",
#             type=str,
#             default="",
#             help="load the pretrained model from the specified location",
#         )
#
#         # for instance-wise features
#         self.parser.add_argument(
#             "--no_instance", action="store_true", help="if specified, do *not* add instance map as input"
#         )
#         self.parser.add_argument(
#             "--instance_feat",
#             action="store_true",
#             help="if specified, add encoded instance features as input",
#         )
#         self.parser.add_argument(
#             "--label_feat", action="store_true", help="if specified, add encoded label features as input"
#         )
#         self.parser.add_argument("--feat_num", type=int, default=3, help="vector length for encoded features")
#         self.parser.add_argument(
#             "--load_features", action="store_true", help="if specified, load precomputed feature maps"
#         )
#         self.parser.add_argument(
#             "--n_downsample_E", type=int, default=4, help="# of downsampling layers in encoder"
#         )
#         self.parser.add_argument(
#             "--nef", type=int, default=16, help="# of encoder filters in the first conv layer"
#         )
#         self.parser.add_argument("--n_clusters", type=int, default=10, help="number of clusters for features")
#
#         # diy
#         self.parser.add_argument("--self_gen", action="store_true", help="self generate")
#         self.parser.add_argument(
#             "--mapping_n_block", type=int, default=3, help="number of resblock in mapping"
#         )
#         self.parser.add_argument("--map_mc", type=int, default=64, help="max channel of mapping")
#         self.parser.add_argument("--kl", type=float, default=0, help="KL Loss")
#         self.parser.add_argument(
#             "--load_pretrainA",
#             type=str,
#             default="",
#             help="load the pretrained model from the specified location",
#         )
#         self.parser.add_argument(
#             "--load_pretrainB",
#             type=str,
#             default="",
#             help="load the pretrained model from the specified location",
#         )
#         self.parser.add_argument("--feat_gan", action="store_true")
#         self.parser.add_argument("--no_cgan", action="store_true")
#         self.parser.add_argument("--map_unet", action="store_true")
#         self.parser.add_argument("--map_densenet", action="store_true")
#         self.parser.add_argument("--fcn", action="store_true")
#         self.parser.add_argument("--is_image", action="store_true", help="train image recon only pair data")
#         self.parser.add_argument("--label_unpair", action="store_true")
#         self.parser.add_argument("--mapping_unpair", action="store_true")
#         self.parser.add_argument("--unpair_w", type=float, default=1.0)
#         self.parser.add_argument("--pair_num", type=int, default=-1)
#         self.parser.add_argument("--Gan_w", type=float, default=1)
#         self.parser.add_argument("--feat_dim", type=int, default=-1)
#         self.parser.add_argument("--abalation_vae_len", type=int, default=-1)
#
#         ######################### useless, just to cooperate with docker
#         self.parser.add_argument("--gpu", type=str)
#         self.parser.add_argument("--dataDir", type=str)
#         self.parser.add_argument("--modelDir", type=str)
#         self.parser.add_argument("--logDir", type=str)
#         self.parser.add_argument("--data_dir", type=str)
#
#         self.parser.add_argument("--use_skip_model", action="store_true")
#         self.parser.add_argument("--use_segmentation_model", action="store_true")
#
#         self.parser.add_argument("--spatio_size", type=int, default=64)
#         self.parser.add_argument("--test_random_crop", action="store_true")
#         ##########################
#
#         self.parser.add_argument("--contain_scratch_L", action="store_true")
#         self.parser.add_argument(
#             "--mask_dilation", type=int, default=0
#         )  ## Don't change the input, only dilation the mask
#
#         self.parser.add_argument(
#             "--irregular_mask", type=str, default="", help="This is the root of the mask"
#         )
#         self.parser.add_argument(
#             "--mapping_net_dilation",
#             type=int,
#             default=1,
#             help="This parameter is the dilation size of the translation net",
#         )
#
#         self.parser.add_argument(
#             "--VOC", type=str, default="VOC_RGB_JPEGImages.bigfile", help="The root of VOC dataset"
#         )
#
#         self.parser.add_argument("--non_local", type=str, default="", help="which non_local setting")
#         self.parser.add_argument(
#             "--NL_fusion_method",
#             type=str,
#             default="add",
#             help="how to fuse the origin feature and nl feature",
#         )
#         self.parser.add_argument(
#             "--NL_use_mask", action="store_true", help="If use mask while using Non-local mapping model"
#         )
#         self.parser.add_argument(
#             "--correlation_renormalize",
#             action="store_true",
#             help="Since after mask out the correlation matrix(which is softmaxed), the sum is not 1 any more, enable this param to re-weight",
#         )
#
#         self.parser.add_argument("--Smooth_L1", action="store_true", help="Use L1 Loss in image level")
#
#         self.parser.add_argument(
#             "--face_restore_setting", type=int, default=1, help="This is for the aligned face restoration"
#         )
#         self.parser.add_argument("--face_clean_url", type=str, default="")
#         self.parser.add_argument("--syn_input_url", type=str, default="")
#         self.parser.add_argument("--syn_gt_url", type=str, default="")
#
#         self.parser.add_argument(
#             "--test_on_synthetic",
#             action="store_true",
#             help="If you want to test on the synthetic data, enable this parameter",
#         )
#
#         self.parser.add_argument("--use_SN", action="store_true", help="Add SN to every parametric layer")
#
#         self.parser.add_argument(
#             "--use_two_stage_mapping", action="store_true", help="choose the model which uses two stage"
#         )
#
#         self.parser.add_argument("--L1_weight", type=float, default=10.0)
#         self.parser.add_argument("--softmax_temperature", type=float, default=1.0)
#         self.parser.add_argument(
#             "--patch_similarity",
#             action="store_true",
#             help="Enable this denotes using 3*3 patch to calculate similarity",
#         )
#         self.parser.add_argument(
#             "--use_self",
#             action="store_true",
#             help="Enable this denotes that while constructing the new feature maps, using original feature (diagonal == 1)",
#         )
#
#         self.parser.add_argument("--use_own_dataset", action="store_true")
#
#         self.parser.add_argument(
#             "--test_hole_two_folders",
#             action="store_true",
#             help="Enable this parameter means test the restoration with inpainting given twp folders which are mask and old respectively",
#         )
#
#         self.parser.add_argument(
#             "--no_hole",
#             action="store_true",
#             help="While test the full_model on non_scratch data, do not add random mask into the real old photos",
#         )  ## Only for testing
#         self.parser.add_argument(
#             "--random_hole",
#             action="store_true",
#             help="While training the full model, 50% probability add hole",
#         )
#
#         self.parser.add_argument("--NL_res", action="store_true", help="NL+Resdual Block")
#
#         self.parser.add_argument("--image_L1", action="store_true", help="Image level loss: L1")
#         self.parser.add_argument(
#             "--hole_image_no_mask",
#             action="store_true",
#             help="while testing, give hole image but not give the mask",
#         )
#
#         self.parser.add_argument(
#             "--down_sample_degradation",
#             action="store_true",
#             help="down_sample the image only, corresponds to [down_sample_face]",
#         )
#
#         self.parser.add_argument(
#             "--norm_G", type=str, default="spectralinstance", help="The norm type of Generator"
#         )
#         self.parser.add_argument(
#             "--init_G",
#             type=str,
#             default="xavier",
#             help="normal|xavier|xavier_uniform|kaiming|orthogonal|none",
#         )
#
#         self.parser.add_argument("--use_new_G", action="store_true")
#         self.parser.add_argument("--use_new_D", action="store_true")
#
#         self.parser.add_argument(
#             "--only_voc", action="store_true", help="test the trianed celebA face model using VOC face"
#         )
#
#         self.parser.add_argument(
#             "--cosin_similarity",
#             action="store_true",
#             help="For non-local, using cosin to calculate the similarity",
#         )
#
#         self.parser.add_argument(
#             "--downsample_mode",
#             type=str,
#             default="nearest",
#             help="For partial non-local, choose how to downsample the mask",
#         )
#
#         self.parser.add_argument("--mapping_exp",type=int,default=0,help='Default 0: original PNL|1: Multi-Scale Patch Attention')
#         self.parser.add_argument("--inference_optimize",action='store_true',help='optimize the memory cost')
#
#
#         self.initialized = True
#
#     def parse(self, save=True):
#         if not self.initialized:
#             self.initialize()
#         self.opt = self.parser.parse_args()
#         self.opt.isTrain = self.isTrain  # train or test
#
#         str_ids = self.opt.gpu_ids.split(",")
#         self.opt.gpu_ids = []
#         for str_id in str_ids:
#             int_id = int(str_id)
#             if int_id >= 0:
#                 self.opt.gpu_ids.append(int_id)
#
#         # set gpu ids
#         if len(self.opt.gpu_ids) > 0:
#             # pass
#             torch.cuda.set_device(self.opt.gpu_ids[0])
#
#         args = vars(self.opt)
#
#         # print('------------ Options -------------')
#         # for k, v in sorted(args.items()):
#         #     print('%s: %s' % (str(k), str(v)))
#         # print('-------------- End ----------------')
#
#         # save to the disk
#         expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
#         util.mkdirs(expr_dir)
#         if save and not self.opt.continue_train:
#             file_name = os.path.join(expr_dir, "opt.txt")
#             with open(file_name, "wt") as opt_file:
#                 opt_file.write("------------ Options -------------\n")
#                 for k, v in sorted(args.items()):
#                     opt_file.write("%s: %s\n" % (str(k), str(v)))
#                 opt_file.write("-------------- End ----------------\n")
#         return self.opt
