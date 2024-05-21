# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from .models.mapping_model import Pix2PixHDModel_Mapping
from PIL import Image
import torch
import torchvision.utils as vutils
import torchvision.transforms as transforms
import numpy as np
import cv2


def data_transforms(img, method=Image.Resampling.BILINEAR, scale=False):
    ow, oh = img.size
    pw, ph = ow, oh
    if scale:
        ow = 256 if ow < oh else pw / ph * 256
        oh = 256 if oh <= ow else ph / pw * 256

    h = int(round(oh / 4) * 4)
    w = int(round(ow / 4) * 4)

    if (h == ph) and (w == pw):
        return img

    return img.resize((w, h), method)


def data_transforms_rgb_old(img):
    w, h = img.size
    A = img

    if w < 256 or h < 256:
        A = transforms.Scale(256, Image.BILINEAR)(img)

    return transforms.CenterCrop(256)(A)


def irregular_hole_synthesize(img, mask):
    img_np = np.array(img).astype("uint8")
    mask_np = np.array(mask).astype("uint8")
    mask_np = mask_np / 255
    img_new = img_np * (1 - mask_np) + mask_np * 255

    hole_img = Image.fromarray(img_new.astype("uint8")).convert("RGB")

    return hole_img


class Options:
    def __init__(self, **kwargs):
        self.mask_dilation = None
        self.NL_use_mask = None
        self.how_many = None
        self.contain_dontcare_label = None
        for key, value in kwargs.items():
            setattr(self, key, value)


opt = Options(
    name="label2city",  # name of the experiment. It decides where to store samples and models
    gpu_ids=0,
    checkpoints_dir="./checkpoints",  # models are saved here
    outputs_dir="./outputs",  # models are saved here
    model="pix2pixHD",  # selects model to use for netG
    norm="instance",  # instance normalization or batch normalization
    use_dropout=False,  # use dropout for the generator
    data_type=32,  # Supported data type i.e. 8, 16, 32 bit
    verbose=False,  # toggles verbose

    # input/output sizes
    batchSize=1,  # input batch size
    loadSize=1024,  # scale images to this size
    fineSize=512,  # then crop to this size
    label_nc=35,  # # of input label channels
    input_nc=3,  # # of input image channels
    output_nc=3,  # # of output image channels

    # for setting inputs
    dataroot="./datasets/cityscapes/",  # root directory of the dataset
    resize_or_crop="scale_width",
    # scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]
    serial_batches=True,  # if true, takes images in order to make batches, otherwise takes them randomly
    no_flip=True,  # if specified, do not flip the images for data argumentation
    nThreads=2,  # # threads for loading data
    max_dataset_size=float("inf"),  # Max number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.

    # for displays
    display_winsize=512,  # display window size
    tf_log=False,  # if specified, use tensorboard logging. Requires tensorflow installed

    # for generator
    netG="global",  # selects model to use for netG
    ngf=64,  # # of gen filters in first conv layer
    k_size=3,  # # kernel size conv layer
    use_v2=True,  # use DCDCv2
    mc=1024,
    start_r=3,
    n_downsample_global=4,
    n_blocks_global=9,
    n_blocks_local=3,
    n_local_enhancers=1,
    niter_fix_global=0,
    load_pretrain="",

    # for instance-wise features
    no_instance=False,
    instance_feat=False,
    label_feat=False,
    feat_num=3,
    load_features=False,
    n_downsample_E=4,
    nef=16,
    n_clusters=10,

    # diy
    self_gen=False,
    mapping_n_block=3,
    map_mc=64,
    kl=0,
    load_pretrainA="",
    load_pretrainB="",
    feat_gan=False,
    no_cgan=False,
    map_unet=False,
    map_densenet=False,
    fcn=False,
    is_image=False,
    label_unpair=False,
    mapping_unpair=False,
    unpair_w=1.0,
    pair_num=-1,
    Gan_w=1,
    feat_dim=-1,
    abalation_vae_len=-1,

    # useless, just to cooperate with docker
    gpu="",
    dataDir="",
    modelDir="",
    logDir="",
    data_dir="",

    use_skip_model=False,
    use_segmentation_model=False,

    spatio_size=64,
    test_random_crop=False,

    contain_scratch_L=False,
    mask_dilation=0,
    irregular_mask="",
    mapping_net_dilation=1,

    VOC="VOC_RGB_JPEGImages.bigfile",

    non_local="",
    NL_fusion_method="add",
    NL_use_mask=False,
    correlation_renormalize=False,

    Smooth_L1=False,

    face_restore_setting=1,
    face_clean_url="",
    syn_input_url="",
    syn_gt_url="",

    test_on_synthetic=False,

    use_SN=False,

    use_two_stage_mapping=False,

    L1_weight=10.0,
    softmax_temperature=1.0,
    patch_similarity=False,
    use_self=False,

    use_own_dataset=False,

    test_hole_two_folders=False,

    no_hole=False,
    random_hole=False,

    NL_res=False,

    image_L1=False,
    hole_image_no_mask=False,

    down_sample_degradation=False,

    norm_G="spectralinstance",
    init_G="xavier",

    use_new_G=False,
    use_new_D=False,

    only_voc=False,

    cosin_similarity=False,

    downsample_mode="nearest",

    mapping_exp=0,
    inference_optimize=False,

    initialized=True,
    aspect_ratio=1.0,
    results_dir="./results/",
    phase="test",
    which_epoch="latest",
    how_many=50,
    cluster_path="features_clustered_010.npy",
    use_encoded_image=False,
    export_onnx=None,
    engine=None,
    onnx=None,
    start_epoch=-1,
    test_dataset="Real_RGB_old.bigfile",
    no_degradation=False,
    no_load_VAE=False,
    no_lsgan=False,
    use_v2_degradation=False,
    use_vae_which_epoch="latest",
    isTrain=False,
    generate_pair=False,
    multi_scale_test=0.5,
    multi_scale_threshold=0.5,
    mask_need_scale=False,
    scale_num=1,
    save_feature_url="",
    test_input="",
    test_mask="",
    test_gt="",
    scale_input=False,
    save_feature_name="features.json",
    test_rgb_old_wo_scratch=False,
    test_mode="Crop",
    Quality_restore=False,
    Scratch_and_Quality_restore=False,
    HR=False,

)


def parameter_set(opt):
    ## Default parameters
    opt.name = "label2city"
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.label_nc = 0
    opt.n_downsample_global = 3
    opt.mc = 64
    opt.NL_use_mask = False
    opt.non_local = ""
    opt.k_size = 4
    opt.start_r = 1
    opt.mapping_n_block = 6
    opt.map_mc = 512
    opt.no_instance = True
    opt.checkpoints_dir = "./checkpoints/restoration"
    opt.isTrain = False
    if opt.Quality_restore:
        opt.name = "mapping_quality"
        opt.load_pretrainA = os.path.join(opt.checkpoints_dir, "VAE_A_quality")
        opt.load_pretrainB = os.path.join(opt.checkpoints_dir, "VAE_B_quality")
    if opt.Scratch_and_Quality_restore:
        opt.NL_res = True
        opt.use_SN = True
        opt.correlation_renormalize = True
        opt.NL_use_mask = True
        opt.NL_fusion_method = "combine"
        opt.non_local = "Setting_42"
        opt.name = "mapping_scratch"
        opt.load_pretrainA = os.path.join(opt.checkpoints_dir, "VAE_A_quality")
        opt.load_pretrainB = os.path.join(opt.checkpoints_dir, "VAE_B_scratch")
        if opt.HR:
            opt.mapping_exp = 1
            opt.inference_optimize = True
            opt.mask_dilation = 3
            opt.name = "mapping_Patch_Attention"


def perform(test_input, test_mask, test_mode, outputs_dir, Quality_restore=False, Scratch_and_Quality_restore=False,
            HR=False, gpu_ids=-1):
    # opt = TestOptions().parse(save=False)
    # Set additional parameters
    opt.Quality_restore = Quality_restore
    opt.gpu_ids = gpu_ids
    opt.Scratch_and_Quality_restore = Scratch_and_Quality_restore
    opt.test_input = test_input
    opt.outputs_dir = outputs_dir
    opt.test_mode = test_mode
    opt.test_mask = test_mask
    opt.HR = HR
    print('test_input:',test_input)
    # Debugging: Print parsed options
    # print(f'Parsed Options: {opt.__dict__}')
    parameter_set(opt)
    model = Pix2PixHDModel_Mapping()
    model.initialize(opt)
    model.eval()

    if not os.path.exists(outputs_dir + "/" + "input_image"):
        os.makedirs(outputs_dir + "/" + "input_image")
    if not os.path.exists(outputs_dir + "/" + "restored_image"):
        os.makedirs(outputs_dir + "/" + "restored_image")
    if not os.path.exists(outputs_dir + "/" + "origin"):
        os.makedirs(outputs_dir + "/" + "origin")

    input_loader = os.listdir(test_input)
    dataset_size = len(input_loader)
    print('dataset_size', dataset_size)
    input_loader.sort()

    if test_mask != "":
        mask_loader = os.listdir(test_mask)
        dataset_size = len(os.listdir(test_mask))
        mask_loader.sort()
    print('start to transform images')
    img_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    mask_transform = transforms.ToTensor()

    for i in range(dataset_size):
        input_name = input_loader[i]
        input_file = os.path.join(opt.test_input, input_name)
        print('input_file', input_file)
        if not os.path.isfile(input_file):
            print("Skipping non-file %s" % input_name)
            continue
        input = Image.open(input_file).convert("RGB")
        # if input.width <= 4000 and input.height <= 4000:

        print("Now you are processing %s" % (input_name))

        if opt.NL_use_mask:
            print('using mask')
            mask_name = mask_loader[i]
            mask = Image.open(os.path.join(opt.test_mask, mask_name)).convert("RGB")
            if opt.mask_dilation != 0:
                print('dilating')
                kernel = np.ones((3, 3), np.uint8)
                mask = np.array(mask)
                mask = cv2.dilate(mask, kernel, iterations=opt.mask_dilation)
                mask = Image.fromarray(mask.astype('uint8'))
            origin = input
            input = irregular_hole_synthesize(input, mask)
            print('transforming')
            mask = mask_transform(mask)
            mask = mask[:1, :, :]  ## Convert to single channel
            mask = mask.unsqueeze(0)
            print('transforming2')
            input = img_transform(input)
            print('transforming3')
            input = input.unsqueeze(0)
            print('transforming4')
        else:
            if opt.test_mode == "Scale":
                print('scaling')
                input = data_transforms(input, scale=True)
            if opt.test_mode == "Full":
                print('full')
                input = data_transforms(input, scale=False)
            if opt.test_mode == "Crop":
                print('crop')
                input = data_transforms_rgb_old(input)
            origin = input
            input = img_transform(input)
            input = input.unsqueeze(0)
            mask = torch.zeros_like(input)

        try:
            with torch.no_grad():
                generated = model.inference(input, mask, gpu)
        except Exception as ex:
            print("Skip %s due to an error:\n%s" % (input_name, str(ex)))
            continue

        if input_name.endswith(".jpg"):
            input_name = input_name[:-4] + ".png"

        vutils.save_image(
            (input + 1.0) / 2.0,
            outputs_dir + "/input_image/" + input_name,
            nrow=1,
            padding=0,
            normalize=True,
        )
        vutils.save_image(
            (generated.data.cpu() + 1.0) / 2.0,
            outputs_dir + "/restored_image/" + input_name,
            nrow=1,
            padding=0,
            normalize=True,
        )

        origin.save(outputs_dir + "/origin/" + input_name)
