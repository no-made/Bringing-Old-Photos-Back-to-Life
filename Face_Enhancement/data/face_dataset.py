# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .base_dataset import BaseDataset, get_params, get_transform
from PIL import Image
# import util.util as util
import os
import torch


class FaceTestDataset(BaseDataset):
    def __init__(self, opt):
        super(FaceTestDataset, self).__init__()
        self.opt = opt
        self.parts = [
            "skin",
            "hair",
            "l_brow",
            "r_brow",
            "l_eye",
            "r_eye",
            "eye_g",
            "l_ear",
            "r_ear",
            "ear_r",
            "nose",
            "mouth",
            "u_lip",
            "l_lip",
            "neck",
            "neck_l",
            "cloth",
            "hat",
        ]
        self.label_paths = []
        self.image_paths = []
        self.dataset_size = 0

    @staticmethod
    def modify_options(opt, is_train):
        opt.no_pairing_check = True
        opt.contain_dontcare_label = False
        opt.no_instance = True
        return opt

    def initialize(self, opt):
        print('///////////////////////initialize///////////////////////')
        image_path = os.path.join(opt.dataroot, opt.old_face_folder)
        label_path = os.path.join(opt.dataroot, opt.old_face_label_folder)

        image_list = os.listdir(image_path)
        image_list = sorted(image_list)
        print(image_list)
        self.label_paths = label_path
        self.image_paths = image_list
        self.dataset_size = len(self.image_paths)

    def __getitem__(self, index):

        params = get_params(self.opt, (-1, -1))
        image_name = self.image_paths[index]
        print(self.opt.dataroot, self.opt.old_face_folder, image_name)
        image_path = os.path.join(self.opt.dataroot, self.opt.old_face_folder, image_name)
        image = Image.open(image_path)
        image = image.convert("RGB")

        transform_image = get_transform(self.opt, params)
        image_tensor = transform_image(image)

        img_name = image_name[:-4]
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        full_label = []

        cnt = 0

        for each_part in self.parts:
            part_name = img_name + "_" + each_part + ".png"
            part_url = os.path.join(self.label_paths, part_name)

            if os.path.exists(part_url):
                label = Image.open(part_url).convert("RGB")
                label_tensor = transform_label(label)  ## 3 channels and pixel [0,1]
                full_label.append(label_tensor[0])
            else:
                current_part = torch.zeros((self.opt.load_size, self.opt.load_size))
                full_label.append(current_part)
                cnt += 1

        full_label_tensor = torch.stack(full_label, 0)

        input_dict = {
            "label": full_label_tensor,
            "image": image_tensor,
            "path": image_path,
        }

        return input_dict

    def __len__(self):
        return self.dataset_size

