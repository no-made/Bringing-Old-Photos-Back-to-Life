# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import importlib

import torch
from .base_network import BaseNetwork
from .generator import *
from .encoder import *

# import util.util as util
# from Face_Enhancement.util import util


def find_network_using_name(target_network_name, filename):
    target_class_name = target_network_name + filename
    module_name = "Face_Enhancement.models.networks." + filename
    print('target_class_name:', target_class_name, 'module_name:', module_name)
    network = find_class_in_module(target_class_name, module_name)

    assert issubclass(network, BaseNetwork), "Class %s should be a subclass of BaseNetwork" % network

    return network


def modify_options(opt, is_train):
    # Modify the 'opt' object directly based on your requirements
    netG_cls = find_network_using_name(opt.netG, "generator")
    print("netG_cls", netG_cls)
    netG_cls.modify_options(opt, is_train)

    if is_train:
        netD_cls = find_network_using_name(opt.netD, "discriminator")
        netD_cls.modify_options(opt, is_train)

    netE_cls = find_network_using_name("conv", "encoder")
    netE_cls.modify_options(opt, is_train)

    return opt

def find_class_in_module(target_cls_name, module):
    print('module:', module)
    target_cls_name = target_cls_name.replace("_", "").lower()
    clslib = importlib.import_module(module)
    cls = None
    for name, clsobj in clslib.__dict__.items():
        if name.lower() == target_cls_name:
            cls = clsobj

    if cls is None:
        print(
            "In %s, there should be a class whose name matches %s in lowercase without underscore(_)"
            % (module, target_cls_name)
        )
        exit(0)
    print('class:', cls)
    return cls

def create_network(cls, opt):
    net = cls(opt)
    net.print_network()
    if opt.gpu_ids:
        assert torch.cuda.is_available()
        net.cuda()
    net.init_weights(opt.init_type, opt.init_variance)
    return net


def define_G(opt):
    netG_cls = find_network_using_name(opt.netG, "generator")
    return create_network(netG_cls, opt)


def define_D(opt):
    netD_cls = find_network_using_name(opt.netD, "discriminator")
    return create_network(netD_cls, opt)


def define_E(opt):
    # there exists only one encoder type
    netE_cls = find_network_using_name("conv", "encoder")
    return create_network(netE_cls, opt)
