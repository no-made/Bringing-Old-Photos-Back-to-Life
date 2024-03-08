# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import importlib
import torch


def find_model_using_name(model_name):
    # Given the option --model [modelname],
    # the file "models/modelname_model.py"
    # will be imported.
    model_name += "_model"
    model_filename = "." + model_name
    modellib = importlib.import_module(model_filename, package=__name__)

    # In the file, the class called ModelNameModel() will
    # be instantiated. It has to be a subclass of torch.nn.Module,
    # and it is case-insensitive.
    model = None
    target_model_name = model_name.replace("_", "")  # Convert to lowercase for case-insensitive comparison
    print('target_model_name:', target_model_name)
    for name, cls in modellib.__dict__.items():
        print('name:', name)
        if name.lower() == target_model_name.lower() and issubclass(cls, torch.nn.Module):
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
