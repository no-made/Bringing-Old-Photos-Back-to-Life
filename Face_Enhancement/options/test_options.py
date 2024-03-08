# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import importlib

from .base_options import BaseOptions


class TestOptions(BaseOptions):

    def __init__(self):
        super(TestOptions, self).__init__()
        self.results_dir = "./results/"  # save all the testing results to this directory
        self.which_epoch = "latest"  # which epoch to load? set to latest to use latest cached model
        self.how_many = float("inf")  # how many test images to run
        self.num_upsampling_layers = "normal"  # "normal", "more", "most"
        # Set default values
        self.preprocess_mode = "scale_width_and_crop"  # scaling and cropping of images at load time
        self.crop_size = 512    # crop images to this size
        self.load_size = 1024    # scale images to this size
        self.display_winsize = 256  # display window size for both visdom and HTML
        self.serial_batches = True
        self.no_flip = True
        self.phase = "test"     # test mode
        # self.semantic_nc = 18
        self.isTrain = False

    def print_options(self):
        super(TestOptions, self).print_options()
        print("{:>25}: {:<30}{}\n".format("results_dir", str(self.results_dir), ""))
        print("{:>25}: {:<30}{}\n".format("which_epoch", str(self.which_epoch), ""))
        print("{:>25}: {:<30}{}\n".format("how_many", str(self.how_many), ""))

    def get_option_setter(self, model_name):
        model_class = self.find_model_using_name(model_name)
        return model_class.modify_options
    @staticmethod
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