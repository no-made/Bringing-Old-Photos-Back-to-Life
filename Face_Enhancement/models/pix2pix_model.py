# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os

import torch
# import .networks.__init__ as networks
from .networks.__init__ import modify_options as networks_modify_options
from .networks.__init__ import define_G as networks_define_G , define_D as networks_define_D, define_E as networks_define_E
# import .models.networks as networks
# from ..util import util
# from Global.models.networks import GANLoss, VGGLoss, KLDLoss


class Pix2PixModel(torch.nn.Module):
    @staticmethod
    def modify_options(opt, is_train):
        networks_modify_options(opt, is_train)
        return opt

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        # self.opt.gpu_ids
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() else torch.ByteTensor
        print('********************************netG:', opt.netG)
        self.modify_options(opt, is_train=False)
        self.netG, self.netD, self.netE = self.initialize_networks(opt)

        # set loss functions
        if opt.isTrain:
            print('It shouldn\'t be True')
            # self.criterionGAN = GANLoss(opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            # self.criterionFeat = torch.nn.L1Loss()
            # if not opt.no_vgg_loss:
            #     self.criterionVGG = VGGLoss(self.opt.gpu_ids)
            # if opt.use_vae:
            #     self.KLDLoss = KLDLoss()

    # Entry point for all calls involving forward pass
    # of deep networks. We used this approach since DataParallel module
    # can't parallelize custom functions, we branch to different
    # routines based on |mode|.
    def forward(self, data, mode):
        input_semantics, real_image, degraded_image = self.preprocess_input(data)
        print('in forward', input_semantics.shape, real_image.shape, degraded_image.shape)
        if mode == "generator":
            g_loss, generated = self.compute_generator_loss(input_semantics, degraded_image, real_image)
            return g_loss, generated
        elif mode == "discriminator":
            d_loss = self.compute_discriminator_loss(input_semantics, degraded_image, real_image)
            return d_loss
        elif mode == "encode_only":
            z, mu, logvar = self.encode_z(real_image)
            return mu, logvar
        elif mode == "inference":
            with torch.no_grad():
                fake_image, _ = self.generate_fake(input_semantics, degraded_image, real_image)
            return fake_image
        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):
        print('in create_optimizers')
        G_params = list(self.netG.parameters())
        if opt.use_vae:
            G_params += list(self.netE.parameters())
        if opt.isTrain:
            D_params = list(self.netD.parameters())

        beta1, beta2 = opt.beta1, opt.beta2
        if opt.no_TTUR:
            G_lr, D_lr = opt.lr, opt.lr
        else:
            G_lr, D_lr = opt.lr / 2, opt.lr * 2

        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))

        return optimizer_G, optimizer_D

    def save(self, epoch):
        print('in save')
        save_network(self.netG, "G", epoch, self.opt)
        save_network(self.netD, "D", epoch, self.opt)
        if self.opt.use_vae:
            save_network(self.netE, "E", epoch, self.opt)

    ############################################################################
    # Private helper methods
    ############################################################################

    def initialize_networks(self, opt):
        netG = networks_define_G(opt)
        netD = networks_define_D(opt) if opt.isTrain else None
        netE = networks_define_E(opt) if opt.use_vae else None
        print('initialized networks')
        if not opt.isTrain or opt.continue_train:
            netG = load_network(netG, "G", opt.which_epoch, opt)
            if opt.isTrain:
                netD = load_network(netD, "D", opt.which_epoch, opt)
            if opt.use_vae:
                netE = load_network(netE, "E", opt.which_epoch, opt)

        return netG, netD, netE

    # preprocess the input, such as moving the tensors to GPUs and
    # transforming the label map to one-hot encoding
    # |data|: dictionary of the input data

    def preprocess_input(self, data):
        # move to GPU and change data types
        # data['label'] = data['label'].long()
        print('in preprocess_input', data['label'].shape, data['image'].shape)
        if not self.opt.isTrain:
            if self.use_gpu():
                data["label"] = data["label"].cuda()
                data["image"] = data["image"].cuda()
            return data["label"], data["image"], data["image"]

        ## While testing, the input image is the degraded face
        print('degraded_image', data['degraded_image'].shape)
        if self.use_gpu():
            data["label"] = data["label"].cuda()
            data["degraded_image"] = data["degraded_image"].cuda()
            data["image"] = data["image"].cuda()

        # # create one-hot label map
        # label_map = data['label']
        # bs, _, h, w = label_map.size()
        # nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label \
        #     else self.opt.label_nc
        # input_label = self.FloatTensor(bs, nc, h, w).zero_()
        # input_semantics = input_label.scatter_(1, label_map, 1.0)

        return data["label"], data["image"], data["degraded_image"]

    def compute_generator_loss(self, input_semantics, degraded_image, real_image):
        print('in compute_generator_loss')
        G_losses = {}

        fake_image, KLD_loss = self.generate_fake(
            input_semantics, degraded_image, real_image, compute_kld_loss=self.opt.use_vae
        )

        if self.opt.use_vae:
            G_losses["KLD"] = KLD_loss

        pred_fake, pred_real = self.discriminate(input_semantics, fake_image, real_image)

        G_losses["GAN"] = self.criterionGAN(pred_fake, True, for_discriminator=False)

        if not self.opt.no_ganFeat_loss:
            num_D = len(pred_fake)
            GAN_Feat_loss = self.FloatTensor(1).fill_(0)
            for i in range(num_D):  # for each discriminator
                # last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(pred_fake[i]) - 1
                for j in range(num_intermediate_outputs):  # for each layer output
                    unweighted_loss = self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach())
                    GAN_Feat_loss += unweighted_loss * self.opt.lambda_feat / num_D
            G_losses["GAN_Feat"] = GAN_Feat_loss

        if not self.opt.no_vgg_loss:
            G_losses["VGG"] = self.criterionVGG(fake_image, real_image) * self.opt.lambda_vgg

        return G_losses, fake_image

    def compute_discriminator_loss(self, input_semantics, degraded_image, real_image):
        print('in compute_discriminator_loss')
        D_losses = {}
        with torch.no_grad():
            fake_image, _ = self.generate_fake(input_semantics, degraded_image, real_image)
            fake_image = fake_image.detach()
            fake_image.requires_grad_()

        pred_fake, pred_real = self.discriminate(input_semantics, fake_image, real_image)

        D_losses["D_Fake"] = self.criterionGAN(pred_fake, False, for_discriminator=True)
        D_losses["D_real"] = self.criterionGAN(pred_real, True, for_discriminator=True)

        return D_losses

    def encode_z(self, real_image):
        mu, logvar = self.netE(real_image)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def generate_fake(self, input_semantics, degraded_image, real_image, compute_kld_loss=False):
        print('in generate_fake')
        z = None
        KLD_loss = None
        if self.opt.use_vae:
            z, mu, logvar = self.encode_z(real_image)
            if compute_kld_loss:
                KLD_loss = self.KLDLoss(mu, logvar) * self.opt.lambda_kld

        fake_image = self.netG(input_semantics, degraded_image, z=z)

        assert (
            not compute_kld_loss
        ) or self.opt.use_vae, "You cannot compute KLD loss if opt.use_vae == False"

        return fake_image, KLD_loss

    # Given fake and real image, return the prediction of discriminator
    # for each fake and real image.

    def discriminate(self, input_semantics, fake_image, real_image):
        print('in discriminate')
        if self.opt.no_parsing_map:
            fake_concat = fake_image
            real_concat = real_image
        else:
            fake_concat = torch.cat([input_semantics, fake_image], dim=1)
            real_concat = torch.cat([input_semantics, real_image], dim=1)

        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)

        discriminator_out = self.netD(fake_and_real)

        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real

    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        print('in divide_pred')
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[: tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2 :] for tensor in p])
        else:
            fake = pred[: pred.size(0) // 2]
            real = pred[pred.size(0) // 2 :]

        return fake, real

    def get_edges(self, t):
        print('in get_edges')
        edge = self.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu

    def use_gpu(self):
        return False #self.opt.gpu_ids

def save_network(net, label, epoch, opt):
    save_filename = "%s_net_%s.pth" % (epoch, label)
    save_path = os.path.join(opt.checkpoints_dir, opt.name, save_filename)
    torch.save(net.cpu().state_dict(), save_path)
    if len(opt.gpu_ids) and torch.cuda.is_available():
        net.cuda()


def load_network(net, label, epoch, opt):
    print("loading the model from %s" % opt.checkpoints_dir, epoch, label)
    save_filename = "%s_net_%s.pth" % (epoch, label)
    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    save_path = os.path.join(save_dir, save_filename)
    if os.path.exists(save_path):
        weights = torch.load(save_path)
        net.load_state_dict(weights)
    return net