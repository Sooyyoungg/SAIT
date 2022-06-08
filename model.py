import numpy as np
import torch
from torch import nn
import networks

class Pix2Pix(nn.Module):
    def __init__(self, config):
        super(Pix2Pix, self).__init__()
        self.config = config
        self.device = torch.device('cuda:' + str(self.config.gpu_ids[0]) if torch.cuda.is_available() else 'cpu')

        # self.loss_names = ['G_GAN', 'G_l1', 'D_real', 'D_fake']
        # self.visual_names = ['input', 'fake_output', 'real_output']

        self.netG = networks.define_G(self.config.input_nc, self.config.output_nc, self.config.ngf, self.config.netG, self.config.norm,
                                      not self.config.no_dropout, self.config.init_type, self.config.init_gain, self.config.gpu_ids)
        self.netD = networks.define_D(self.config.input_nc + self.config.output_nc, self.config.ndf, self.config.netD,
                                          self.config.n_layers_D, self.config.norm, self.config.init_type, self.config.init_gain, self.config.gpu_ids)

        self.criterion_GAN = networks.GANLoss(config.gan_mode).to(self.device)
        self.criterion_L1 = torch.nn.L1Loss()

        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=self.config.lr, betas=(self.config.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=self.config.lr, betas=(self.config.beta1, 0.999))
        self.G_scheduler = networks.get_scheduler(self.optimizer_G, config)
        self.D_scheduler = networks.get_scheduler(self.optimizer_D, config)

    def train(self, data):
        sem = data['sem']
        depth = data['depth']

        # forward - Generator
        fake_dep = self.netG(sem)
        print(sem.shape, fake_dep.shape)

        # forward - Discriminator - fake
        # D_fake_in = np.concatenate((fake_dep, sem), axis=1)
        D_fake_in = fake_dep
        D_fake_out = self.netD(D_fake_in)

        # forward - Discriminator - real
        # D_real_in = np.concatenate((depth, sem), axis=1)
        D_real_in = depth
        D_real_out = self.netD(D_real_in)

        # Loss - G
        self.optimizer_G.zero_grad()
        G_loss_GAN = self.criterion_GAN(D_fake_out, True)
        G_loss_L1 = self.criterion_L1(fake_dwi, real_dwi)
        G_loss = G_loss_GAN + G_loss_L1
        G_loss.backward()
        self.optimizer_G.step()

        # Loss - D
        self.optimizer_D.zero_grad()
        D_loss_real = self.criterion_GAN(D_real_out, True)
        D_loss_fake = self.criterion_GAN(D_fake_out, False)
        D_loss = D_loss_real + D_loss_fake
        D_loss.backward()
        self.optimizer_D.step()

        train_dict = {}
        train_dict['G_loss'] = G_loss
        train_dict['D_loss'] = D_loss
        train_dict['fake_depth'] = fake_depth
        train_dict['real_depth'] = real_depth

        return train_dict

    def val(self, data):
        print("Validation Start!")
        with torch.no_grad():
            t1 = data['t1']
            b0 = data['b0']
            real_dwi = data['dwi']

            # Generator
            input_img = np.concatenate((t1, b0), axis=1)
            fake_dwi = self.netG(input_img, data['grad'])

            # Loss - G
            G_loss_GAN = self.criterion_GAN(D_fake_out, True)
            G_loss_L1 = self.criterion_L1(fake_dwi, real_dwi)
            G_loss = G_loss_GAN + G_loss_L1

            val_dict = {}
            val_dict['G_loss'] = G_loss
            val_dict['dwi'] = real_dwi
            val_dict['pred'] = fake_dwi

        return val_dict

    def test(self, data):
        with torch.no_grad():
            t1 = data['t1']
            b0 = data['b0']
            real_dwi = data['dwi']

            # Generator
            input_img = np.concatenate((t1, b0), axis=1)
            fake_dwi = self.netG(input_img, data['grad'])

            # Loss - G
            G_loss_GAN = self.criterion_GAN(D_fake_out, True)
            G_loss_L1 = self.criterion_L1(fake_dwi, real_dwi)
            G_loss = G_loss_GAN + G_loss_L1

            test_dict = {}
            test_dict['G_loss'] = G_loss
            test_dict['dwi'] = real_dwi
            test_dict['pred'] = fake_dwi

        return test_dict