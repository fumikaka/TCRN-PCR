import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import neuralnet_pytorch as nnt
import torch.nn.functional as F

from model.vggg_alter import vgg16_bn


from layers.af_decoder import *


# Pending imports


class TDPNet(nn.Module):
    # Core TDPNet module
    # configuration -- The setting of network
    # prototypes -- The initialized 3D prototypes
    def __init__(self, configuration, num_pts=2048, adain=True):
        super(TDPNet, self).__init__()
        self.opt = configuration
        self.num_points = num_pts
        self.batch_size = self.opt.batch_size
        self.model_dim = 512
        self.num_latent = 512
        self.device = self.opt.device
        # self.num_prototypes = prototypes.shape[0]
        self.num_slaves = self.opt.num_slaves
        # self.adain = adain
        # self.num_pts_per_proto = num_pts // self.num_prototypes

        # Image Encoder Part
        # self.img_feature_extractor = vgg16_bn(pretrained=False).features
        self.img_feature_extractor = vgg16_bn()

        self.img_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))


        self.linear = nn.Linear(512, 128)


        self.decoder = SP_DecoderEigen3steps()


    def forward(self, x, noise, img_flag=True):
        if img_flag:
            # _, _, _, _, x5 = self.img_feature_extractor(x);
            x5 = self.img_feature_extractor(x)
            x5 = self.img_pool(x5).squeeze(-1).squeeze(-1)
            # x4 = self.img_pool(x4).squeeze(-1).squeeze(-1)
            # x3 = self.img_pool(x3).squeeze(-1).squeeze(-1)
            # z=self.mlp(torch.cat([x5, x4, x3], dim=-1))

            feat = self.linear(x5)

            outputs = self.decoder(noise, feat)
            # print(outputs.size())



            return outputs.transpose(1, 2).contiguous()
            # return points.transpose(1, 2).contiguous()

        # else:
        #     # Dummy image feature
        #     latent_vector = torch.cuda.FloatTensor(np.ones((x.shape[0], 1472)))
        #
        #     # Deform each Patch using real pc features
        #     output_points = torch.cat([self.decoder[idx](latent_vector, x) for idx in range(self.num_prototypes)], dim=2)
        #
        #
        #     return output_points.transpose(1, 2).contiguous()

    def _set_finetune(self):
        active_layer = 3
        for idx in range(len(self.img_feature_extractor) - 1, -1, -1):
            if isinstance(self.img_feature_extractor[idx], nn.Conv2d):
                if active_layer > 0:
                    self.img_feature_extractor[idx].requires_grad_(True)
                    active_layer -= 1
                else:
                    self.img_feature_extractor[idx].requires_grad_(False)
        return None

    def activate_prototype_finetune(self):
        for idx in range(self.num_prototypes):
            self.decoder[idx].activate_prototype_finetune()
        return None

    def update_prototypes(self, prototypes):
        for idx in range(self.num_prototypes):
            self.decoder[idx].update_prototype(np.expand_dims(prototypes[idx], axis=0))



    def transform(self, pc_feat, img_feat):
        pc_feat = (pc_feat - torch.mean(pc_feat, -1, keepdim=True)) / torch.sqrt(
            torch.var(pc_feat, -1, keepdim=True) + 1e-8)
        mean, var = torch.mean(img_feat, (2, 3)), torch.var(torch.flatten(img_feat, 2), 2)
        # print("!!!!!")
        # print(pc_feat.shape)
        # print(mean.shape)
        # print(var.shape)
        # c = nnt.utils.dimshuffle(mean, (0, 1, 'x'))
        # print(c.shape)
        # output = (pc_feat + nnt.utils.dimshuffle(mean, (0, 1, 'x'))) * torch.sqrt(
        #     nnt.utils.dimshuffle(var, (0, 1, 'x')) + 1e-8)
        output = (pc_feat * torch.sqrt(nnt.utils.dimshuffle(var, (0, 1, 'x')) + 1e-8)) + nnt.utils.dimshuffle(mean, (
            0, 1, 'x'))
        return output

    def extract_prototypes(self):
        return np.concatenate([self.decoder[idx].extract_prototype() for idx in range(self.num_prototypes)])
