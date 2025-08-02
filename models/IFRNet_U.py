import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import warp
from loss import Charbonnier_L1, Ternary, Charbonnier_Ada, Geometry
from models.IFRNet_S import convrelu, ResBlock, Encoder, Decoder4, Decoder3, Decoder2

def resize(x, scale_factor):
    return F.interpolate(x, scale_factor=scale_factor, mode="bilinear", align_corners=False)

class Decoder1(nn.Module):
    def __init__(self):
        super(Decoder1, self).__init__();
        self.convblock = nn.Sequential(
            convrelu(76, 72),
            ResBlock(72, 24),
            nn.ConvTranspose2d(72, 12, 4, 2, 1, bias=True)  # 12채널: flow_0_1(2), flow_1_2(2), mask0(1), mask1(1), residual0(3), residual2(3)
        )
    def forward(self, ft_, f0, f1, up_flow0, up_flow1):
        f0_warp = warp(f0, up_flow0)
        f1_warp = warp(f1, up_flow1)
        f_in = torch.cat([ft_, f0_warp, f1_warp, up_flow0, up_flow1], 1)
        f_out = self.convblock(f_in)
        return f_out

class Model(nn.Module):
    def __init__(self, local_rank=-1, lr=1e-4):
        super(Model, self).__init__()
        self.encoder = Encoder()
        self.decoder4 = Decoder4()
        self.decoder3 = Decoder3()
        self.decoder2 = Decoder2()
        self.decoder1 = Decoder1()
        self.l1_loss = Charbonnier_L1()
        self.tr_loss = Ternary(7)
        self.rb_loss = Charbonnier_Ada()
        self.gc_loss = Geometry(3)

    def inference(self, img_0, img_1, embt, scale_factor=1.0):
        img_0_ = resize(img_0, scale_factor=scale_factor)
        img_1_ = resize(img_1, scale_factor=scale_factor)

        f0_1, f0_2, f0_3, f0_4 = self.encoder(img_0_)
        f1_1, f1_2, f1_3, f1_4 = self.encoder(img_1_)

        out4 = self.decoder4(f0_4, f1_4, embt)
        up_flow0_4 = out4[:, 0:2]
        up_flow1_4 = out4[:, 2:4]
        ft_3_ = out4[:, 4:]

        out3 = self.decoder3(ft_3_, f0_3, f1_3, up_flow0_4, up_flow1_4)
        up_flow0_3 = out3[:, 0:2] + 2.0 * resize(up_flow0_4, scale_factor=2.0)
        up_flow1_3 = out3[:, 2:4] + 2.0 * resize(up_flow1_4, scale_factor=2.0)
        ft_2_ = out3[:, 4:]

        out2 = self.decoder2(ft_2_, f0_2, f1_2, up_flow0_3, up_flow1_3)
        up_flow0_2 = out2[:, 0:2] + 2.0 * resize(up_flow0_3, scale_factor=2.0)
        up_flow1_2 = out2[:, 2:4] + 2.0 * resize(up_flow1_3, scale_factor=2.0)
        ft_1_ = out2[:, 4:]

        out1 = self.decoder1(ft_1_, f0_1, f1_1, up_flow0_2, up_flow1_2)
        # 0:2 = flow_0_1, 2:4 = flow_1_2, 4:5 = mask0, 5:6 = mask1, 6:9 = residual0, 9:12 = residual2
        flow_0_1 = out1[:, 0:2] + 2.0 * resize(up_flow0_2, scale_factor=2.0)
        flow_1_2 = out1[:, 2:4] + 2.0 * resize(up_flow1_2, scale_factor=2.0)
        mask0 = torch.sigmoid(out1[:, 4:5])
        mask1 = torch.sigmoid(out1[:, 5:6])
        residual0 = out1[:, 6:9]
        residual2 = out1[:, 9:12]

        # 예측: img1, img2'
        img1_pred_warp = warp(img_0, flow_0_1)
        img2_pred_warp = warp(img_1, flow_1_2)
        img1_pred = mask0 * img1_pred_warp + (1 - mask0) * img_0 + residual0
        img2_pred = mask1 * img2_pred_warp + (1 - mask1) * img_1 + residual2
        img1_pred = torch.clamp(img1_pred, 0, 1)
        img2_pred = torch.clamp(img2_pred, 0, 1)
        return img1_pred, img2_pred, flow_0_1, flow_1_2, mask0, mask1, residual0, residual2, img1_pred_warp, img2_pred_warp

    def forward(self, img_0, img_1, embt, gt_1, gt_2):
        # gt_1: ground truth for 1, gt_2: ground truth for 2
        img1_pred, img2_pred, flow_0_1, flow_1_2, mask0, mask1, residual0, residual2, img1_pred_warp, img2_pred_warp = self.inference(img_0, img_1, embt)
        loss_1 = 0.25 * self.l1_loss(img1_pred - gt_1) + self.tr_loss(img1_pred, gt_1) + 0.75 * self.l1_loss(img1_pred_warp - gt_1)
        loss_2 = 0.25 * self.l1_loss(img2_pred - gt_2) + self.tr_loss(img2_pred, gt_2) + 0.75 * self.l1_loss(img2_pred_warp - gt_2)
        loss = 1.25 * loss_1 + 0.75 * loss_2
        return img1_pred, img2_pred, flow_0_1, flow_1_2, mask0, mask1, residual0, residual2, loss 

def main():
    model = Model()
    img0 = torch.randn(1, 3, 256, 256)
    img1 = torch.randn(1, 3, 256, 256)
    embt = torch.randn(1, 1, 1, 1)

    imgt_pred = model.inference(img0, img1, embt)
    print(imgt_pred[0].shape)


if __name__ == "__main__":
    main()
