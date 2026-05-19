import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp

class fusion_prompt_loss(nn.Module):
    def __init__(self):
        super(fusion_prompt_loss, self).__init__()
        self.fusion_loss = fusion_loss()

    def forward(self, vi, ir, image_fused, vi_mask, ir_mask):
        total_loss, loss_ssim, loss_ssim_mask, loss_consist, loss_consist_mask, loss_text, loss_text_mask = self.fusion_loss( vi, ir, image_fused, vi_mask, ir_mask)

        return total_loss, loss_ssim, loss_ssim_mask, loss_consist, loss_consist_mask, loss_text, loss_text_mask, 


class fusion_loss(nn.Module):
    def __init__(self):
        super(fusion_loss, self).__init__()
        self.loss_func_ssim = L_SSIM(window_size=11)
        self.loss_func_Grad = GradientMaxLoss()
        self.loss_func_Consist = L_Intensity_Consist()
        self.loss_func_Grad_mask = GradientMaxLoss_mask()
        self.loss_func_Consist_mask = L_Intensity_Consist_mask()


    def forward(self, image_visible, image_infrared, image_fused, vi_mask, ir_mask):
        image_vi_mask_cut = torch.mul(image_visible, vi_mask)
        image_ir_mask_cut = torch.mul(image_infrared, ir_mask)

        outputs_vi_mask_cut = torch.mul(image_fused, vi_mask)
        outputs_ir_mask_cut = torch.mul(image_fused, ir_mask)




        loss_ssim       =  self.loss_func_ssim(image_visible, image_fused) + self.loss_func_ssim(image_infrared, image_fused)
        loss_ssim_mask  = self.loss_func_ssim(image_vi_mask_cut, outputs_vi_mask_cut) +  self.loss_func_ssim(image_ir_mask_cut, outputs_ir_mask_cut)

        loss_consist        =  self.loss_func_Consist(image_visible, image_infrared, image_fused)
        loss_consist_mask   =  self.loss_func_Consist_mask(image_vi_mask_cut, image_ir_mask_cut, outputs_vi_mask_cut, outputs_ir_mask_cut )

        loss_text = self.loss_func_Grad(image_visible, image_infrared, image_fused)
        loss_text_mask = self.loss_func_Grad_mask(image_vi_mask_cut, image_ir_mask_cut, outputs_vi_mask_cut, outputs_ir_mask_cut )


        total_loss = loss_ssim + loss_ssim_mask + loss_consist + loss_consist_mask + loss_text + loss_text_mask
        
        return total_loss, loss_ssim, loss_ssim_mask, loss_consist, loss_consist_mask, loss_text, loss_text_mask

    def rgb2gray(self, image):
        b, c, h, w = image.size()
        if c == 1:
            return image
        image_gray = 0.299 * image[:, 0, :, :] + 0.587 * image[:, 1, :, :] + 0.114 * image[:, 2, :, :]
        image_gray = image_gray.unsqueeze(dim=1)
        return image_gray



class L_Intensity_Consist(nn.Module):
    def __init__(self):
        super(L_Intensity_Consist, self).__init__()

    def forward(self, image_visible, image_infrared, image_fused, ir_compose =1 , consist_mode="l1"):
        if consist_mode == "l2":
            Loss_intensity = (F.mse_loss(image_visible, image_fused) + ir_compose * F.mse_loss(image_infrared, image_fused))/2
        else:
            Loss_intensity = (F.l1_loss(image_visible, image_fused) + ir_compose * F.l1_loss(image_infrared, image_fused))/2
        return Loss_intensity


class L_Intensity_Consist_mask(nn.Module):
    def __init__(self):
        super(L_Intensity_Consist_mask, self).__init__()

    def forward(self, image_visible, image_infrared, image_fuse_vi_mask, image_fuse_ir_mask , ir_compose =1 , consist_mode="l1"):
        if consist_mode == "l2":
            Loss_intensity = (F.mse_loss(image_visible, image_fuse_vi_mask) + ir_compose * F.mse_loss(image_infrared, image_fuse_ir_mask))/2
        else:
            Loss_intensity = (F.l1_loss(image_visible, image_fuse_vi_mask) + ir_compose * F.l1_loss(image_infrared, image_fuse_ir_mask))/2
        return Loss_intensity


class GradientMaxLoss(nn.Module):
    def __init__(self):
        super(GradientMaxLoss, self).__init__()
        self.sobel_x = nn.Parameter(torch.FloatTensor([[-1, 0, 1],
                                                       [-2, 0, 2],
                                                       [-1, 0, 1]]).view(1, 1, 3, 3), requires_grad=False).cuda()
        self.sobel_y = nn.Parameter(torch.FloatTensor([[-1, -2, -1],
                                                       [0, 0, 0],
                                                       [1, 2, 1]]).view(1, 1, 3, 3), requires_grad=False).cuda()
        self.padding = (1, 1, 1, 1)

    def forward(self, image_A, image_B, image_fuse):
        gradient_A_x, gradient_A_y = self.gradient(image_A)
        gradient_B_x, gradient_B_y = self.gradient(image_B)
        gradient_fuse_x, gradient_fuse_y = self.gradient(image_fuse)
        loss = F.l1_loss(gradient_fuse_x, torch.max(gradient_A_x, gradient_B_x)) + F.l1_loss(gradient_fuse_y, torch.max(gradient_A_y, gradient_B_y))
        return loss

    def gradient(self, image):
        image = F.pad(image, self.padding, mode='replicate')
        gradient_x = F.conv2d(image, self.sobel_x, padding=0)
        gradient_y = F.conv2d(image, self.sobel_y, padding=0)
        return torch.abs(gradient_x), torch.abs(gradient_y)

class GradientMaxLoss_mask(nn.Module):
    def __init__(self):
        super(GradientMaxLoss_mask, self).__init__()

    def forward(self, image_vi, image_ir, image_fuse_vi_mask, image_fuse_ir_mask ):
        gradient_vi = self.gradient_mask(image_vi)
        gradient_ir = self.gradient_mask(image_ir)
        gradient_fuse_vi = self.gradient_mask(image_fuse_vi_mask)
        gradient_fuse_ir = self.gradient_mask(image_fuse_ir_mask)
        loss = F.l1_loss(gradient_vi, gradient_fuse_vi) + F.l1_loss(gradient_ir, gradient_fuse_ir)

        return loss

    def gradient_mask(self,image):
        filter1 = nn.Conv2d(kernel_size=3, in_channels=1, out_channels=1, bias=False, padding=1, stride=1)
        filter2 = nn.Conv2d(kernel_size=3, in_channels=1, out_channels=1, bias=False, padding=1, stride=1)
        filter1.weight.data = torch.tensor([    [-1., 0., 1.],
                                                [-2., 0., 2.],
                                                [-1., 0., 1.]    ]).reshape(1, 1, 3, 3).cuda()
        filter2.weight.data = torch.tensor([    [1., 2., 1.],
                                                [0., 0., 0.],
                                                [-1., -2., -1.]  ]).reshape(1, 1, 3, 3).cuda()
        g1 = filter1(image)
        g2 = filter2(image)
        image_gradient = torch.abs(g1) + torch.abs(g2)
        return image_gradient


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, window=None, size_average=True, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    return 1 - ret


class L_SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(L_SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        (_, channel_2, _, _) = img2.size()

        if channel != channel_2 and channel == 1:
            img1 = torch.concat([img1, img1, img1], dim=1)
            channel = 3

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window.cuda()
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window.cuda()
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)
