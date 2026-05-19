import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp


class fusion_prompt_loss(nn.Module):
    def __init__(self):
        super(fusion_prompt_loss, self).__init__()
        self.fusion_loss = fusion_loss()

    def forward(self, image_t1, image_t2, image_fused):
        total_loss, loss_ssim, loss_consist, loss_grad = self.fusion_loss(image_t1, image_t2, image_fused)
        return total_loss, loss_ssim, loss_consist, loss_grad


class fusion_loss(nn.Module):
    def __init__(self):
        super(fusion_loss, self).__init__()
        self.loss_func_ssim = L_SSIM(window_size=11)
        self.loss_func_grad = GradientMaxLoss()
        self.loss_func_consist = L_Intensity_Consist()

    def forward(self, image_t1, image_t2, image_fused):
        loss_ssim = self.loss_func_ssim(image_t1, image_fused) + self.loss_func_ssim(image_t2, image_fused)
        loss_consist = self.loss_func_consist(image_t1, image_t2, image_fused)
        loss_grad = self.loss_func_grad(image_t1, image_t2, image_fused)

        total_loss = loss_ssim + loss_consist + loss_grad
        return total_loss, loss_ssim, loss_consist, loss_grad


class L_Intensity_Consist(nn.Module):
    def __init__(self):
        super(L_Intensity_Consist, self).__init__()

    def forward(self, image_t1, image_t2, image_fused, t2_weight=1, consist_mode="l1"):
        if consist_mode == "l2":
            loss_intensity = (F.mse_loss(image_t1, image_fused) + t2_weight * F.mse_loss(image_t2, image_fused)) / 2
        else:
            loss_intensity = (F.l1_loss(image_t1, image_fused) + t2_weight * F.l1_loss(image_t2, image_fused)) / 2
        return loss_intensity


class GradientMaxLoss(nn.Module):
    def __init__(self):
        super(GradientMaxLoss, self).__init__()
        sobel_x = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
        sobel_y = torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)
        self.padding = (1, 1, 1, 1)

    def forward(self, image_t1, image_t2, image_fuse):
        gradient_t1_x, gradient_t1_y = self.gradient(image_t1)
        gradient_t2_x, gradient_t2_y = self.gradient(image_t2)
        gradient_fuse_x, gradient_fuse_y = self.gradient(image_fuse)
        loss = F.l1_loss(gradient_fuse_x, torch.max(gradient_t1_x, gradient_t2_x))
        loss += F.l1_loss(gradient_fuse_y, torch.max(gradient_t1_y, gradient_t2_y))
        return loss

    def gradient(self, image):
        image = F.pad(image, self.padding, mode="replicate")
        gradient_x = F.conv2d(image, self.sobel_x, padding=0)
        gradient_y = F.conv2d(image, self.sobel_y, padding=0)
        return torch.abs(gradient_x), torch.abs(gradient_y)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    window_1d = gaussian(window_size, 1.5).unsqueeze(1)
    window_2d = window_1d.mm(window_1d.t()).float().unsqueeze(0).unsqueeze(0)
    window = window_2d.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, val_range=None):
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        dynamic_range = max_val - min_val
    else:
        dynamic_range = val_range

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

    c1 = (0.01 * dynamic_range) ** 2
    c2 = (0.03 * dynamic_range) ** 2

    v1 = 2.0 * sigma12 + c2
    v2 = sigma1_sq + sigma2_sq + c2
    ssim_map = ((2 * mu1_mu2 + c1) * v1) / ((mu1_sq + mu2_sq + c1) * v2)

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
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        (_, channel_2, _, _) = img2.size()

        if channel != channel_2 and channel == 1:
            img1 = torch.concat([img1, img1, img1], dim=1)
            channel = 3

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window.to(img1.device)
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)