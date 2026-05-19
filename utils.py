import os
import sys
import torch
from tqdm import tqdm
from losses import fusion_prompt_loss
from torchvision import transforms

def tokenize_text_batch(tokenizer, texts, device, max_length=512):
    encoded = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    return {key: value.to(device) for key, value in encoded.items()}


def train_one_epoch(model, tokenizer, optimizer, lr_scheduler, data_loader, device, epoch):
    model.train()
    loss_function_prompt = fusion_prompt_loss()
    loss_function_prompt = loss_function_prompt.to(device)

    accu_total_loss = torch.zeros(1).to(device)
    accu_ssim_loss = torch.zeros(1).to(device)
    accu_ssim_loss_mask = torch.zeros(1).to(device)
    accu_loss_consist = torch.zeros(1).to(device)
    accu_loss_consist_mask = torch.zeros(1).to(device)
    accu_text_loss = torch.zeros(1).to(device)
    accu_text_loss_mask = torch.zeros(1).to(device)

    optimizer.zero_grad()

    data_loader = tqdm(data_loader, file=sys.stdout)
    for  image_ir, vis_text, ir_text, image_vis_mask,image_ir_mask,vis_y_image, vis_cb_image, vis_cr_image in data_loader:

        image_vis_text = tokenize_text_batch(tokenizer, vis_text, device)
        image_ir_text = tokenize_text_batch(tokenizer, ir_text, device)

        vis_y_image = vis_y_image.to(device)
        image_ir = image_ir.to(device)
        image_vis_mask = image_vis_mask.to(device)
        image_ir_mask = image_ir_mask.to(device)

        I_fused = model(vis_y_image, image_ir, image_vis_text, image_ir_text)

        loss, loss_ssim, loss_ssim_mask, loss_consist, loss_consist_mask, loss_text, loss_text_mask = \
            loss_function_prompt(vis_y_image, image_ir, I_fused, image_vis_mask, image_ir_mask)

        loss.backward()

        accu_total_loss += loss.detach()
        accu_ssim_loss += loss_ssim.detach()
        accu_ssim_loss_mask += loss_ssim_mask.detach()
        accu_loss_consist += loss_consist.detach()
        accu_loss_consist_mask += loss_consist_mask.detach()
        accu_text_loss += loss_text.detach()
        accu_text_loss_mask += loss_text_mask.detach()


        lr = optimizer.param_groups[0]["lr"]

        data_loader.desc = "[train epoch {}] loss: {:.3f}  ssim: {:.3f}  ssim_mask: {:.3f}  " \
                           "consist: {:.3f}   consist_mask: {:.3f}     text: {:.3f}  text_mask: {:.3f}   lr: {:.6f}".format(epoch, accu_total_loss.item(),
            accu_ssim_loss.item(), accu_ssim_loss_mask.item(), accu_loss_consist.item(), accu_loss_consist_mask.item(),  accu_text_loss.item(), accu_text_loss_mask.item(), lr)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    return accu_total_loss.item(), accu_ssim_loss.item(), accu_ssim_loss_mask.item(), accu_loss_consist.item(), accu_loss_consist_mask.item(), accu_text_loss.item(), accu_text_loss_mask.item(), lr


@torch.no_grad()
def evaluate(model, tokenizer, data_loader, device, epoch, lr, filefold_path):
    loss_function_prompt = fusion_prompt_loss()
    model.eval()

    accu_total_loss = torch.zeros(1).to(device)
    accu_ssim_loss = torch.zeros(1).to(device)
    accu_ssim_loss_mask = torch.zeros(1).to(device)
    accu_loss_consist = torch.zeros(1).to(device)
    accu_loss_consist_mask = torch.zeros(1).to(device)
    accu_text_loss = torch.zeros(1).to(device)
    accu_text_loss_mask = torch.zeros(1).to(device)

    save_epoch = 1

    loss_function_prompt = loss_function_prompt.to(device)
    
    if epoch % save_epoch == 0:
        evalfold_path = os.path.join(filefold_path, str(epoch))
        if os.path.exists(evalfold_path) is False:
            os.makedirs(evalfold_path)

    data_loader = tqdm(data_loader, file=sys.stdout)

    for image_ir, vis_text, ir_text, image_vis_mask, image_ir_mask, name,vis_y_image, vis_cb_image, vis_cr_image in data_loader:

        image_vis_text = tokenize_text_batch(tokenizer, vis_text, device)
        image_ir_text = tokenize_text_batch(tokenizer, ir_text, device)
        
        
        vis_y_image = vis_y_image.to(device)
        image_ir = image_ir.to(device)
        image_vis_mask = image_vis_mask.to(device)
        image_ir_mask = image_ir_mask.to(device)
        vis_cb_image = vis_cb_image.to(device)
        vis_cr_image = vis_cr_image.to(device)

        I_fused = model(vis_y_image, image_ir, image_vis_text, image_ir_text)


        fused_img = clamp(I_fused)
        
        fused_img = YCrCb2RGB(fused_img[0], vis_cb_image[0], vis_cr_image[0])
        fused_img = transforms.ToPILImage()(fused_img)


        fused_img.save(os.path.join(evalfold_path, name[0]))
        


        loss, loss_ssim, loss_ssim_mask, loss_consist, loss_consist_mask, loss_text, loss_text_mask = \
            loss_function_prompt(vis_y_image, image_ir, I_fused, image_vis_mask, image_ir_mask)



        accu_total_loss += loss.detach()
        accu_ssim_loss += loss_ssim.detach()
        accu_ssim_loss_mask += loss_ssim_mask.detach()

        accu_loss_consist += loss_consist.detach()
        accu_loss_consist_mask += loss_consist_mask.detach()

        accu_text_loss += loss_text.detach()
        accu_text_loss_mask += loss_text_mask.detach()


        data_loader.desc = "[eval epoch {}] loss:{:.3f}  ssim:{:.3f}  ssim_mask:{:.3f}  " \
                           "consist:{:.3f} consist_mask:{:.3f}  text:{:.3f}  text_mask:{:.3f} lr:{:.6f}".format( epoch, accu_total_loss.item(),  accu_ssim_loss.item(), accu_ssim_loss_mask.item(), accu_loss_consist.item(), accu_loss_consist_mask.item(), accu_text_loss.item(), accu_text_loss_mask.item(), lr)

    return accu_total_loss.item(), accu_ssim_loss.item(), accu_ssim_loss_mask.item(), accu_loss_consist.item(), accu_loss_consist_mask.item(),accu_text_loss.item(), accu_text_loss_mask.item(), lr


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)






def RGB2YCrCb(rgb_image):
    """
    灏哛GB鏍煎紡杞崲涓篩CrCb鏍煎紡

    :param rgb_image: RGB鏍煎紡鐨勫浘鍍忔暟鎹?    :return: Y, Cr, Cb
    """

    R = rgb_image[0:1]
    G = rgb_image[1:2]
    B = rgb_image[2:3]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5

    Y = clamp(Y)
    Cr = clamp(Cr)
    Cb = clamp(Cb)
    return Y, Cb, Cr


def YCrCb2RGB(Y, Cb, Cr):
    """
    灏哬crCb鏍煎紡杞崲涓篟GB鏍煎紡

    :param Y:
    :param Cb:
    :param Cr:
    :return:
    """
    ycrcb = torch.cat([Y, Cr, Cb], dim=0)
    C, W, H = ycrcb.shape
    im_flat = ycrcb.reshape(3, -1).transpose(0, 1)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(Y.device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(Y.device)
    temp = (im_flat + bias).mm(mat)
    out = temp.transpose(0, 1).reshape(C, W, H)
    out = clamp(out)
    return out


def clamp(value, min=0., max=1.0):
    """
    灏嗗儚绱犲€煎己鍒剁害鏉熷湪[0,1], 浠ュ厤鍑虹幇寮傚父鏂戠偣
    :param value:
    :param min:
    :param max:
    :return:
    """
    return torch.clamp(value, min=min, max=max)
