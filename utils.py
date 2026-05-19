import os
import sys

import torch
import torch.distributed as dist
from torch.cuda.amp import autocast
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from losses import fusion_prompt_loss


def tokenize_text_batch(tokenizer, texts, device, max_length=512):
    encoded = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {key: value.to(device) for key, value in encoded.items()}


def resolve_text_model_source(model_name_or_path, explicit_path=""):
    candidates = []
    if explicit_path:
        candidates.append(explicit_path)

    env_path = os.environ.get("TEXTFUSE_TEXT_MODEL_PATH", "").strip()
    if env_path:
        candidates.append(env_path)

    if model_name_or_path:
        candidates.append(model_name_or_path)
        candidates.append(os.path.join(".", model_name_or_path))
        candidates.append(os.path.join(".", "models", model_name_or_path))

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate, True

    return model_name_or_path, False


def load_text_encoder(model_name_or_path, device, explicit_path="", load_on_cpu=False):
    source, is_local = resolve_text_model_source(model_name_or_path, explicit_path=explicit_path)
    model_device = torch.device("cpu") if load_on_cpu else device

    try:
        tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=is_local)
        model = AutoModel.from_pretrained(source, local_files_only=is_local).to(model_device)
        return tokenizer, model, source
    except OSError as exc:
        hint = (
            "Unable to load the text encoder. "
            f"Tried source: '{source}'. "
            "For offline servers, download a Hugging Face model directory in advance and pass it with "
            "--text-model-path or set TEXTFUSE_TEXT_MODEL_PATH."
        )
        raise RuntimeError(hint) from exc


def reduce_tensor(value, world_size):
    if world_size < 2:
        return value

    reduced = value.detach().clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= world_size
    return reduced


def train_one_epoch(model, tokenizer, optimizer, lr_scheduler, data_loader, device, epoch, is_main_process=True, world_size=1, scaler=None, use_amp=False):
    model.train()
    loss_function_prompt = fusion_prompt_loss().to(device)

    accu_total_loss = torch.zeros(1).to(device)
    accu_ssim_loss = torch.zeros(1).to(device)
    accu_consist_loss = torch.zeros(1).to(device)
    accu_grad_loss = torch.zeros(1).to(device)

    optimizer.zero_grad()
    data_loader = tqdm(data_loader, file=sys.stdout) if is_main_process else data_loader

    for image_t2, pathology_text, ultrasound_text, t1_y_image, t1_cb_image, t1_cr_image in data_loader:
        pathology_tokens = tokenize_text_batch(tokenizer, pathology_text, device)
        ultrasound_tokens = tokenize_text_batch(tokenizer, ultrasound_text, device)

        t1_y_image = t1_y_image.to(device)
        image_t2 = image_t2.to(device)

        with autocast(enabled=use_amp):
            image_fused = model(t1_y_image, image_t2, pathology_tokens, ultrasound_tokens)
            loss, loss_ssim, loss_consist, loss_grad = loss_function_prompt(t1_y_image, image_t2, image_fused)

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        accu_total_loss += loss.detach()
        accu_ssim_loss += loss_ssim.detach()
        accu_consist_loss += loss_consist.detach()
        accu_grad_loss += loss_grad.detach()

        lr = optimizer.param_groups[0]["lr"]

        if is_main_process:
            data_loader.desc = "[train epoch {}] loss: {:.3f}  ssim: {:.3f}  consist: {:.3f}  grad: {:.3f}  lr: {:.6f}".format(
                epoch,
                accu_total_loss.item(),
                accu_ssim_loss.item(),
                accu_consist_loss.item(),
                accu_grad_loss.item(),
                lr,
            )

        if not torch.isfinite(loss):
            print("WARNING: non-finite loss, ending training ", loss)
            sys.exit(1)

        if scaler is not None and use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    accu_total_loss = reduce_tensor(accu_total_loss, world_size)
    accu_ssim_loss = reduce_tensor(accu_ssim_loss, world_size)
    accu_consist_loss = reduce_tensor(accu_consist_loss, world_size)
    accu_grad_loss = reduce_tensor(accu_grad_loss, world_size)

    return accu_total_loss.item(), accu_ssim_loss.item(), accu_consist_loss.item(), accu_grad_loss.item(), lr


@torch.no_grad()
def evaluate(model, tokenizer, data_loader, device, epoch, lr, filefold_path, is_main_process=True):
    loss_function_prompt = fusion_prompt_loss().to(device)
    model.eval()

    accu_total_loss = torch.zeros(1).to(device)
    accu_ssim_loss = torch.zeros(1).to(device)
    accu_consist_loss = torch.zeros(1).to(device)
    accu_grad_loss = torch.zeros(1).to(device)

    evalfold_path = os.path.join(filefold_path, str(epoch))
    if os.path.exists(evalfold_path) is False:
        os.makedirs(evalfold_path)

    data_loader = tqdm(data_loader, file=sys.stdout) if is_main_process else data_loader

    for image_t2, pathology_text, ultrasound_text, name, t1_y_image, t1_cb_image, t1_cr_image in data_loader:
        pathology_tokens = tokenize_text_batch(tokenizer, pathology_text, device)
        ultrasound_tokens = tokenize_text_batch(tokenizer, ultrasound_text, device)

        t1_y_image = t1_y_image.to(device)
        image_t2 = image_t2.to(device)
        t1_cb_image = t1_cb_image.to(device)
        t1_cr_image = t1_cr_image.to(device)

        with autocast(enabled=(device.type == "cuda")):
            image_fused = model(t1_y_image, image_t2, pathology_tokens, ultrasound_tokens)
            fused_img = clamp(image_fused)
            loss, loss_ssim, loss_consist, loss_grad = loss_function_prompt(t1_y_image, image_t2, image_fused)

        fused_img = YCrCb2RGB(fused_img[0], t1_cb_image[0], t1_cr_image[0])
        fused_img = transforms.ToPILImage()(fused_img)
        fused_img.save(os.path.join(evalfold_path, name[0]))

        accu_total_loss += loss.detach()
        accu_ssim_loss += loss_ssim.detach()
        accu_consist_loss += loss_consist.detach()
        accu_grad_loss += loss_grad.detach()

        if is_main_process:
            data_loader.desc = "[eval epoch {}] loss:{:.3f}  ssim:{:.3f}  consist:{:.3f}  grad:{:.3f}  lr:{:.6f}".format(
                epoch,
                accu_total_loss.item(),
                accu_ssim_loss.item(),
                accu_consist_loss.item(),
                accu_grad_loss.item(),
                lr,
            )

    return accu_total_loss.item(), accu_ssim_loss.item(), accu_consist_loss.item(), accu_grad_loss.item(), lr


def create_lr_scheduler(optimizer, num_step: int, epochs: int, warmup=True, warmup_epochs=1, warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def RGB2YCrCb(rgb_image):
    r_channel = rgb_image[0:1]
    g_channel = rgb_image[1:2]
    b_channel = rgb_image[2:3]
    y_channel = 0.299 * r_channel + 0.587 * g_channel + 0.114 * b_channel
    cr_channel = (r_channel - y_channel) * 0.713 + 0.5
    cb_channel = (b_channel - y_channel) * 0.564 + 0.5

    y_channel = clamp(y_channel)
    cr_channel = clamp(cr_channel)
    cb_channel = clamp(cb_channel)
    return y_channel, cb_channel, cr_channel


def YCrCb2RGB(y_channel, cb_channel, cr_channel):
    ycrcb = torch.cat([y_channel, cr_channel, cb_channel], dim=0)
    channels, width, height = ycrcb.shape
    im_flat = ycrcb.reshape(3, -1).transpose(0, 1)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(y_channel.device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(y_channel.device)
    temp = (im_flat + bias).mm(mat)
    out = temp.transpose(0, 1).reshape(channels, width, height)
    out = clamp(out)
    return out


def clamp(value, min=0.0, max=1.0):
    return torch.clamp(value, min=min, max=max)