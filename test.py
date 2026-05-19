import os
import sys
import time
import argparse
import glob

import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

from model import TextFuse as create_model
from prompt_dataset import PromptDataSet
from utils import tokenize_text_batch, load_text_encoder


@torch.no_grad()
def evaluate(model, tokenizer, data_loader, device):
    model.eval()
    data_loader = tqdm(data_loader, file=sys.stdout)
    model_to_run = model.module if hasattr(model, "module") else model

    for image_t2, pathology_text, ultrasound_text, name, t1_y_image, t1_cb_image, t1_cr_image, h, w, flag in data_loader:
        pathology_tokens = tokenize_text_batch(tokenizer, pathology_text, device)
        ultrasound_tokens = tokenize_text_batch(tokenizer, ultrasound_text, device)

        t1_y_image = t1_y_image.to(device)
        image_t2 = image_t2.to(device)
        t1_cb_image = t1_cb_image.to(device)
        t1_cr_image = t1_cr_image.to(device)

        fused = model_to_run(t1_y_image, image_t2, pathology_tokens, ultrasound_tokens)

        fused_img = clamp(fused)
        fused_img_tensor = YCrCb2RGB(fused_img[0], t1_cb_image[0], t1_cr_image[0])
        fused_img = transforms.ToPILImage()(fused_img_tensor)

        resize_flag = flag.item() if isinstance(flag, torch.Tensor) else flag
        width = w.item() if isinstance(w, torch.Tensor) else w
        height = h.item() if isinstance(h, torch.Tensor) else h

        if resize_flag == 1:
            fused_img = transforms.ToPILImage()(fused_img_tensor).resize((width, height), resample=Image.BICUBIC)

        save_path = "./results/MSRS"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        fused_img.save(os.path.join(save_path, name[0]))

    return 0


def YCrCb2RGB(Y, Cb, Cr):
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


def clamp(value, min=0.0, max=1.0):
    return torch.clamp(value, min=min, max=max)


def resolve_checkpoint_path(requested_path):
    if requested_path and os.path.exists(requested_path):
        return requested_path

    candidate_patterns = [
        "./experiments/*/weights/checkpoint_lastest.pth",
        "./experiments/*/weights/checkpoint.pth",
        "./checkpoint/checkpoint.pth",
    ]

    candidates = []
    for pattern in candidate_patterns:
        candidates.extend(glob.glob(pattern))

    if candidates:
        candidates.sort(key=os.path.getmtime, reverse=True)
        return candidates[0]

    raise FileNotFoundError(
        "No checkpoint file was found. "
        "Pass --checkpoint-path explicitly or place weights under ./experiments/.../weights/."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-model-name", default="bert-base-uncased", help="huggingface text encoder name")
    parser.add_argument("--text-model-path", default="./bert-base-uncased", help="local path to a downloaded Hugging Face text encoder")
    parser.add_argument("--checkpoint-path", default="./checkpoint/checkpoint.pth", help="checkpoint path")
    args = parser.parse_args()

    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = PromptDataSet("test")

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=1,
        drop_last=True,
    )
    tokenizer, model_clip, text_model_source = load_text_encoder(
        args.text_model_name,
        device,
        explicit_path=args.text_model_path,
    )
    print("Using text encoder source: {}".format(text_model_source))
    model = create_model(model_clip).to(device)
    model_weight_path = resolve_checkpoint_path(args.checkpoint_path)
    print("Using checkpoint: {}".format(model_weight_path))
    model.load_state_dict(torch.load(model_weight_path, map_location=device, weights_only=False)["model"], strict=False)
    model.eval()

    for param in model.model_clip.parameters():
        param.requires_grad = False

    evaluate(model=model, tokenizer=tokenizer, data_loader=test_loader, device=device)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.6f} s")