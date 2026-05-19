import os
import sys
import time
import argparse

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

    for batch in data_loader:
        if len(batch) == 11:
            image_ir, vis_text, ir_text, name, vis_y_image, vis_cb_image, vis_cr_image, h, w, flag, vis_target_text = batch
            vis_target_text = tokenize_text_batch(tokenizer, vis_target_text, device)
        else:
            image_ir, vis_text, ir_text, name, vis_y_image, vis_cb_image, vis_cr_image, h, w, flag = batch
            vis_target_text = None

        vis_text = tokenize_text_batch(tokenizer, vis_text, device)
        ir_text = tokenize_text_batch(tokenizer, ir_text, device)

        vis_y_image = vis_y_image.to(device)
        image_ir = image_ir.to(device)
        vis_cb_image = vis_cb_image.to(device)
        vis_cr_image = vis_cr_image.to(device)

        fused = model_to_run(vis_y_image, image_ir, vis_text, ir_text, vis_target_text)

        fused_img = clamp(fused)
        fused_img_tensor = YCrCb2RGB(fused_img[0], vis_cb_image[0], vis_cr_image[0])
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-model-name", default="bert-base-uncased", help="huggingface text encoder name")
    parser.add_argument("--text-model-path", default="", help="local path to a downloaded Hugging Face text encoder")
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
    model_weight_path = args.checkpoint_path
    model.load_state_dict(torch.load(model_weight_path, map_location=device, weights_only=False)["model"], strict=False)
    model.eval()

    for param in model.model_clip.parameters():
        param.requires_grad = False

    evaluate(model=model, tokenizer=tokenizer, data_loader=test_loader, device=device)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.6f} s")
