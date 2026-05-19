import os
import sys
import torch
from tqdm import tqdm
from torchvision import transforms
from prompt_dataset import PromptDataSet
from model import TextFuse as create_model
from PIL import Image
import time
from transformers import AutoModel, AutoTokenizer


def tokenize_text_batch(tokenizer, texts, device, max_length=512):
    encoded = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    return {key: value.to(device) for key, value in encoded.items()}

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
        I_fused = model_to_run(vis_y_image, image_ir, vis_text, ir_text, vis_target_text)

        # from thop import profile
        # flops, params = profile(model, inputs=(vis_y_image, image_ir,vis_text, ir_text), verbose=True)
        # print('thop: FLOPs = ' + str(flops / 1000 ** 2) + 'M')
        # print('thop: Params = ' + str(params / 1000 ** 2) + 'M')

        fused_img = clamp(I_fused)
        fused_img_tensor = YCrCb2RGB(fused_img[0], vis_cb_image[0], vis_cr_image[0])
        fused_img = transforms.ToPILImage()(fused_img_tensor)
        
        
        resize_flag = flag.item() if isinstance(flag, torch.Tensor) else flag
        width = w.item() if isinstance(w, torch.Tensor) else w
        height = h.item() if isinstance(h, torch.Tensor) else h

        if resize_flag == 1:
            fused_img = transforms.ToPILImage()(fused_img_tensor).resize((width, height), resample=Image.BICUBIC)


        
        save_path = "./results/MSRS"
        if not os.path.exists(save_path):#妫€鏌ョ洰褰曟槸鍚﹀瓨鍦?
                os.makedirs(save_path)


        fused_img.save(os.path.join(save_path, name[0]))

    return 0




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



if __name__ == '__main__':

    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = PromptDataSet("test")

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=1,
                                             drop_last=True)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model_clip = AutoModel.from_pretrained("bert-base-uncased").to(device)
    model = create_model(model_clip).to(device)
    model_weight_path = "./checkpoint/checkpoint.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device, weights_only=False)['model'], strict=False)
    model.eval()

    for param in model.model_clip.parameters():
        param.requires_grad = False
    
    evaluate(model=model, tokenizer=tokenizer, data_loader=test_loader, device=device)
    end_time = time.time()
    elapsed_time = end_time - start_time  # 璁＄畻杩愯鏃堕棿锛堢锛?
    print(f"绋嬪簭杩愯鏃堕棿: {elapsed_time:.6f} 绉?)
