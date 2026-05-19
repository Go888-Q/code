import os
import sys
import clip
import torch
from tqdm import tqdm
from torchvision import transforms
from prompt_dataset import PromptDataSet
from model import TextFuse as create_model
from PIL import Image
import time

@torch.no_grad()
def evaluate(model, model_clip, data_loader, device):
    model.eval()
    model_clip.eval()
    data_loader = tqdm(data_loader, file=sys.stdout)
    model_to_run = model.module if hasattr(model, "module") else model

    for batch in data_loader:
        if len(batch) == 11:
            image_ir, vis_text, ir_text, name, vis_y_image, vis_cb_image, vis_cr_image, h, w, flag, vis_target_text = batch
            vis_target_text = clip.tokenize(vis_target_text).to(device)
        else:
            image_ir, vis_text, ir_text, name, vis_y_image, vis_cb_image, vis_cr_image, h, w, flag = batch
            vis_target_text = None

        vis_text = clip.tokenize(vis_text).to(device)
        ir_text = clip.tokenize(ir_text).to(device)
        
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
    model_clip, _ = clip.load("ViT-B/32", device=device)
    model = create_model(model_clip).to(device)
    model_weight_path = "./checkpoint/checkpoint.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device, weights_only=False)['model'])
    model.eval()

    for param in model.model_clip.parameters():
        param.requires_grad = False
    
    evaluate(model=model, model_clip=model_clip, data_loader=test_loader, device=device)
    end_time = time.time()
    elapsed_time = end_time - start_time  # 璁＄畻杩愯鏃堕棿锛堢锛?
    print(f"绋嬪簭杩愯鏃堕棿: {elapsed_time:.6f} 绉?)
