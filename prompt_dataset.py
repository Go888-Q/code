
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import glob
import os
import torch
from utils import RGB2YCrCb

DATASET_ROOT = "/data/wangjiaqi/fusion"
VIS_IMAGE_DIR = os.path.join(DATASET_ROOT, "MRI-T1")
IR_IMAGE_DIR = os.path.join(DATASET_ROOT, "MRI-T2")
VIS_TEXT_DIR = os.path.join(DATASET_ROOT, "Pathology_Orders")
IR_TEXT_DIR = os.path.join(DATASET_ROOT, "Ultrasound_Orders")


def prepare_data_path(dataset_path):
    filenames = os.listdir(dataset_path)
    data_dir = dataset_path
    data = glob.glob(os.path.join(data_dir, "*.jpg"))
    data.extend(glob.glob(os.path.join(data_dir, "*.png")))
    data.extend(glob.glob(os.path.join(data_dir, "*.txt")))
    data.sort()
    filenames.sort()
    return data, filenames


def load_mask_or_default(mask_path, reference_image):
    if mask_path and os.path.exists(mask_path):
        return to_tensor(Image.open(mask_path).convert("L"))
    return torch.ones((1, reference_image.height, reference_image.width), dtype=torch.float32)


to_tensor = transforms.Compose([transforms.ToTensor()])

class PromptDataSet(Dataset):
    def __init__(self, split):
        super(PromptDataSet, self).__init__()
        assert split in ['train', 'eval', 'test'], 'split must be "train"|"eval"|"test"'
        self.transform = to_tensor
        self.filepath_vis_mask = []
        self.filepath_ir_mask = []

        if split == 'train':
            data_dir_vis = VIS_IMAGE_DIR
            data_dir_ir = IR_IMAGE_DIR
            data_dir_vis_text = VIS_TEXT_DIR
            data_dir_ir_text = IR_TEXT_DIR

            self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
            self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
            self.filepath_vis_text, self.filenames_vis_text = prepare_data_path(data_dir_vis_text)
            self.filepath_ir_text, self.filenames_ir_text = prepare_data_path(data_dir_ir_text)

            self.split = split
            self.length = min(len(self.filenames_vis), len(self.filenames_ir),
                              len(self.filenames_vis_text), len(self.filenames_ir_text))

        elif split == 'eval':
            data_dir_vis = VIS_IMAGE_DIR
            data_dir_ir = IR_IMAGE_DIR
            data_dir_vis_text = VIS_TEXT_DIR
            data_dir_ir_text = IR_TEXT_DIR

            self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
            self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
            self.filepath_vis_text, self.filenames_vis_text = prepare_data_path(data_dir_vis_text)
            self.filepath_ir_text, self.filenames_ir_text = prepare_data_path(data_dir_ir_text)

            self.split = split
            self.length = min(len(self.filenames_vis), len(self.filenames_ir),
                              len(self.filenames_vis_text), len(self.filenames_ir_text))

        elif split == 'test':
            data_dir_vis = VIS_IMAGE_DIR
            data_dir_ir = IR_IMAGE_DIR
            data_dir_vis_text = VIS_TEXT_DIR
            data_dir_ir_text = IR_TEXT_DIR
            self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
            self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
            self.filepath_vis_text, self.filenames_vis_text = prepare_data_path(data_dir_vis_text)
            self.filepath_ir_text, self.filenames_ir_text = prepare_data_path(data_dir_ir_text)

            self.split = split
            self.length = min(len(self.filenames_vis), len(self.filenames_ir),
                              len(self.filenames_vis_text), len(self.filenames_ir_text))

    def __getitem__(self, index):
        if self.split=='train':
            vis_path        = self.filepath_vis[index]
            ir_path         = self.filepath_ir[index]
            
            vis_path_text = self.filepath_vis_text[index]
            ir_path_text = self.filepath_ir_text[index]
            
            vis_path_mask = self.filepath_vis_mask[index] if index < len(self.filepath_vis_mask) else None
            ir_path_mask = self.filepath_ir_mask[index] if index < len(self.filepath_ir_mask) else None

            vis_image = Image.open(vis_path).convert(mode='RGB')
            ir_image = Image.open(ir_path).convert('L')
            image_vis = self.transform(vis_image)
            image_ir = self.transform(ir_image)
            image_vis_mask = load_mask_or_default(vis_path_mask, vis_image)
            image_ir_mask = load_mask_or_default(ir_path_mask, ir_image)
            vis_text = open(vis_path_text).readline()
            ir_text = open(ir_path_text).readline()

            vis_y_image, vis_cb_image, vis_cr_image = RGB2YCrCb(image_vis)


            return image_ir, vis_text, ir_text, image_vis_mask, image_ir_mask ,vis_y_image, vis_cb_image, vis_cr_image

        elif self.split == 'eval':
            vis_path        = self.filepath_vis[index]
            ir_path         = self.filepath_ir[index]
            vis_path_text = self.filepath_vis_text[index]
            ir_path_text = self.filepath_ir_text[index]
            vis_path_mask = self.filepath_vis_mask[index] if index < len(self.filepath_vis_mask) else None
            ir_path_mask = self.filepath_ir_mask[index] if index < len(self.filepath_ir_mask) else None
            name = self.filenames_vis[index]

            vis_image = Image.open(vis_path).convert(mode='RGB')
            ir_image = Image.open(ir_path).convert('L')
            image_vis = self.transform(vis_image)
            image_ir = self.transform(ir_image)
            image_vis_mask = load_mask_or_default(vis_path_mask, vis_image)
            image_ir_mask = load_mask_or_default(ir_path_mask, ir_image)
            
            vis_text = open(vis_path_text).readline()
            ir_text = open(ir_path_text).readline()
                             
            vis_y_image, vis_cb_image, vis_cr_image = RGB2YCrCb(image_vis)
            return  image_ir, vis_text, ir_text, image_vis_mask, image_ir_mask, name ,vis_y_image, vis_cb_image, vis_cr_image


        elif self.split=='test':
            vis_path = self.filepath_vis[index]
            ir_path = self.filepath_ir[index]
            vis_path_text = self.filepath_vis_text[index]
            ir_path_text = self.filepath_ir_text[index]
            name = self.filenames_vis[index]
            w = Image.open(vis_path).width  # 鍥剧墖鐨勫
            h = Image.open(vis_path).height  # 鍥剧墖鐨勯珮
            vis_text = open(vis_path_text).readline()
            ir_text = open(ir_path_text).readline()
            vis_target_text = vis_text

            new_w = int(round(w // 8) * 8)
            new_h = int(round(h // 8) * 8)

            if new_w == w and new_h == h:
                flag = 0
                image_vis = self.transform(Image.open(vis_path).convert(mode='RGB'))
                image_ir = self.transform(Image.open(ir_path).convert('L'))
                vis_y_image, vis_cb_image, vis_cr_image = RGB2YCrCb(image_vis)

            else:
                flag = 1
                image_vis = self.transform(Image.open(vis_path).convert(mode='RGB').resize((new_w, new_h), resample=Image.BICUBIC))
                image_ir = self.transform(Image.open(ir_path).convert('L').resize((new_w, new_h), resample=Image.BICUBIC))
                vis_y_image, vis_cb_image, vis_cr_image = RGB2YCrCb(image_vis)

            return  image_ir, vis_text, ir_text,  name ,vis_y_image, vis_cb_image, vis_cr_image ,h ,w,flag, vis_target_text

    def __len__(self):
        return self.length

    
    

