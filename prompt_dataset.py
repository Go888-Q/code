
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import glob
import os
from utils import RGB2YCrCb


def prepare_data_path(dataset_path):
    filenames = os.listdir(dataset_path)
    data_dir = dataset_path
    data = glob.glob(os.path.join(data_dir, "*.jpg"))
    data.extend(glob.glob(os.path.join(data_dir, "*.png")))
    data.extend(glob.glob(os.path.join(data_dir, "*.txt")))
    data.sort()
    filenames.sort()
    return data, filenames

to_tensor = transforms.Compose([transforms.ToTensor()])

class PromptDataSet(Dataset):
    def __init__(self, split):
        super(PromptDataSet, self).__init__()
        assert split in ['train', 'eval', 'test'], 'split must be "train"|"eval"|"test"'
        self.transform = to_tensor

        if split == 'train':
            data_dir_vis = 'Data/train/vi'
            data_dir_ir = 'Data/train/ir'
            data_dir_vis_text = 'Data/train/vi_text'
            data_dir_ir_text = 'Data/train/ir_text'
            data_dir_vis_mask = 'Data/train/vi_mask'
            data_dir_ir_mask = 'Data/train/ir_mask'

            self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
            self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
            self.filepath_vis_text, self.filenames_vis_text = prepare_data_path(data_dir_vis_text)
            self.filepath_ir_text, self.filenames_ir_text = prepare_data_path(data_dir_ir_text)
            self.filepath_vis_mask, self.filenames_vis_mask = prepare_data_path(data_dir_vis_mask)
            self.filepath_ir_mask, self.filenames_ir_mask = prepare_data_path(data_dir_ir_mask)

            self.split = split
            self.length = min(len(self.filenames_vis), len(self.filenames_ir))

        elif split == 'eval':
            data_dir_vis = 'Data/train/vi'
            data_dir_ir = 'Data/train/ir'
            data_dir_vis_text = 'Data/train/vi_text'
            data_dir_ir_text = 'Data/train/ir_text'
            data_dir_vis_mask = 'Data/train/vi_mask'
            data_dir_ir_mask = 'Data/train/ir_mask'

            self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
            self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
            self.filepath_vis_text, self.filenames_vis_text = prepare_data_path(data_dir_vis_text)
            self.filepath_ir_text, self.filenames_ir_text = prepare_data_path(data_dir_ir_text)
            self.filepath_vis_mask, self.filenames_vis_mask = prepare_data_path(data_dir_vis_mask)
            self.filepath_ir_mask, self.filenames_ir_mask = prepare_data_path(data_dir_ir_mask)

            self.split = split
            self.length = min(len(self.filenames_vis), len(self.filenames_ir))

        elif split == 'test':


            data_dir_vis = 'MSRS/vi'
            data_dir_ir = 'MSRS/ir'
            data_dir_vis_text = 'MSRS/vi_text'
            data_dir_ir_text = 'MSRS/ir_text'
            self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
            self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
            self.filepath_vis_text, self.filenames_vis_text = prepare_data_path(data_dir_vis_text)
            self.filepath_ir_text, self.filenames_ir_text = prepare_data_path(data_dir_ir_text)

            self.split = split
            self.length = min(len(self.filenames_vis), len(self.filenames_ir))

    def __getitem__(self, index):
        if self.split=='train':
            vis_path        = self.filepath_vis[index]
            ir_path         = self.filepath_ir[index]
            
            vis_path_text = self.filepath_vis_text[index]
            ir_path_text = self.filepath_ir_text[index]
            
            vis_path_mask = self.filepath_vis_mask[index]
            ir_path_mask = self.filepath_ir_mask[index]
            
            image_vis       = self.transform(Image.open(vis_path).convert(mode='RGB') )
            image_ir        = self.transform(Image.open(ir_path).convert('L') )
            image_vis_mask  = self.transform(Image.open(vis_path_mask).convert('L') )
            image_ir_mask   = self.transform(Image.open(ir_path_mask).convert('L') )
            vis_text = open(vis_path_text).readline()
            ir_text = open(ir_path_text).readline()

            vis_y_image, vis_cb_image, vis_cr_image = RGB2YCrCb(image_vis)


            return image_ir, vis_text, ir_text, image_vis_mask, image_ir_mask ,vis_y_image, vis_cb_image, vis_cr_image

        elif self.split == 'eval':
            vis_path        = self.filepath_vis[index]
            ir_path         = self.filepath_ir[index]
            vis_path_text = self.filepath_vis_text[index]
            ir_path_text = self.filepath_ir_text[index]
            vis_path_mask   = self.filepath_vis_mask[index]
            ir_path_mask    = self.filepath_ir_mask[index]
            name = self.filenames_vis[index]

            image_vis       = self.transform(Image.open(vis_path).convert(mode='RGB'))
            image_ir        = self.transform(Image.open(ir_path).convert('L'))
            image_vis_mask  = self.transform(Image.open(vis_path_mask).convert('L'))
            image_ir_mask   = self.transform(Image.open(ir_path_mask).convert('L'))
            
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

            new_w = int(round(w // 8) * 8)
            new_h = int(round(h // 8) * 8)

            if new_w == w and new_h == h:
                flag = 0
                image_vis = self.transform(Image.open(vis_path))
                image_ir = self.transform(Image.open(ir_path).convert('L'))
                vis_y_image, vis_cb_image, vis_cr_image = RGB2YCrCb(image_vis)

            else:
                flag = 1
                image_vis = self.transform(Image.open(vis_path).resize((new_w, new_h), resample=Image.BICUBIC))
                image_ir = self.transform(Image.open(ir_path).convert('L').resize((new_w, new_h), resample=Image.BICUBIC))
                vis_y_image, vis_cb_image, vis_cr_image = RGB2YCrCb(image_vis)

            return  image_ir, vis_text, ir_text,  name ,vis_y_image, vis_cb_image, vis_cr_image ,h ,w,flag

    def __len__(self):
        return self.length

    
    

