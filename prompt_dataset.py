from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import glob
import os
import torch

from utils import RGB2YCrCb


DATASET_ROOT = "/data/wangjiaqi/fusion"
MRI_T1_IMAGE_DIR = os.path.join(DATASET_ROOT, "MRI-T1")
MRI_T2_IMAGE_DIR = os.path.join(DATASET_ROOT, "MRI-T2")
PATHOLOGY_TEXT_DIR = os.path.join(DATASET_ROOT, "Pathology_Orders")
ULTRASOUND_TEXT_DIR = os.path.join(DATASET_ROOT, "Ultrasound_Orders")


def prepare_data_path(dataset_path):
    filenames = os.listdir(dataset_path)
    data = glob.glob(os.path.join(dataset_path, "*.jpg"))
    data.extend(glob.glob(os.path.join(dataset_path, "*.png")))
    data.extend(glob.glob(os.path.join(dataset_path, "*.txt")))
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
        assert split in ["train", "eval", "test"], 'split must be "train"|"eval"|"test"'
        self.transform = to_tensor
        self.filepath_t1_mask = []
        self.filepath_t2_mask = []
        self.split = split

        self.filepath_t1, self.filenames_t1 = prepare_data_path(MRI_T1_IMAGE_DIR)
        self.filepath_t2, self.filenames_t2 = prepare_data_path(MRI_T2_IMAGE_DIR)
        self.filepath_pathology_text, self.filenames_pathology_text = prepare_data_path(PATHOLOGY_TEXT_DIR)
        self.filepath_ultrasound_text, self.filenames_ultrasound_text = prepare_data_path(ULTRASOUND_TEXT_DIR)

        self.length = min(
            len(self.filenames_t1),
            len(self.filenames_t2),
            len(self.filenames_pathology_text),
            len(self.filenames_ultrasound_text),
        )

    def __getitem__(self, index):
        if self.split == "train":
            t1_path = self.filepath_t1[index]
            t2_path = self.filepath_t2[index]
            pathology_text_path = self.filepath_pathology_text[index]
            ultrasound_text_path = self.filepath_ultrasound_text[index]

            t1_mask_path = self.filepath_t1_mask[index] if index < len(self.filepath_t1_mask) else None
            t2_mask_path = self.filepath_t2_mask[index] if index < len(self.filepath_t2_mask) else None

            t1_image = Image.open(t1_path).convert(mode="RGB")
            t2_image = Image.open(t2_path).convert("L")
            image_t1 = self.transform(t1_image)
            image_t2 = self.transform(t2_image)
            image_t1_mask = load_mask_or_default(t1_mask_path, t1_image)
            image_t2_mask = load_mask_or_default(t2_mask_path, t2_image)
            pathology_text = open(pathology_text_path).readline()
            ultrasound_text = open(ultrasound_text_path).readline()

            t1_y_image, t1_cb_image, t1_cr_image = RGB2YCrCb(image_t1)

            return image_t2, pathology_text, ultrasound_text, image_t1_mask, image_t2_mask, t1_y_image, t1_cb_image, t1_cr_image

        elif self.split == "eval":
            t1_path = self.filepath_t1[index]
            t2_path = self.filepath_t2[index]
            pathology_text_path = self.filepath_pathology_text[index]
            ultrasound_text_path = self.filepath_ultrasound_text[index]

            t1_mask_path = self.filepath_t1_mask[index] if index < len(self.filepath_t1_mask) else None
            t2_mask_path = self.filepath_t2_mask[index] if index < len(self.filepath_t2_mask) else None
            name = self.filenames_t1[index]

            t1_image = Image.open(t1_path).convert(mode="RGB")
            t2_image = Image.open(t2_path).convert("L")
            image_t1 = self.transform(t1_image)
            image_t2 = self.transform(t2_image)
            image_t1_mask = load_mask_or_default(t1_mask_path, t1_image)
            image_t2_mask = load_mask_or_default(t2_mask_path, t2_image)
            pathology_text = open(pathology_text_path).readline()
            ultrasound_text = open(ultrasound_text_path).readline()

            t1_y_image, t1_cb_image, t1_cr_image = RGB2YCrCb(image_t1)
            return image_t2, pathology_text, ultrasound_text, image_t1_mask, image_t2_mask, name, t1_y_image, t1_cb_image, t1_cr_image

        elif self.split == "test":
            t1_path = self.filepath_t1[index]
            t2_path = self.filepath_t2[index]
            pathology_text_path = self.filepath_pathology_text[index]
            ultrasound_text_path = self.filepath_ultrasound_text[index]
            name = self.filenames_t1[index]

            width = Image.open(t1_path).width
            height = Image.open(t1_path).height
            pathology_text = open(pathology_text_path).readline()
            ultrasound_text = open(ultrasound_text_path).readline()
            target_text = pathology_text

            new_width = int(round(width // 8) * 8)
            new_height = int(round(height // 8) * 8)

            if new_width == width and new_height == height:
                flag = 0
                image_t1 = self.transform(Image.open(t1_path).convert(mode="RGB"))
                image_t2 = self.transform(Image.open(t2_path).convert("L"))
                t1_y_image, t1_cb_image, t1_cr_image = RGB2YCrCb(image_t1)
            else:
                flag = 1
                image_t1 = self.transform(
                    Image.open(t1_path).convert(mode="RGB").resize((new_width, new_height), resample=Image.BICUBIC)
                )
                image_t2 = self.transform(
                    Image.open(t2_path).convert("L").resize((new_width, new_height), resample=Image.BICUBIC)
                )
                t1_y_image, t1_cb_image, t1_cr_image = RGB2YCrCb(image_t1)

            return image_t2, pathology_text, ultrasound_text, name, t1_y_image, t1_cb_image, t1_cr_image, height, width, flag, target_text

    def __len__(self):
        return self.length
