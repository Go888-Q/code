from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import glob
import os

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


def load_aligned_t1_t2_images(t1_path, t2_path, resize_to=None):
    t1_image = Image.open(t1_path).convert(mode="RGB")
    t2_image = Image.open(t2_path).convert("L")

    target_size = t1_image.size if resize_to is None else resize_to

    if t1_image.size != target_size:
        t1_image = t1_image.resize(target_size, resample=Image.BICUBIC)
    if t2_image.size != target_size:
        t2_image = t2_image.resize(target_size, resample=Image.BICUBIC)

    return t1_image, t2_image


to_tensor = transforms.Compose([transforms.ToTensor()])


class PromptDataSet(Dataset):
    def __init__(self, split):
        super(PromptDataSet, self).__init__()
        assert split in ["train", "eval", "test"], 'split must be "train"|"eval"|"test"'
        self.transform = to_tensor
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
        t1_path = self.filepath_t1[index]
        t2_path = self.filepath_t2[index]
        pathology_text_path = self.filepath_pathology_text[index]
        ultrasound_text_path = self.filepath_ultrasound_text[index]

        pathology_text = open(pathology_text_path).readline()
        ultrasound_text = open(ultrasound_text_path).readline()

        if self.split == "train":
            t1_image, t2_image = load_aligned_t1_t2_images(t1_path, t2_path)
            image_t1 = self.transform(t1_image)
            image_t2 = self.transform(t2_image)
            t1_y_image, t1_cb_image, t1_cr_image = RGB2YCrCb(image_t1)
            return image_t2, pathology_text, ultrasound_text, t1_y_image, t1_cb_image, t1_cr_image

        if self.split == "eval":
            t1_image, t2_image = load_aligned_t1_t2_images(t1_path, t2_path)
            image_t1 = self.transform(t1_image)
            image_t2 = self.transform(t2_image)
            t1_y_image, t1_cb_image, t1_cr_image = RGB2YCrCb(image_t1)
            name = self.filenames_t1[index]
            return image_t2, pathology_text, ultrasound_text, name, t1_y_image, t1_cb_image, t1_cr_image

        t1_image = Image.open(t1_path).convert(mode="RGB")
        width = t1_image.width
        height = t1_image.height

        new_width = int(round(width // 8) * 8)
        new_height = int(round(height // 8) * 8)
        target_size = (new_width, new_height)

        if new_width == width and new_height == height:
            flag = 0
            aligned_t1, aligned_t2 = load_aligned_t1_t2_images(t1_path, t2_path, resize_to=t1_image.size)
        else:
            flag = 1
            aligned_t1, aligned_t2 = load_aligned_t1_t2_images(t1_path, t2_path, resize_to=target_size)

        image_t1 = self.transform(aligned_t1)
        image_t2 = self.transform(aligned_t2)
        t1_y_image, t1_cb_image, t1_cr_image = RGB2YCrCb(image_t1)
        name = self.filenames_t1[index]

        return image_t2, pathology_text, ultrasound_text, name, t1_y_image, t1_cb_image, t1_cr_image, height, width, flag

    def __len__(self):
        return self.length