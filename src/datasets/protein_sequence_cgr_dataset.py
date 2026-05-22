from torch.utils.data import Dataset
from utils import utils, nn_utils
from PIL import Image
import torch
from torchvision import transforms
torch.hub.list('pytorch/vision', force_reload=True)
import numpy as np
import os


class ProteinSequenceCGRDataset(Dataset):
    def __init__(self, df, id_col, label_col, img_dir, img_size):
        super(ProteinSequenceCGRDataset, self).__init__()
        self.id_col = id_col
        self.label_col = label_col
        self.img_dir = img_dir
        self.data = df
        self.transforms = transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor()
        ])

    def __len__(self) -> int:
        return self.data.shape[0]

    def get_image(self, id):
        image_filepath = os.path.join(self.img_dir, f"{id}.png")
        # mode 1: 1-bit pixels, black and white, stored with one pixel per byte
        # mode L: 8-bit pixels, grayscale. But this option considers the n_mlp_layers (brightness) of each pixel and will have a value in the range 0-1
        # in our case, we only have points and it is binary - presence or absence of a point, hence mode=1
        # the other modes have multiple channels (rgb: 3 x img_size x img_size)
        return Image.open(image_filepath)

    def get_labels(self):
        return self.data[self.label_col]

    def __getitem__(self, idx: int):
        # loc selects based on index in df
        # iloc selects based on integer location (0, 1, 2, ...)
        record = self.data.iloc[idx, :]

        image = self.get_image(record[self.id_col])
        image_tensor = self.transforms(image)
        # image_tensor is of the shape 4 x img_size x img_size
        # sum across the three channels
        image_tensor = torch.sum(image_tensor, dim=0, keepdim=True)
        # the image tensor is of the shape 1 x img_size x img_size because we have read it with mode=1
        # squeeze the tensor to drop the first dimension
        # image_tensor = image_tensor.squeeze()

        label = record[self.label_col]
        label_vector = np.array([label])

        return image_tensor.to(nn_utils.get_device()), \
               torch.tensor(label_vector, device=nn_utils.get_device()).squeeze()
