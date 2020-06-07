import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from sklearn.utils import shuffle


class DatasetGenerator(Dataset):
    def _find_classes(self, dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __init__(self, path_dataset, image_mode, device, transform=None):
        self.list_images_path = []
        self.list_images_label = []
        self.device = device
        self.path_dataset = path_dataset
        self.classes = self._find_classes(self.path_dataset)[0]
        self.image_mode = image_mode
        self.transform = transform
        self.included_extensions = ['jpg', 'jpeg', 'png']
        self.count_classes = len(self.classes)

        for c in range(self.count_classes):
            for f in os.listdir(f'{path_dataset}/{self.classes[c]}'):
                if any(f.endswith(ext) for ext in self.included_extensions):
                    label = [0] * self.count_classes
                    label[c] = 1
                    file = f'{path_dataset}/{self.classes[c]}/{f}'
                    self.list_images_path.append(file)
                    self.list_images_label.append(label)

        # Random shuffle
        self.list_images_path, self.list_images_label = shuffle(self.list_images_path, self.list_images_label,
                                                                random_state=0)

    def __getitem__(self, index):
        image_path = self.list_images_path[index]

        image_data = Image.open(image_path).convert(self.image_mode)

        image_label = torch.FloatTensor(self.list_images_label[index])

        if self.transform is not None: image_data = self.transform(image_data)

        return image_data, image_label

    def __len__(self):
        return len(self.list_images_path)
