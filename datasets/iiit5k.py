import os
import string
import scipy.io
from PIL import Image
import torch
from torch.nn import functional as F
from torchvision.datasets import VisionDataset


class IIIT5K(VisionDataset):
    """IIIT5K Dataset"""
    MAX_LABEL_LENGTH = 22
    MAX_WIDTH = 2836
    MAX_HEIGHT = 704
    characters = '-' + string.digits + string.ascii_uppercase

    def __init__(self, root='.datasets', split='train', transform=None,
                 target_transform=None, max_width=128, max_height=32):
        self.root = root
        self.data_root = os.path.join(root, 'IIIT5K')
        self.split = split
        mat_filename = os.path.join(self.data_root, f'{split}data.mat')
        self.data = scipy.io.loadmat(mat_filename)[f'{split}data'][0]
        self.transform = transform
        self.target_transform = target_transform
        self.max_width = max_width
        self.max_height = max_height
    
    @property
    def input_shape(self):
        return [3, self.max_height, self.max_width] # [channels, height, width]
    
    @property
    def num_classes(self):
        return len(self.characters)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_name = self.data[index][0][0]
        img = Image.open(os.path.join(self.data_root, img_name))
        img = img.convert('RGB') # few images in the dataset are grayscales
        width, height = img.size
        ratio = width / height
        new_height = min(int(self.max_width / ratio), self.max_height)
        new_width = min(int(self.max_height * ratio), self.max_width)
        img = img.resize((new_width, new_height), Image.BICUBIC)
        if self.transform:
            img = self.transform(img)
        label = self.data[index][1][0]
        encoded_label = IIIT5K.encode_label(label)
        if self.target_transform:
            encoded_label = self.target_transform(encoded_label)
        target = {
            'label': label,
            'length': len(label),
            'encoded_label': encoded_label}
        return img, target
    
    @staticmethod
    def encode_label(label):
        """Encode label for pytorch model"""
        label = [IIIT5K.characters.index(char) for char in label]
        label = torch.tensor(label, dtype=torch.int32)
        label = F.pad(label, (0, IIIT5K.MAX_LABEL_LENGTH - label.shape[0]))
        return label
    
    @staticmethod
    def decode_label(label):
        """Inverse of encode label"""
        label = ''.join(IIIT5K.characters[idx] for idx in label)
        label = label.rstrip('-')
        return label
    
    def decode_predictions(self, logits):
        """Decode model predictions with 'best path decoding' strategy"""
        labels = []
        batch_size = logits.size(1)
        for i in range(batch_size):
            encoded_label = logits[:,i,:].argmax(dim=-1)
            label = []
            for j in range(IIIT5K.MAX_LABEL_LENGTH):
                if encoded_label[j].item() == 0: continue
                if j > 0 and encoded_label[j] == encoded_label[j-1]: continue
                label.append(IIIT5K.characters[encoded_label[j].item()])
            labels.append(''.join(label))
        return labels
        
    def collate_fn(self, data):
        """Collate function for DataLoader. Pad images to the same shape to enable batching"""
        images = []
        widths = []
        labels = []
        lengths = []
        encoded_labels = []
        for img, target in data:
            _, height, width = img.size()
            pad_height = self.max_height - height
            pad_width = self.max_width - width
            img = F.pad(img, (0, pad_width, 0, pad_height))
            images.append(img)
            widths.append(width)
            labels.append(target['label'])
            lengths.append(target['length'])
            encoded_labels.append(target['encoded_label'])
        images = {
            'data': torch.stack(images),
            'width': torch.tensor(widths)}
        targets = {
            'label': labels,
            'length': torch.tensor(lengths),
            'encoded_label': torch.stack(encoded_labels)}
        return images, targets


def get_dataset_info():
    """Get dataset info only from train split to avoid data leaks"""
    train_dataset = IIIT5K(root='.datasets', split='train')
    max_label_length = 0
    max_height = 0
    max_width = 0
    characters = set()
    for i in range(len(train_dataset)):
        img, target = train_dataset[i]
        max_label_length = max(target['length'], max_label_length)
        width, height = img.size
        max_width = max(width, max_width)
        max_height = max(height, max_height)
        characters |= set(IIIT5K.decode_label(target['label']))
    print(f'Max label length: {max_label_length}')
    print(f'Max width: {max_width}')
    print(f'Max height: {max_height}')
    print(f'Characters: {sorted(characters)}')


if __name__ == '__main__':
    get_dataset_info()