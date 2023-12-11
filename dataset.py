from torch.utils import data
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import math
import os


class IMDBWIKI(data.Dataset):
    def __init__(self, df, data_dir, img_size = 224, split='train', group_num = 10, group_mode = 'i_g', ord_binary = False, reweight = None, max_group=100):
        self.groups = group_num
        self.df = df
        self.data_dir = data_dir
        self.img_size = img_size
        self.split = split    
        self.group_range = max_group/group_num
        #self.group_mode = group_mode
        self.ord_binary = ord_binary
        self.re_weight = reweight
        #self.key_list = [i for i in range(group_num)]
        # key is the group is, value is the group num
        #
        if split == 'train':
            group_dict = {}
            for i in range(len(self.df)):
                row = self.df.iloc[i]
                age = row['age']
                group_id = math.floor(age/self.group_range)
                # put the age 0 into the first group
                if group_id > self.groups - 1:
                    group_id = self.groups - 1
                if group_id in group_dict.keys():
                    group_dict[group_id] += 1
                else:
                    group_dict[group_id] = 1
            list_group = sorted(group_dict.items(), key = lambda group_dict : group_dict[0])
            self.group_list = [i[1] for i in list_group]
            #
            self.weights = self.weights_prepare(reweight=reweight)
        else:
            pass
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        index = index % len(self.df)
        row = self.df.iloc[index]
        img = Image.open(os.path.join(self.data_dir, row['path'])).convert('RGB')
        transform = self.get_transform()
        img = transform(img)
        label = np.asarray([row['age']]).astype('float32')
        group_index = math.floor(label/self.group_range)
        #group_index = np.clip(group_index, 0, self.groups - 1)
        group = np.asarray([group_index]).astype('float32')
        group = np.clip(group, 0, self.groups - 1)
        if self.split == 'train':
            if self.re_weight is not None:
                weight = np.asarray([self.weights[index]]).astype(
                    'float32') if self.weights is not None else np.asarray([np.float32(1.)])
                return img, label, group, weight
            else:
                return img, label, group, 1
        else:
            return img, label, group


    def get_group(self):
        return self.group_list


    def get_transform(self):
        if self.split == 'train':
            transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomCrop(self.img_size, padding=16),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([.5, .5, .5], [.5, .5, .5]),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize([.5, .5, .5], [.5, .5, .5]),
            ])
        return transform


    def weights_prepare(self, reweight='sqrt_inv', max_target=121):
        assert reweight in {None, 'inverse', 'sqrt_inv'}
        #
        value_dict = {x: 0 for x in range(max_target)}
        #
        if reweight is None:
            return None
        #
        labels = self.df['age'].values
        #
        for label in labels:
            value_dict[min(max_target - 1, int(label))] += 1
        if reweight == 'sqrt_inv':
            value_dict = {k: np.sqrt(v) for k, v in value_dict.items()}
        elif reweight == 'inverse':
            # clip weights for inverse re-weight
            value_dict = {k: np.clip(v, 5, 1000)
                          for k, v in value_dict.items()}
        num_per_label = [
            value_dict[min(max_target - 1, int(label))] for label in labels]
        if not len(num_per_label):
            return None
        print(f"Using re-weighting: [{reweight.upper()}]")
        weights = [np.float32(1 / x) for x in num_per_label]
        scaling = len(weights) / np.sum(weights)
        weights = [scaling * x for x in weights]
        return weights