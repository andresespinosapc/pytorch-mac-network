import torch
import numpy as np
import pickle

import torchvision
import h5py

class FeatureLoader(torch.utils.data.Dataset):

    def __init__(self, path_features, get_item_id=False, split='train'):
        self.path_features = path_features

        self.get_item_id = get_item_id
        self.split=split

        self.data = h5py.File(self.path_features, 'r')[self.split]

        self.unique_target = []
        for elem in self.data['target']:
            if elem not in self.unique_target:
                self.unique_target.append(elem)

    def __getitem__(self, index):
        """
        [!] FPS jittering doesn't work with AV dataloader as of now
        """

        item = self.data['data'][index]
        target = self.unique_target.index(self.data['target'][index])
        ind = self.data['video_id'][index]

        # format data to torch
        if self.get_item_id:
            return (item, target, ind)
        else:
            return (item, target)

    def __len__(self):
        return len(self.data['target'])


if __name__ == '__main__':

    loader = FeatureLoader(path_features="/workspace1/jahurtado/20BN-SOMETHING-SOMETHING-V2-1/s2s_feats_10percent.h5")

    import time
    from tqdm import tqdm

    batch_loader = torch.utils.data.DataLoader(
        loader,
        batch_size=128, shuffle=False,
        num_workers=0, pin_memory=True)

    start = time.time()
    for i, a in enumerate(batch_loader):
        print(a[0].shape)
        print(a[1].shape)
        #if i > 10:
        #    break
        pass
    print("Size --> {}".format(a[0].size()))
    print(time.time() - start)
