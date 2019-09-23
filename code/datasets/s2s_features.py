import torch.utils.data as data
import numpy as np
import h5py


class S2SFeatureDataset(data.Dataset):

    def __init__(self, path_features, get_item_id=False, split='train'):
        self.path_features = path_features

        self.get_item_id = get_item_id
        self.split=split

        # self.data = h5py.File(self.path_features, 'r')[self.split]

        # self.unique_target = np.unique(self.data['target'])
        with h5py.File(self.path_features, 'r') as f:
            data = f[self.split]
            self.unique_target = list(np.unique(data['target']))

    def __getitem__(self, index):
        """
        [!] FPS jittering doesn't work with AV dataloader as of now
        """

        with h5py.File(self.path_features, 'r') as f:
            data = f[self.split]
            item = data['data'][index]
            if data['target'][index] not in self.unique_target:
                raise IndexError('Target value {} from index {} does not exist'.format(data['target'][index], index))
            target = self.unique_target.index(data['target'][index])
            # where_result = np.where(self.unique_target == data['target'][index])[0]
            # if where_result.shape[0] == 0:
            #     raise ValueError('Index {} is not in unique_target'.format(data['target'][index]))
            # target = where_result[0]
            ind = data['video_id'][index]

        # format data to torch
        if self.get_item_id:
            return (item, target, ind)
        else:
            return (item, target)

    def __len__(self):
        with h5py.File(self.path_features, 'r') as f:
            data = f[self.split]
            return len(data['target'])