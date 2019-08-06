import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import UCF101
from models.rgb_resnet import rgb_resnet50


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def resnet50_single_frame():
    with torch.no_grad():
        params = torch.load(config.model_path, map_location=device)
        spatial_net = rgb_resnet50(pretrained=False, num_classes=101)
        spatial_net.load_state_dict(params['state_dict'])
        spatial_net.to(device)
        spatial_net.eval()

        ucf101 = UCF101(root=config.data_dir, split_file_path=config.split_file_path)
        train_set = DataLoader(
            ucf101, batch_size=config.batch_size, num_workers=4)
        dataset = iter(train_set)
        pbar = tqdm(dataset)

        n_correct = 0
        n_total = 0
        pred_idxs = []
        for frames, action in pbar:
            frames, action = frames.to(device), action.to(device)

            out = spatial_net(frames.view(-1, *frames.shape[2:])).view(*frames.shape[:2], -1)
            out = out.cpu().numpy()
            avg_pred = np.mean(out, axis=1)
            pred_idx = np.argmax(avg_pred, axis=1)
            pred_idxs.extend(pred_idx)
            
            action = action.cpu().numpy()
            n_correct += np.count_nonzero(pred_idx == action)
            n_total += pred_idx.shape[0]

        acc = n_correct / n_total
        print('Accuracy: {:.4f}'.format(acc))
        with open(os.path.join(config.out_dir, 'ucf101_rgb_resnet50.npy'), 'w') as f:
            f.write('\n'.join(map(lambda x: str(x), pred_idxs)))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train Actions Transformations')
    parser.add_argument('data_dir', help='the path to the frames')
    parser.add_argument('split_file_path', help='Path to split file')
    parser.add_argument('model_path', help='Path to the resnet model')
    parser.add_argument('out_dir', help='Path to output predictions')
    parser.add_argument('--batch_size', default=70)
    config = parser.parse_args()

    resnet50_single_frame()
