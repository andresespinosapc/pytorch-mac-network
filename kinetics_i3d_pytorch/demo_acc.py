import argparse
import os

import numpy as np
import torch

from src.i3dpt import I3D

rgb_pt_checkpoint = 'model/model_rgb.pth'


class DataSet(object):
    def __init__(self, path, frames):
        self.path = path
        self.num_frames = frames

        self.name_clss = {}
        self.data = []
        self.target = []

        self.transform = None
        self.target_transform = None

    def getPaths(self):
        with open(os.path.join(self.path,'val_videofolder.txt')) as f: 
            for line in f: 
                name, n_frame, target = line.strip().split()
                
                self.data.append(name)
                self.target.append(int(target))

        for elem in os.listdir(os.path.join(self.path, 'val')):
            video_elem = [ v for v in os.listdir(os.path.join(self.path, 'val', elem)) ]
            for i,video in enumerate(self.data):
                if video in video_elem:
                    self.name_clss[self.target[i]] = elem
                    break

    def __getitem__(self, index):
        video, target = self.data[index], self.target[index]

        sample = self.load_video(os.path.join(self.path,'val',self.name_clss[target],video))
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.data)

    def sample_frame(self, frames):
        s_frames = np.random.uniform(0, len(frames), self.num_frames)


    def load_video(path, resize=(224, 224)):
        cap = cv2.VideoCapture(path)
        frames = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = crop_center_square(frame)
                frame = cv2.resize(frame, resize)
                frame = frame[:, :, [2, 1, 0]]
                frames.append(frame)
          
                #if len(frames) == self.max_frames:
                #    break
        finally:
            cap.release()
        frames = self.sample_frame(frames)
        return torch.stack(frames) / 255.0

def run_demo(args):
    

    def get_scores(sample, model):
        sample_var = torch.autograd.Variable(torch.from_numpy(sample).cuda())
        out_var, out_logit = model(sample_var)
        out_tensor = out_var.data.cpu()

        top_val, top_idx = torch.sort(out_tensor, 1, descending=True)

        print(
            'Top {} classes and associated probabilities: '.format(args.top_k))
        for i in range(args.top_k):
            print('[{}]: {:.6E}'.format(kinetics_classes[top_idx[0, i]],
                                        top_val[0, i]))
        return out_logit



    dataset = DataSet('/mnt/nas/GrimaRepo/datasets/kinetics-400/', 16)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False)


    # i3d_rgb = I3D(num_classes=400, modality='rgb')
    # i3d_rgb.eval()
    # i3d_rgb.load_state_dict(torch.load(args.rgb_weights_path))
    # i3d_rgb.cuda()

    # rgb_sample = np.load(args.rgb_sample_path).transpose(0, 4, 1, 2, 3)
    # out_rgb_logit = get_scores(rgb_sample, i3d_rgb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Runs inflated inception v1 network on\
    cricket sample from tensorflow demo (generate the network weights with\
    i3d_tf_to_pt.py first)')

    # RGB arguments
    parser.add_argument(
        '--rgb', action='store_true', help='Evaluate RGB pretrained network')
    parser.add_argument(
        '--rgb_weights_path',
        type=str,
        default='model/model_rgb.pth',
        help='Path to rgb model state_dict')
    run_demo(args)
