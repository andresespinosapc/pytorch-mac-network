import argparse
import os

import numpy as np
import torch
import cv2

from torchvision import transforms

from src.i3dpt import I3D

rgb_pt_checkpoint = 'model/model_rgb.pth'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DataSet(object):
    def __init__(self, path, frames):
        self.path = path
        self.num_frames = frames

        self.name_clss = {}
        self.data = []
        self.target = []

        self.norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        self.transform = None
        self.target_transform = None

        self.getPaths()

    def getPaths(self):
        with open(os.path.join(self.path,'val_videofolder.txt')) as f: 
            for line in f: 
                name, n_frame, target = line.strip().split()
                
                self.data.append(name)
                self.target.append(int(target))

        # for elem in os.listdir(os.path.join(self.path, 'val')):
        #     video_elem = [ v.split('.')[0] for v in os.listdir(os.path.join(self.path, 'val', elem)) ]
        #     for i,video in enumerate(self.data):
        #         if video in video_elem:
        #             self.name_clss[self.target[i]] = elem
        #             break

        for elem in self.data:
            self.name_clss[elem] = [ os.path.join(self.path, 'frames', elem, f) for f in os.listdir(os.path.join(self.path, 'frames', elem)) ]

    def __getitem__(self, index):
        video, target = self.data[index], self.target[index]

        #sample = self.load_video(os.path.join(self.path,'val',self.name_clss[target],video+'.mp4'))
        sample = self.load_images(self.name_clss[video])

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.data)

    def load_images(self, list_frames, resize=(224, 224)):
        s_frames = np.random.uniform(0, len(list_frames), self.num_frames)

        frames = []
        for s in s_frames:
            frame = cv2.imread(list_frames[s])
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(torch.from_numpy(frame).float() / 255)

        return (frames).permute(3,0,1,2) 

    def sample_frame(self, frames):
        s_frames = np.random.uniform(0, len(frames), self.num_frames)
        return torch.tensor(frames, dtype=torch.float )[s_frames]

    def load_video(self, path, resize=(224, 224)):
        cap = cv2.VideoCapture(path)
        frames = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, resize)
                frame = frame[:, :, [2, 1, 0]]
                frames.append(frame)
          
                #if len(frames) == self.max_frames:
                #    break
        finally:
            cap.release()
        frames = self.sample_frame(frames)
        return (frames /255).permute(3,0,1,2) 


def test(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    for i, (input, target) in enumerate(data_loader):
        input, target = input.to(device), target.to(device)
        output = model(input)[0]
        correct += (output.max(dim=1)[1] == target).data.sum()
        total += target.size(0)
        print("Iteracion[%d/%d] Acc: %f" %(i, len(data_loader), (correct.item() / total)*100))
    return correct.item() / len(data_loader.dataset)

def run_demo(args):

    dataset = DataSet('/mnt/nas/GrimaRepo/datasets/kinetics-400/', 16)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=False)


    i3d_rgb = I3D(num_classes=400, modality='rgb')
    i3d_rgb.eval()
    i3d_rgb.load_state_dict(torch.load(args.rgb_weights_path))
    i3d_rgb.to(device)

    print(test(i3d_rgb, loader))

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
    args = parser.parse_args()
    run_demo(args)
