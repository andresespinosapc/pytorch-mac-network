import av
import torch
import numpy as np

from datasets.data_parser import WebmDataset
from datasets.data_augmentor import Augmentor
import torchvision
from datasets.transforms_video import *
# from utils import save_images_for_debug

import time

FRAMERATE = 12  # default value


class VideoFolder(torch.utils.data.Dataset):

    def __init__(self, root, json_file_input, json_file_labels, clip_size,
                 nclips, step_size, is_val, transform_pre=None, transform_post=None,
                 augmentation_mappings_json=None, augmentation_types_todo=None,
                 get_item_id=False, is_test=False):
        self.dataset_object = WebmDataset(json_file_input, json_file_labels,
                                          root, is_test=is_test)
        self.json_data = self.dataset_object.json_data
        self.classes = self.dataset_object.classes
        self.classes_dict = self.dataset_object.classes_dict
        self.root = root
        self.transform_pre = transform_pre
        self.transform_post = transform_post

        self.augmentor = Augmentor(augmentation_mappings_json,
                                   augmentation_types_todo)

        self.clip_size = clip_size
        self.nclips = nclips
        self.step_size = step_size
        self.is_val = is_val
        self.get_item_id = get_item_id

        #self.get_labels_10_percent()

    def get_labels_10_percent(self):
        self.few_class = []
        with open("/mnt/nas/GrimaRepo/aespinosa/label_names_10percent.txt") as f: 
            for line in f: 
                self.few_class.append(line.strip()) 

    # def __getitem__(self, index):
    #     """
    #     [!] FPS jittering doesn't work with AV dataloader as of now
    #     """

    #     start_time = time.time()
    #     item = self.json_data[index]

    #     # Open video file
    #     reader = av.open(item.path)

    #     try:
    #         imgs = []
    #         imgs = [f.to_rgb().to_ndarray() for f in reader.decode(video=0)]
    #     except (RuntimeError, ZeroDivisionError) as exception:
    #         print('{}: WEBM reader cannot open {}. Empty '
    #               'list returned.'.format(type(exception).__name__, item.path))

    #     reader.close()

    #     print("---Lectura Video  %s seconds ---" % (time.time() - start_time))
    #     start_time = time.time()

    #     imgs = self.transform_pre(imgs)
    #     imgs, label = self.augmentor(imgs, item.label)
    #     imgs = self.transform_post(imgs)

    #     num_frames = len(imgs)
    #     target_idx = self.classes_dict[label]
    #     #target_idx = self.few_class.index(label)

    #     print("---Procesamiento Video  %s seconds ---" % (time.time() - start_time))
    #     start_time = time.time()

    #     if self.nclips > -1:
    #         num_frames_necessary = self.clip_size * self.nclips * self.step_size
    #     else:
    #         num_frames_necessary = num_frames
    #     offset = 0
    #     if num_frames_necessary < num_frames:
    #         # If there are more frames, then sample starting offset.
    #         diff = (num_frames - num_frames_necessary)
    #         # temporal augmentation
    #         if not self.is_val:
    #             offset = np.random.randint(0, diff)

    #     imgs = imgs[offset: num_frames_necessary + offset: self.step_size]

    #     print("---Cortar Video  %s seconds ---" % (time.time() - start_time))
    #     start_time = time.time()

    #     if len(imgs) < (self.clip_size * self.nclips):
    #         imgs.extend([imgs[-1]] *
    #                     ((self.clip_size * self.nclips) - len(imgs)))

    #     # format data to torch
    #     print("---Extender Video  %s seconds ---" % (time.time() - start_time))
    #     start_time = time.time()
    #     data = torch.stack(imgs)
    #     data = data.permute(1, 0, 2, 3)
    #     print("---Stack Video  %s seconds ---" % (time.time() - start_time))
    #     if self.get_item_id:
    #         return (data, target_idx, item.id)
    #     else:
    #         return (data, target_idx)

    def __len__(self):
        return len(self.json_data)


    def __getitem__(self, index):
        item = self.json_data[index]

        start_time = time.time()
        cap = cv2.VideoCapture(item.path)

        imgs = []
        try:
            while True:
                ret, frame = cap.read()

                if not ret:
                    break
                frame = cv2.resize(frame, resize)
                frame = frame[:, :, [2, 1, 0]]
                imgs.append(torch.from_numpy(frame).float() / 255 )

        finally:
            cap.release()

        num_frames = len(imgs)
        target_idx = self.classes_dict[label]
        print("---Lectura Video  %s seconds ---" % (time.time() - start_time))
        start_time = time.time()

        if self.nclips > -1:
            num_frames_necessary = self.clip_size * self.nclips * self.step_size
        else:
            num_frames_necessary = num_frames
        offset = 0
        if num_frames_necessary < num_frames:
            # If there are more frames, then sample starting offset.
            diff = (num_frames - num_frames_necessary)
            # temporal augmentation
            if not self.is_val:
                offset = np.random.randint(0, diff)

        imgs = imgs[offset: num_frames_necessary + offset: self.step_size]

        print("---Cortar Video  %s seconds ---" % (time.time() - start_time))
        start_time = time.time()

        if len(imgs) < (self.clip_size * self.nclips):
            imgs.extend([imgs[-1]] *
                        ((self.clip_size * self.nclips) - len(imgs)))

        # format data to torch
        print("---Extender Video  %s seconds ---" % (time.time() - start_time))


        frames = torch.stack(frames)
        data = (frames).permute(3,0,1,2) 

        if self.get_item_id:
            return (data, target_idx, item.id)
        else:
            return (data, target_idx)



if __name__ == '__main__':
    upscale_size = int(84 * 1.1)
    transform_pre = ComposeMix([
            # [RandomRotationVideo(20), "vid"],
            [Scale(upscale_size), "img"],
            [RandomCropVideo(84), "vid"],
            # [RandomHorizontalFlipVideo(0), "vid"],
            # [RandomReverseTimeVideo(1), "vid"],
            # [torchvision.transforms.ToTensor(), "img"],
             ])
    # identity transform
    transform_post = ComposeMix([
                        [torchvision.transforms.ToTensor(), "img"],
                         ])

    loader = VideoFolder(root="/data-ssd1/20bn-something-something-v2/videos",
                         json_file_input="/data-ssd1/20bn-something-something-v2/annotations/something-something-v2-train.json",
                         json_file_labels="/data-ssd1/20bn-something-something-v2/annotations/something-something-v2-labels.json",
                         clip_size=36,
                         nclips=1,
                         step_size=1,
                         is_val=False,
                         transform_pre=transform_pre,
                         transform_post=transform_post,
                         # augmentation_mappings_json="notebooks/augmentation_mappings.json",
                         # augmentation_types_todo=["left/right", "left/right agnostic", "jitter_fps"],
                         )
    # fetch a sample
    # data_item, target_idx = loader[1]
    # save_images_for_debug("input_images_2", data_item.unsqueeze(0))
    # print("Label = {}".format(loader.classes_dict[target_idx]))

    import time
    from tqdm import tqdm

    batch_loader = torch.utils.data.DataLoader(
        loader,
        batch_size=10, shuffle=False,
        num_workers=8, pin_memory=True)

    start = time.time()
    for i, a in enumerate(tqdm(batch_loader)):
        if i > 100:
            break
        pass
    print("Size --> {}".format(a[0].size()))
    print(time.time() - start)
