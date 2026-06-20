import os
import math
from glob import glob

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset
from torchvision import transforms

from configs.paths import dataset_root


MPII_JOINT_NUM = 17
PATCH_SIZE = 14


class MuPoTS(Dataset):
    """
    MuPoTS-3D evaluation loader that mirrors the official MATLAB evaluation inputs.
    It consumes the original TSV sequences with annot.mat / occlusion.mat under:
        data/mupots-3d/TS{1-20}/
    """

    def __init__(self, split='test', input_size=1288, **kwargs):
        super().__init__()
        assert split == 'test', 'Only test split is supported for MuPoTS.'
        self.split = split
        self.input_size = input_size

        self.dataset_root = os.path.join(dataset_root, 'mupots/MultiPersonTestSet')
        if not os.path.isdir(self.dataset_root):
            raise FileNotFoundError(
                f'Cannot find MuPoTS-3D root under {self.dataset_root}. '
                f'Please place the official MultiPerson Test Set in this folder.'
            )

        self.seq_dirs = sorted(glob(os.path.join(self.dataset_root, 'TS*')))
        if len(self.seq_dirs) == 0:
            raise RuntimeError(f'No MuPoTS sequences found in {self.dataset_root}.')

        self.records = self._load_annotations()
        if len(self.records) == 0:
            raise RuntimeError('No valid MuPoTS samples after filtering.')

        total_persons = sum(rec['kp2d'].shape[0] for rec in self.records)
        print(f"[MuPoTS] Loaded {len(self.records)} images, total persons={total_persons}.")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        img = cv2.imread(record['img_path'])
        if img is None:
            raise FileNotFoundError(f'Fail to read image: {record["img_path"]}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]

        resize_rate = self.input_size / max(orig_h, orig_w)
        new_w = max(1, int(round(orig_w * resize_rate)))
        new_h = max(1, int(round(orig_h * resize_rate)))
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_h = math.ceil(new_h / PATCH_SIZE) * PATCH_SIZE
        pad_w = math.ceil(new_w / PATCH_SIZE) * PATCH_SIZE
        canvas = np.zeros((pad_h, pad_w, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = img_resized

        tensor_img = self.transform(Image.fromarray(canvas))

        persons_2d = torch.from_numpy(record['kp2d']).clone()
        persons_vis = torch.from_numpy(record['vis']).bool()
        persons_occ = torch.from_numpy(record['occ']).bool()
        persons_3d = torch.from_numpy(record['kp3d']).clone()

        if persons_2d.numel() > 0:
            persons_2d *= resize_rate

        targets = {
            'img_path': record['img_path'],
            'sequence': record['sequence'],
            'frame_index': record['frame_index'],
            'img_size': torch.tensor([new_h, new_w], dtype=torch.int32),
            'orig_size': torch.tensor([orig_h, orig_w], dtype=torch.int32),
            'resize_rate': torch.tensor(resize_rate, dtype=torch.float32),
            'gt_kp2d_mpi17': persons_2d,
            'gt_kp2d_vis_mpi17': persons_vis,
            'gt_kp3d_mpi17': persons_3d,
            'gt_joint_occlusion_mpi17': persons_occ,
        }
        return tensor_img, targets

    def _load_annotations(self):
        records = []
        for seq_dir in self.seq_dirs:
            seq_name = os.path.basename(seq_dir)
            annot_file = os.path.join(seq_dir, 'annot.mat')
            occlusion_file = os.path.join(seq_dir, 'occlusion.mat')
            if not (os.path.isfile(annot_file) and os.path.isfile(occlusion_file)):
                continue

            annotations = loadmat(annot_file)['annotations']
            occlusion_labels = loadmat(occlusion_file)['occlusion_labels']
            num_frames = annotations.shape[0]

            for fid in range(num_frames):
                frame_annots = annotations[fid]
                frame_occlusion = occlusion_labels[fid]
                persons_2d, persons_3d = [], []
                persons_vis, persons_occ = [], []

                for pid in range(frame_annots.shape[0]):
                    subj = frame_annots[pid][0, 0]
                    kp2ds, _, univ_kp3ds, valid_flag = subj[0], subj[1], subj[2], subj[3]
                    if not bool(valid_flag[0, 0]):
                        continue

                    kp2d = kp2ds.transpose((1, 0)).astype(np.float32)  # (17,2)
                    kp3d = univ_kp3ds.transpose((1, 0)).astype(np.float32)  # (17,3)
                    occ_mask = frame_occlusion[pid][0].astype(np.float32)  # 1 means occluded
                    vis_mask = (1.0 - occ_mask).astype(np.float32)

                    persons_2d.append(kp2d)
                    persons_3d.append(kp3d)
                    persons_vis.append(vis_mask)
                    persons_occ.append(occ_mask)

                if len(persons_2d) == 0:
                    continue

                records.append({
                    'img_path': os.path.join(seq_dir, f'img_{fid:06d}.jpg'),
                    'sequence': seq_name,
                    'frame_index': fid,
                    'kp2d': np.stack(persons_2d, axis=0),
                    'kp3d': np.stack(persons_3d, axis=0),
                    'vis': np.stack(persons_vis, axis=0),
                    'occ': np.stack(persons_occ, axis=0),
                })
        return records
