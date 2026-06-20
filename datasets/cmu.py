import os
import pickle
import math
from glob import glob

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from configs.paths import dataset_root
from utils.constants import J24_TO_H36M, H36M_EVAL_JOINTS


ROMP_BASE_RESOLUTION = 832.0
ROMP_MIN_VISIBLE_KPTS = 5


class CMU(Dataset):
    """
    CMU Panoptic dataset loader that mirrors ROMP's load_gts preprocessing.
    It consumes ROMP processed pickle files:
        data/cmu/processed/annotations/*.pkl
    """

    def __init__(self, split='test', input_size=1288, **kwargs):
        super().__init__()
        assert split == 'test', 'Only test split is supported for CMU Panoptic evaluation.'
        self.split = split
        self.input_size = input_size

        self.dataset_root = os.path.join(dataset_root, 'cmu')
        self.processed_root = os.path.join(self.dataset_root, 'processed')
        self.annots_dir = os.path.join(self.processed_root, 'annotations')

        if not os.path.isdir(self.annots_dir):
            raise FileNotFoundError(
                f'Cannot find CMU annotations under {self.annots_dir}. '
                f'Please place ROMP processed Panoptic data in data/cmu/processed.'
            )

        self.total_persons_raw = 0
        self.total_persons_visible = 0
        self.records = self._load_annotations()
        if len(self.records) == 0:
            raise RuntimeError(f'No CMU samples found in {self.annots_dir}.')
        print(
            f"[CMU] Loaded {len(self.records)} images, "
            f"raw persons={self.total_persons_raw}, "
            f"visible persons={self.total_persons_visible}."
        )

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

        patch_size = 14
        pad_h = math.ceil(new_h / patch_size) * patch_size
        pad_w = math.ceil(new_w / patch_size) * patch_size
        canvas = np.zeros((pad_h, pad_w, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = img_resized

        tensor_img = self.transform(Image.fromarray(canvas))

        gt_kp2d = torch.from_numpy(record['kp2d_j14']).clone()
        if gt_kp2d.numel() > 0:
            gt_kp2d[:, :, :2] *= resize_rate
            gt_kp2d_vis = gt_kp2d[:, :, 2] > 0.5
        else:
            gt_kp2d = torch.zeros((0, 14, 3), dtype=torch.float32)
            gt_kp2d_vis = torch.zeros((0, 14), dtype=torch.bool)

        targets = {
            'img_path': record['img_path'],
            'img_size': torch.tensor([new_h, new_w], dtype=torch.int32),
            'gt_kp2d_j14': gt_kp2d[:, :, :2],
            'gt_kp2d_vis_j14': gt_kp2d_vis,
            'gt_kp3d_j14': torch.from_numpy(record['kp3d_j14']),
        }
        return tensor_img, targets

    def _load_annotations(self):
        records = []
        ann_files = sorted(glob(os.path.join(self.annots_dir, '*.pkl')))
        for ann_path in ann_files:
            with open(ann_path, 'rb') as f:
                entries = pickle.load(f)
            for entry in entries:
                img_path = self._resolve_image_path(entry['filename'].replace('\\', '/'))
                if img_path is None:
                    continue
                kp2d = entry.get('kpts2d')
                kp3d = entry.get('kpts3d')
                if kp2d is None or kp3d is None:
                    continue
                kp2d = np.asarray(kp2d, dtype=np.float32)
                kp3d = np.asarray(kp3d, dtype=np.float32)
                self.total_persons_raw += int(kp2d.shape[0])
                if kp2d.ndim != 3 or kp3d.ndim != 3:
                    continue

                kp2d_h36m = kp2d[:, J24_TO_H36M]
                kp3d_h36m = kp3d[:, J24_TO_H36M]
                kp3d_root = kp3d_h36m[:, [0], :3]
                kp3d_h36m = kp3d_h36m.copy()
                kp3d_h36m[:, :, :3] = kp3d_h36m[:, :, :3] - kp3d_root
                kp2d_j14 = kp2d_h36m[:, H36M_EVAL_JOINTS]
                kp3d_j14 = kp3d_h36m[:, H36M_EVAL_JOINTS]
                visible_ids, kp2d_visible = self._determine_visible_person(
                    kp2d_j14,
                    entry.get('width', ROMP_BASE_RESOLUTION),
                    entry.get('height', ROMP_BASE_RESOLUTION),
                )
                if len(visible_ids) == 0:
                    continue
                self.total_persons_visible += int(len(visible_ids))

                kp2d_visible = kp2d_visible.astype(np.float32)
                kp2d_visible_mask = kp2d_visible[:, :, 2] > 0.5
                kp3d_visible = kp3d_j14[visible_ids].astype(np.float32)

                persons_2d_j14 = np.zeros((len(visible_ids), 14, 3), dtype=np.float32)
                persons_3d_j14 = np.zeros((len(visible_ids), 14, 3), dtype=np.float32)
                for pid, (kp2d_person, kp3d_person) in enumerate(zip(kp2d_visible, kp3d_visible)):
                    persons_2d_j14[pid] = kp2d_person
                    coords3d_j14 = kp3d_person[:, :3].copy()
                    coords3d_j14[:13] += np.array([0.0, 0.06, 0.03], dtype=np.float32)
                    invis_3d = kp3d_person[:, -1] < 0.2
                    coords3d_j14[invis_3d] = -2.0
                    if kp2d_visible_mask.shape[0] > pid:
                        invis_2d = ~kp2d_visible_mask[pid]
                        coords3d_j14[invis_2d] = -2.0
                    persons_3d_j14[pid] = coords3d_j14

                records.append({
                    'img_path': img_path,
                    'kp2d_j14': persons_2d_j14,
                    'kp3d_j14': persons_3d_j14,
                })
        return records

    def _resolve_image_path(self, rel_path):
        if os.path.isabs(rel_path):
            candidates = [rel_path]
        else:
            candidates = [
                os.path.join(self.dataset_root, rel_path),
                os.path.join(self.processed_root, rel_path),
            ]
        for path in candidates:
            if os.path.isfile(path):
                return path
        return None

    @staticmethod
    def _determine_visible_person(kp2ds, width, height):
        width = float(width) if width else ROMP_BASE_RESOLUTION
        height = float(height) if height else ROMP_BASE_RESOLUTION
        visible_person_id, kp2d_vis = [], []
        for pid, kp2d in enumerate(kp2ds):
            visible_mask = np.logical_and(
                np.logical_and(0 < kp2d[:, 0], kp2d[:, 0] < width),
                np.logical_and(0 < kp2d[:, 1], kp2d[:, 1] < height),
            )
            visible_mask = np.logical_and(visible_mask, kp2d[:, 2] > 0.2)
            if int(visible_mask.sum()) > ROMP_MIN_VISIBLE_KPTS:
                kp2d_vis.append(np.concatenate([kp2d[:, :2], visible_mask[:, None].astype(np.float32)], axis=1))
                visible_person_id.append(pid)
        return np.array(visible_person_id, dtype=np.int64), np.array(kp2d_vis, dtype=np.float32)



