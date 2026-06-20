# Data Preparation

DGG-HMR follows the [SAT-HMR](https://github.com/ChiSu001/SAT-HMR) data preparation protocol for the shared datasets, except that Human3.6M is not used. We do not redistribute dataset images or preprocessed annotations.

## Dataset Root

Place all datasets under `${Project}/data` by default. You can change the dataset root in `${Project}/configs/paths.py`.

## Training Datasets

The default training config uses AGORA, BEDLAM 1fps, COCO, MPII, and CrowdPose:

- [AGORA](https://agora.is.tue.mpg.de/index.html): use the 1280x720 images.
- [BEDLAM](https://bedlam.is.tue.mpg.de/index.html): the default config uses `bedlam_smpl_train_1fps.npz`; use the 6fps annotation file if computational resources are sufficient.
- [COCO](https://cocodataset.org/#home): use the 2017 train images.
- [MPII](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/software-and-datasets/mpii-human-pose-dataset): prepare the images following the official dataset structure.
- [CrowdPose](https://github.com/Jeff-sjtu/CrowdPose): prepare the images following the official dataset structure.

## Evaluation Datasets

The evaluation configs use AGORA, 3DPW, MuPoTS-3D, and CMU Panoptic:

- [AGORA](https://agora.is.tue.mpg.de/index.html): prepare validation/test images and the corresponding SMPL-format annotations.
- [3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/license.html): prepare the official image files and gendered SMPL annotations.
- MuPoTS-3D: download the official Multi-Person Test Set and follow the official instructions. DGG-HMR reads the original `TS1` to `TS20` sequence folders with `annot.mat`, `occlusion.mat`, and frame images.
- CMU Panoptic: download the official CMU Panoptic data and follow the official instructions. DGG-HMR expects ROMP-style processed annotation files under `data/cmu/processed/annotations`.

## Expected Directory Structure

After preparation, the dataset root should look like this:

```text
${Project}
`-- data
    |-- 3dpw
    |   |-- imageFiles
    |   |-- annots_smpl_train_genders.npz
    |   `-- annots_smpl_test_genders.npz
    |-- agora
    |   |-- train
    |   |-- validation
    |   |-- test
    |   `-- smpl_neutral_annots
    |       |-- annots_smpl_train_fit.npz
    |       |-- annots_smpl_validation.npz
    |       `-- annots_smpl_test.npz
    |-- bedlam
    |   |-- train
    |   |-- validation
    |   |-- bedlam_smpl_train_1fps.npz
    |   |-- bedlam_smpl_train_6fps.npz
    |   `-- bedlam_smpl_validation_6fps.npz
    |-- coco
    |   |-- train2017
    |   `-- COCO_small_NA_SMPL.npz
    |-- crowdpose
    |   |-- images
    |   `-- CP_NA_SMPL_train.npz
    |-- mpii
    |   |-- images
    |   `-- MPII_NA_SMPL.npz
    |-- mupots
    |   `-- MultiPersonTestSet
    |       |-- TS1
    |       |   |-- annot.mat
    |       |   |-- occlusion.mat
    |       |   |-- img_000000.jpg
    |       |   `-- ...
    |       |-- ...
    |       `-- TS20
    `-- cmu
        `-- processed
            |-- annotations
            |   `-- *.pkl
            `-- ...
```

The annotation filenames above are the filenames expected by the current dataset loaders. If you use a different preprocessing pipeline, please either export the same filenames and fields or update the corresponding dataset loader.
