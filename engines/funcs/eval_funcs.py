import os
from collections import defaultdict
from tqdm.auto import tqdm
import torch
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from utils.evaluation import cal_3d_position_error, match_2d_greedy, get_matching_dict, compute_prf1, vectorize_distance, calculate_iou, select_and_align
from utils.transforms import unNormalize, pelvis_align
from utils.visualization import tensor_to_BGR, pad_img
from utils.visualization import vis_meshes_img, vis_boxes, get_colors_rgb, vis_meshes_topview
from utils.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from utils.constants import H36M_EVAL_JOINTS, J24_TO_H36M, H36M_TO_MPII
import time
import datetime
import scipy.io as sio
import cv2
import zipfile
import pickle

MPII_JOINT_NUM = 17
MPII_PELVIS_IDX = 14
MPII_MATCHING_JOINTS = list(range(1, 14))
MPII_PARENT_O1 = np.array(
    [2, 16, 2, 3, 4, 2, 6, 7, 15, 9, 10, 15, 12, 13, 15, 15, 2],
    dtype=np.int64
) - 1  # convert to 0-based indexing
MPII_PARENT_O1[MPII_PARENT_O1 < 0] = -1
MPII_PARENT_O1[MPII_PELVIS_IDX] = -1
MPII_TRAVERSAL_ORDER = [14, 15, 1, 0, 16, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
MUPOTS_PCK_THRESH_MM = 150.0
MUPOTS_MATCH_THRESH_PX = 40.0


def cal_pcd_error(pred_depths, gt_depths, threshold=0.2):
    """
    Calculate detailed Pairwise Comparison of Depth (PCD) metrics following ROMP logic.
    Categorizes pairs into:
    - EQ (Equal): |gt_i - gt_j| < threshold
    - CD (Closer): gt_i - gt_j < -threshold
    - FD (Farther): gt_i - gt_j > threshold
    
    Returns:
        A dictionary containing overall and binned stats.
    """
    overall_stats = [0, 0, 0, 0, 0, 0] # eq_c, eq_t, cd_c, cd_t, fd_c, fd_t
    bin_stats = {
        '0-2m': [0, 0],
        '2-5m': [0, 0],
        '5-10m': [0, 0],
        '10m+': [0, 0]
    }
    
    n = len(gt_depths)
    if n < 2:
        return {'overall': overall_stats, 'bins': bin_stats}
    
    bins = [0, 2, 5, 10, float('inf')]
    bin_names = ['0-2m', '2-5m', '5-10m', '10m+']

    for i in range(n):
        for j in range(i + 1, n):
            diff_gt = gt_depths[i] - gt_depths[j]
            diff_pred = pred_depths[i] - pred_depths[j]
            
            avg_depth = (gt_depths[i] + gt_depths[j]) / 2.0
            target_bin = None
            for b_idx in range(len(bins)-1):
                if bins[b_idx] <= avg_depth < bins[b_idx+1]:
                    target_bin = bin_names[b_idx]
                    break

            if abs(diff_gt) < threshold:
                overall_stats[1] += 1
                if target_bin: bin_stats[target_bin][1] += 1
                if abs(diff_pred) < threshold:
                    overall_stats[0] += 1
                    if target_bin: bin_stats[target_bin][0] += 1
            elif diff_gt < -threshold:
                overall_stats[3] += 1
                if target_bin: bin_stats[target_bin][1] += 1
                if diff_pred < -threshold:
                    overall_stats[2] += 1
                    if target_bin: bin_stats[target_bin][0] += 1
            elif diff_gt > threshold:
                overall_stats[5] += 1
                if target_bin: bin_stats[target_bin][1] += 1
                if diff_pred > threshold:
                    overall_stats[4] += 1
                    if target_bin: bin_stats[target_bin][0] += 1
                    
    return {'overall': overall_stats, 'bins': bin_stats}



def _mupots_match_identities(gt_kp2d, gt_vis, pred_kp2d, match_thresh=MUPOTS_MATCH_THRESH_PX):
    """
    Greedy matching following mpii_multiperson_get_identity_matching.
    Returns an array of size num_gt with matched pred indices or -1.
    """
    num_gt = len(gt_kp2d)
    num_pred = len(pred_kp2d)
    matches = np.full((num_gt,), -1, dtype=np.int64)
    if num_gt == 0 or num_pred == 0:
        return matches

    pred_assigned = np.zeros((num_pred,), dtype=bool)
    for gid in range(num_gt):
        best_score = 0
        best_idx = -1
        gt_pts = gt_kp2d[gid, MPII_MATCHING_JOINTS]
        gt_vis_mask = gt_vis[gid, MPII_MATCHING_JOINTS].astype(bool)
        if gt_vis_mask.sum() == 0:
            continue

        for pid in range(num_pred):
            if pred_assigned[pid]:
                continue
            diff = np.abs(pred_kp2d[pid, MPII_MATCHING_JOINTS] - gt_pts)
            matches_mask = (diff[..., 0] < match_thresh) & (diff[..., 1] < match_thresh)
            score = int(np.count_nonzero(matches_mask & gt_vis_mask))
            if score > best_score:
                best_score = score
                best_idx = pid

        if best_score > 0 and best_idx >= 0:
            matches[gid] = best_idx
            pred_assigned[best_idx] = True
    return matches


def _mupots_normalize_bone_lengths(pred_joints, gt_joints):
    """
    Map predicted joints to GT bone lengths using traversal order,
    mimicking mpii_map_to_gt_bone_lengths.
    """
    mapped = pred_joints.copy()
    for joint in MPII_TRAVERSAL_ORDER:
        parent = MPII_PARENT_O1[joint]
        if parent < 0:
            continue
        gt_vec = gt_joints[joint] - gt_joints[parent]
        pred_vec = mapped[joint] - mapped[parent]
        gt_len = np.linalg.norm(gt_vec)
        pred_len = np.linalg.norm(pred_vec)
        if gt_len < 1e-6 or pred_len < 1e-6:
            mapped[joint] = mapped[parent]
            continue
        mapped[joint] = mapped[parent] + pred_vec / pred_len * gt_len
    return mapped


# Modified from ROMP Panoptic evaluation
def evaluate_mupots(model, eval_dataloader, conf_thresh,
                    results_save_path=None, distributed=False, accelerator=None,
                    vis_step=None, vis=False):
    assert results_save_path is not None
    assert accelerator is not None

    os.makedirs(results_save_path, exist_ok=True)
    cur_device = next(model.parameters()).device
    smpl_layer = model.human_model
    smpl2h36m_regressor = torch.from_numpy(
        smpl_layer.smpl2h36m_regressor
    ).float().to(cur_device)

    total_gt_persons = torch.zeros((1,), dtype=torch.int64, device=cur_device)
    matched_gt_persons = torch.zeros((1,), dtype=torch.int64, device=cur_device)
    pck_hits_all = torch.zeros((1,), dtype=torch.float64, device=cur_device)
    pck_hits_matched = torch.zeros((1,), dtype=torch.float64, device=cur_device)
    joint_cnt_all = torch.zeros((1,), dtype=torch.float64, device=cur_device)
    joint_cnt_matched = torch.zeros((1,), dtype=torch.float64, device=cur_device)

    progress_bar = tqdm(
        total=len(eval_dataloader),
        disable=not accelerator.is_local_main_process,
        desc='evaluate_mupots',
    )

    for samples, targets in eval_dataloader:
        samples = [sample.to(device=cur_device, non_blocking=True) for sample in samples]
        with torch.no_grad():
            outputs = model(samples, targets)

        bs = len(targets)
        for idx in range(bs):
            gt_kp2d = targets[idx]['gt_kp2d_mpi17'].cpu().numpy()
            gt_vis = targets[idx]['gt_kp2d_vis_mpi17'].cpu().numpy().astype(bool)
            gt_kp3d = targets[idx]['gt_kp3d_mpi17'].cpu().numpy()
            resize_rate = float(targets[idx].get('resize_rate', 1.0))
            if gt_kp3d.shape[0] == 0:
                continue

            gt_kp3d_rel = gt_kp3d - gt_kp3d[:, [MPII_PELVIS_IDX]]
            num_gt = gt_kp3d_rel.shape[0]

            select_queries_idx = torch.where(outputs['pred_confs'][idx] > conf_thresh)[0]
            if len(select_queries_idx) == 0:
                matches = np.full((num_gt,), -1, dtype=np.int64)
                pred_kp3d_rel_mm = np.zeros((0, MPII_JOINT_NUM, 3), dtype=np.float32)
            else:
                pred_verts = outputs['pred_verts'][idx][select_queries_idx].detach()
                pred_transl = outputs['pred_transl'][idx][select_queries_idx].detach()
                pred_intrinsics = outputs['pred_intrinsics'][idx].detach().cpu().numpy().reshape(3, 3)

                pred_j3ds_h36m17 = torch.einsum(
                    'bik,ji->bjk',
                    [pred_verts - pred_transl[:, None, :], smpl2h36m_regressor]
                ) + pred_transl[:, None, :]
                pred_j3ds_h36m17 = pred_j3ds_h36m17.detach().cpu().numpy()

                pred_j3ds_mpi = pred_j3ds_h36m17[:, H36M_TO_MPII]
                pred_kp3d_rel = pred_j3ds_mpi - pred_j3ds_mpi[:, [MPII_PELVIS_IDX]]
                pred_kp3d_rel_mm = pred_kp3d_rel * 1000.0  # convert to millimeters

                pred_j2d_homo = np.matmul(pred_j3ds_h36m17, pred_intrinsics.T)
                pred_j2ds = pred_j2d_homo[:, :, :2] / (pred_j2d_homo[:, :, 2:3] + 1e-6)
                kp2d_preds = pred_j2ds[:, H36M_TO_MPII]

                match_thresh = MUPOTS_MATCH_THRESH_PX * resize_rate
                matches = _mupots_match_identities(gt_kp2d, gt_vis, kp2d_preds, match_thresh=match_thresh)

            total_gt_persons += num_gt
            matched_gt_persons += int(np.count_nonzero(matches >= 0))

            for gid in range(num_gt):
                joint_cnt_all += MPII_JOINT_NUM
                matched_pid = matches[gid]
                if matched_pid < 0:
                    continue

                pred_joints_mm = pred_kp3d_rel_mm[matched_pid]
                gt_joints_mm = gt_kp3d_rel[gid]
                pred_aligned = _mupots_normalize_bone_lengths(pred_joints_mm.copy(), gt_joints_mm)
                joint_errors = np.linalg.norm(pred_aligned - gt_joints_mm, axis=1)

                hit_count = int(np.count_nonzero(joint_errors < MUPOTS_PCK_THRESH_MM))
                pck_hits_all += hit_count
                pck_hits_matched += hit_count
                joint_cnt_matched += MPII_JOINT_NUM

            # unmatched GT contribute zero hits but already counted in joint_cnt_all

        progress_bar.update(1)

    progress_bar.close()

    if distributed:
        total_gt_persons = accelerator.gather_for_metrics(total_gt_persons).sum(dim=0)
        matched_gt_persons = accelerator.gather_for_metrics(matched_gt_persons).sum(dim=0)
        pck_hits_all = accelerator.gather_for_metrics(pck_hits_all).sum(dim=0)
        pck_hits_matched = accelerator.gather_for_metrics(pck_hits_matched).sum(dim=0)
        joint_cnt_all = accelerator.gather_for_metrics(joint_cnt_all).sum(dim=0)
        joint_cnt_matched = accelerator.gather_for_metrics(joint_cnt_matched).sum(dim=0)

    total_gt = int(total_gt_persons.item())
    matched_gt = int(matched_gt_persons.item())
    joint_all = float(joint_cnt_all.item())
    joint_matched = float(joint_cnt_matched.item())
    hits_all = float(pck_hits_all.item())
    hits_matched = float(pck_hits_matched.item())

    pck_all = (hits_all / joint_all * 100.0) if joint_all > 0 else None
    pck_matched = (hits_matched / joint_matched * 100.0) if joint_matched > 0 else None

    error_dict = {
        'PCK_all': round(pck_all, 2) if pck_all is not None else None,
        'PCK_matched': round(pck_matched, 2) if pck_matched is not None else None,
        'num_total_persons': total_gt,
        'num_matched_persons': matched_gt,
    }

    if accelerator.is_main_process:
        print("[MuPoTS Evaluation] Summary:")
        for k, v in error_dict.items():
            print(f"  {k}: {v}")
        with open(os.path.join(results_save_path, 'mupots_results.txt'), 'w') as f:
            for k, v in error_dict.items():
                f.write(f'{k}: {v}\n')

    return error_dict


def evaluate_agora(model, eval_dataloader, conf_thresh,
                        vis = True, vis_step = 40, results_save_path = None,
                        distributed = False, accelerator = None,
                        pcd_threshold = 0.2):
    assert results_save_path is not None
    assert accelerator is not None
    has_kid = ('train' in eval_dataloader.dataset.split and eval_dataloader.dataset.ds_name == 'agora')
    
    os.makedirs(results_save_path,exist_ok=True)
    if vis:
        imgs_save_dir = os.path.join(results_save_path, 'imgs')
        os.makedirs(imgs_save_dir, exist_ok = True)
    
    step = 0
    total_miss_count = 0
    total_count = 0
    total_fp = 0
    
    # Detailed PCD counters (ROMP-style EQ/CD/FD)
    total_eq_correct, total_eq_total = 0, 0
    total_cd_correct, total_cd_total = 0, 0
    total_fd_correct, total_fd_total = 0, 0
    
    # Binned PCD counters
    total_bin_stats = {'0-2m': [0, 0], '2-5m': [0, 0], '5-10m': [0, 0], '10m+': [0, 0]}
    
    # Absolute depth error counters
    total_abs_depth_err = 0.0
    total_abs_depth_count = 0
    total_bin_abs_err = {'0-2m': [0.0, 0], '2-5m': [0.0, 0], '5-10m': [0.0, 0], '10m+': [0.0, 0]}
    
    mve, mpjpe = [], []

    if has_kid:
        kid_total_miss_count = 0
        kid_total_count = 0
        kid_mve, kid_mpjpe = [], []

    cur_device = next(model.parameters()).device
    smpl_layer = model.human_model
    body_verts_ind = smpl_layer.body_vertex_idx
    
    progress_bar = tqdm(total=len(eval_dataloader), disable=not accelerator.is_local_main_process)
    progress_bar.set_description('evaluate')
    for itr, (samples, targets) in enumerate(eval_dataloader):
        samples=[sample.to(device = cur_device, non_blocking = True) for sample in samples]
        with torch.no_grad():    
           outputs = model(samples, targets)
        bs = len(targets)

        batch_count = []
        batch_miss_count = []
        batch_fp = []
        batch_mve = []
        batch_mpjpe = []
        batch_pcd_stats = [] # Store flattened list of PCD stats per image: overall(6) + bins(8) = 14 elements
        batch_abs_depth_err = [] # Store binned ADE stats: 4 bins * [sum, count] = 8 elements
        if has_kid:
            batch_kid_count = []
            batch_kid_miss_count = []
            batch_kid_mve = []
            batch_kid_mpjpe = []

        for idx in range(bs):
            batch_count.append(0)
            batch_miss_count.append(0)
            batch_fp.append(0)
            sample_mve = [float('inf')]
            sample_mpjpe = [float('inf')]
            if has_kid:
                batch_kid_count.append(0)
                batch_kid_miss_count.append(0)
                sample_kid_mve = [float('inf')]
                sample_kid_mpjpe = [float('inf')]

            #gt
            gt_j2ds = targets[idx]['j2ds'].cpu().numpy()[:,:24,:]
            gt_j3ds = targets[idx]['j3ds'].cpu().numpy()[:,:24,:]
            gt_verts = targets[idx]['verts'].cpu().numpy()

            #pred
            select_queries_idx = torch.where(outputs['pred_confs'][idx] > conf_thresh)[0]
            pred_j2ds = outputs['pred_j2ds'][idx][select_queries_idx].detach().cpu().numpy()[:,:24,:]
            pred_j3ds = outputs['pred_j3ds'][idx][select_queries_idx].detach().cpu().numpy()[:,:24,:]
            pred_verts = outputs['pred_verts'][idx][select_queries_idx].detach().cpu().numpy()
            pred_transl = outputs['pred_transl'][idx][select_queries_idx].detach().cpu().numpy()
            
            pred_j3ds_original = pred_j3ds - pred_transl[:, None, :]


            matched_verts_idx = []
            matched_pcd_gt = []
            matched_pcd_pred = []
            assert len(gt_j2ds.shape) == 3 and len(pred_j2ds.shape) == 3
            gtIdxs = np.arange(len(gt_j3ds))
            if len(pred_j2ds) == 0:
                matchDict = {str(gtIdx): 'miss' for gtIdx in gtIdxs}
                falsePositive_count = 0
            else:
                greedy_match = match_2d_greedy(pred_j2ds, gt_j2ds) # tuples are (idx_pred_kps, idx_gt_kps)
                matchDict, falsePositive_count = get_matching_dict(greedy_match)

            gt_verts_list, pred_verts_list, gt_joints_list, pred_joints_list = [], [], [], []
            miss_flag = []
            for gtIdx in gtIdxs:
                gt_verts_list.append(gt_verts[gtIdx])
                gt_joints_list.append(gt_j3ds[gtIdx])
                if matchDict[str(gtIdx)] == 'miss' or matchDict[str(
                        gtIdx)] == 'invalid':
                    miss_flag.append(1)
                    pred_verts_list.append([])
                    pred_joints_list.append([])
                else:
                    miss_flag.append(0)
                    p_idx = int(matchDict[str(gtIdx)])
                    pred_joints_list.append(pred_j3ds[p_idx])
                    pred_verts_list.append(pred_verts[p_idx])
                    matched_verts_idx.append(p_idx)
                    
                    # Collect depths for PCD
                    if 'depths' in targets[idx]:
                        gt_depth = targets[idx]['depths'][gtIdx, 0].item()
                        pred_depth = outputs['pred_depths'][idx][select_queries_idx[p_idx], 0].item()
                        matched_pcd_gt.append(gt_depth)
                        matched_pcd_pred.append(pred_depth)

            if has_kid:
                gt_kid_list = targets[idx]['kid']

            # Calculate PCD for this image
            if len(matched_pcd_gt) >= 2:
                pcd_results = cal_pcd_error(np.array(matched_pcd_pred), np.array(matched_pcd_gt), threshold=pcd_threshold)
                # Flatten stats for gathering: overall (6) + bins (4 bins * 2 stats = 8) = 14 elements
                stats_list = pcd_results['overall']
                for name in ['0-2m', '2-5m', '5-10m', '10m+']:
                    stats_list.extend(pcd_results['bins'][name])
                batch_pcd_stats.append(stats_list)
            else:
                batch_pcd_stats.append([0] * 14)
                
            # Calculate Absolute Depth Error for matched instances
            bin_ade_stats = [0.0, 0, 0.0, 0, 0.0, 0, 0.0, 0]
            if len(matched_pcd_gt) > 0:
                gt_depths_arr = np.array(matched_pcd_gt)
                pred_depths_arr = np.array(matched_pcd_pred)
                abs_errs = np.abs(pred_depths_arr - gt_depths_arr)
                
                bins = [0, 2, 5, 10, float('inf')]
                for i in range(len(gt_depths_arr)):
                    d = gt_depths_arr[i]
                    err = abs_errs[i]
                    for b in range(len(bins)-1):
                        if bins[b] <= d < bins[b+1]:
                            bin_ade_stats[b*2] += float(err)
                            bin_ade_stats[b*2 + 1] += 1
                            break
            batch_abs_depth_err.append(bin_ade_stats)

            #calculating 3d errors
            for i, (gt3d, pred) in enumerate(zip(gt_joints_list, pred_joints_list)):
                batch_count[-1] += 1
                if has_kid and gt_kid_list[i]:
                    batch_kid_count[-1] += 1

                # Get corresponding ground truth and predicted 3d joints and verts
                if miss_flag[i] == 1:
                    batch_miss_count[-1] += 1
                    if has_kid and gt_kid_list[i]:
                        batch_kid_miss_count[-1] += 1
                    continue

                gt3d = gt3d.reshape(-1, 3)
                pred3d = pred.reshape(-1, 3)
                gt3d_verts = gt_verts_list[i].reshape(-1, 3)
                pred3d_verts = pred_verts_list[i].reshape(-1, 3)
                
                gt3d, gt3d_verts = select_and_align(gt3d, gt3d_verts, body_verts_ind)
                pred3d, pred3d_verts = select_and_align(pred3d, pred3d_verts, body_verts_ind)

                #joints
                error_j, pa_error_j = cal_3d_position_error(pred3d, gt3d)
                sample_mpjpe.append(float(error_j))
                if has_kid and gt_kid_list[i]:
                    sample_kid_mpjpe.append(float(error_j))
                #vertices
                error_v,pa_error_v = cal_3d_position_error(pred3d_verts, gt3d_verts)
                sample_mve.append(float(error_v))
                if has_kid and gt_kid_list[i]:
                    sample_kid_mve.append(float(error_v))


            #counting
            step += 1
            batch_fp[-1] += falsePositive_count

            batch_mve.append(np.array(sample_mve))
            batch_mpjpe.append(np.array(sample_mpjpe))
            if has_kid:
                batch_kid_mve.append(np.array(sample_kid_mve))
                batch_kid_mpjpe.append(np.array(sample_kid_mpjpe))

            img_idx = step + accelerator.process_index*len(eval_dataloader)*bs
            
            if vis and (img_idx%vis_step == 0):
                img_name = targets[idx]['img_path'].split('/')[-1].split('.')[0]
                ori_img = tensor_to_BGR(unNormalize(samples[idx]).cpu())
                # Row1: original image + GT bbox (with GT Tz)
                # Row2: GT mesh + predicted mesh
                # Row3: bbox prior + depth-prior Tz (left) and HMR predicted bbox + depth (right)
                # Row4: GT meshes (left) and predicted meshes (right) in an oblique/top-view for spatial consistency

                colors = [(1.0, 1.0, 0.9)] * len(gt_verts)
                gt_mesh_img = vis_meshes_img(img = ori_img.copy(),
                                            verts = gt_verts,
                                            smpl_faces = smpl_layer.faces,
                                            cam_intrinsics = targets[idx]['cam_intrinsics'].reshape(3,3).detach().cpu(),
                                            colors = colors)

                colors = [(1.0, 0.6, 0.6)] * len(pred_verts)   
                for i in matched_verts_idx:
                    colors[i] = (0.7, 1.0, 0.4)

                pred_mesh_img = vis_meshes_img(img = ori_img.copy(),
                                            verts = pred_verts,
                                            smpl_faces = smpl_layer.faces,
                                            cam_intrinsics = outputs['pred_intrinsics'][idx].reshape(3,3).detach().cpu(),
                                            colors = colors,
                                            )

                gt_Tz_list = None
                try:
                    if 'depths' in targets[idx]:
                        gt_depths = targets[idx]['depths'].detach().cpu().numpy()
                        gt_Tz_list = gt_depths[:, 0]
                    else:
                        gt_Tz_list = gt_j3ds[:, 0, 2]
                except Exception:
                    gt_Tz_list = None

                gt_box_img = ori_img.copy()
                gt_boxes_xyxy = None
                if 'boxes' in targets[idx]:
                    gt_boxes = targets[idx]['boxes']  # cxcywh, normalized
                    if isinstance(gt_boxes, torch.Tensor):
                        gt_boxes = gt_boxes.detach().cpu()
                    gt_boxes_xyxy = box_cxcywh_to_xyxy(gt_boxes) * model.input_size
                    gt_box_img = vis_boxes(ori_img.copy(), gt_boxes_xyxy, color=(0, 255, 0))

                    if gt_Tz_list is not None:
                        for i_box, bbox in enumerate(gt_boxes_xyxy):
                            bbox = bbox.int().tolist()
                            x1, y1, x2, y2 = bbox
                            if i_box < len(gt_Tz_list):
                                tz = float(gt_Tz_list[i_box])
                                text = f"GTz:{tz:.2f}"
                                cv2.putText(gt_box_img, text, (x1, max(y1 - 5, 0)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

                # 3) Row3 left: bbox prior boxes + depth-prior Tz (if available)
                depth_box_img = ori_img.copy()
                if 'bbox_prior_outputs' in outputs and outputs['bbox_prior_outputs'] is not None:
                    bbox_prior_outputs = outputs['bbox_prior_outputs']
                    prior_pred_boxes_all = bbox_prior_outputs['pred_boxes'][-1][idx]      # (num_queries, 4)
                    prior_pred_confs_all = bbox_prior_outputs['pred_confs'][-1][idx]      # (num_queries, 1)

                    prior_pred_boxes_all = prior_pred_boxes_all.detach().cpu()
                    prior_pred_confs_all = prior_pred_confs_all.detach().cpu()[..., 0]

                    prior_keep = torch.where(prior_pred_confs_all > conf_thresh)[0]
                    if len(prior_keep) > 0:
                        prior_boxes = prior_pred_boxes_all[prior_keep]
                        prior_boxes_xyxy = box_cxcywh_to_xyxy(prior_boxes) * model.input_size
                        depth_box_img = vis_boxes(ori_img.copy(), prior_boxes_xyxy, color=(0, 255, 0))

                        # For each prior bbox, optionally draw depth-prior Tz prediction
                        if 'depth_tz_outputs' in outputs and outputs['depth_tz_outputs'] is not None:
                            depth_tz_outputs = outputs['depth_tz_outputs']
                            depth_tz_all = depth_tz_outputs['pred_tz'][idx]
                            if isinstance(depth_tz_all, torch.Tensor):
                                depth_tz_all = depth_tz_all.detach().cpu()[..., 0]
                            else:
                                depth_tz_all = torch.as_tensor(depth_tz_all, dtype=prior_pred_boxes_all.dtype)

                            for local_idx, bbox_t in enumerate(prior_boxes_xyxy):
                                global_idx = int(prior_keep[local_idx])
                                if global_idx >= len(depth_tz_all):
                                    continue
                                tz = float(depth_tz_all[global_idx])
                                x1, y1, x2, y2 = bbox_t.int().tolist()
                                text = f"Tz:{tz:.2f}"
                                cv2.putText(depth_box_img, text, (x1, max(y1 - 5, 0)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
                else:
                    depth_box_img = vis_boxes(ori_img.copy(), [], color=(0, 255, 0))

                pred_boxes = outputs['pred_boxes'][idx][select_queries_idx].detach().cpu()
                pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes) * model.input_size
                hmr_box_img = vis_boxes(ori_img.copy(), pred_boxes_xyxy, color=(255, 0, 255))

                hmr_pred_to_gt = [-1] * len(pred_boxes_xyxy)
                for gtIdx in gtIdxs:
                    match_val = matchDict[str(gtIdx)]
                    if isinstance(match_val, str):
                        continue
                    pred_idx = int(match_val)
                    if 0 <= pred_idx < len(hmr_pred_to_gt):
                        hmr_pred_to_gt[pred_idx] = gtIdx

                if 'pred_depths' in outputs:
                    pred_depths = outputs['pred_depths'][idx]  # (num_queries, 2) -> [root_z, root_z/f]
                    pred_root_all = pred_depths.detach().cpu()[..., 0]
                    pred_root_selected = pred_root_all[select_queries_idx.cpu()]

                    for p_idx, bbox_t in enumerate(pred_boxes_xyxy):
                        if p_idx >= len(pred_root_selected):
                            continue
                        tz = float(pred_root_selected[p_idx])
                        x1, y1, x2, y2 = bbox_t.int().tolist()
                        text = f"Predz:{tz:.2f}"
                        cv2.putText(hmr_box_img, text, (x1, max(y1 - 5, 0)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1, cv2.LINE_AA)

                if len(gt_verts) > 0 or len(pred_verts) > 0:
                    concat_verts = []
                    if len(gt_verts) > 0:
                        concat_verts.append(gt_verts.reshape(-1, 3))
                    if len(pred_verts) > 0:
                        concat_verts.append(pred_verts.reshape(-1, 3))
                    concat_verts = np.concatenate(concat_verts, axis=0)

                    center = concat_verts.mean(axis=0)
                    radius = np.linalg.norm(concat_verts - center, axis=1).max()
                    if radius < 1e-3:
                        radius = 1.0
                else:
                    center, radius = None, None

                if len(gt_verts) > 0:
                    gt_topview_img = vis_meshes_topview(
                        verts=gt_verts,
                        smpl_faces=smpl_layer.faces,
                        out_size=model.input_size,
                        colors=[(1.0, 1.0, 0.9)] * len(gt_verts),
                        view="oblique",
                        center=center,
                        radius=radius,
                    )
                else:
                    gt_topview_img = pad_img(ori_img, model.input_size)

                if len(pred_verts) > 0:
                    pred_topview_img = vis_meshes_topview(
                        verts=pred_verts,
                        smpl_faces=smpl_layer.faces,
                        out_size=model.input_size,
                        colors=get_colors_rgb(len(pred_verts)),
                        view="oblique",
                        center=center,
                        radius=radius,
                    )
                else:
                    pred_topview_img = pad_img(ori_img, model.input_size)

                ori_img_pad = pad_img(ori_img, model.input_size)
                gt_box_img_pad = pad_img(gt_box_img, model.input_size)
                gt_mesh_img_pad = pad_img(gt_mesh_img, model.input_size)
                pred_mesh_img_pad = pad_img(pred_mesh_img, model.input_size)
                depth_box_img_pad = pad_img(depth_box_img, model.input_size)
                hmr_box_img_pad = pad_img(hmr_box_img, model.input_size)
                gt_topview_img_pad = pad_img(gt_topview_img, model.input_size)
                pred_topview_img_pad = pad_img(pred_topview_img, model.input_size)

                rows = [
                    np.hstack([ori_img_pad, gt_box_img_pad]),
                    np.hstack([gt_mesh_img_pad, pred_mesh_img_pad]),
                    np.hstack([depth_box_img_pad, hmr_box_img_pad]),
                    np.hstack([gt_topview_img_pad, pred_topview_img_pad]),
                ]
                full_img = np.vstack(rows)

                cv2.imwrite(os.path.join(imgs_save_dir, f'{img_idx}_{img_name}.png'), full_img)
                
        if distributed:
            batch_count = accelerator.gather_for_metrics(batch_count)
            batch_miss_count = accelerator.gather_for_metrics(batch_miss_count)
            batch_fp = accelerator.gather_for_metrics(batch_fp)
            batch_mve = accelerator.gather_for_metrics(batch_mve)
            batch_mpjpe = accelerator.gather_for_metrics(batch_mpjpe)
            batch_pcd_stats = accelerator.gather_for_metrics(batch_pcd_stats)
            batch_abs_depth_err = accelerator.gather_for_metrics(batch_abs_depth_err)
            if has_kid:
                batch_kid_count = accelerator.gather_for_metrics(batch_kid_count)
                batch_kid_miss_count = accelerator.gather_for_metrics(batch_kid_miss_count)
                batch_kid_mve = accelerator.gather_for_metrics(batch_kid_mve)
                batch_kid_mpjpe = accelerator.gather_for_metrics(batch_kid_mpjpe)

        total_count += sum(batch_count)
        total_miss_count += sum(batch_miss_count)
        total_fp += sum(batch_fp)
        
        # Accumulate detailed PCD stats
        for stats in batch_pcd_stats:
            total_eq_correct += stats[0]
            total_eq_total += stats[1]
            total_cd_correct += stats[2]
            total_cd_total += stats[3]
            total_fd_correct += stats[4]
            total_fd_total += stats[5]
            
            # Accumulate bins
            for b_idx, name in enumerate(['0-2m', '2-5m', '5-10m', '10m+']):
                total_bin_stats[name][0] += stats[6 + b_idx*2]
                total_bin_stats[name][1] += stats[7 + b_idx*2]
            
        # Accumulate Absolute Depth Error for AGORA
        for err_stat in batch_abs_depth_err:
            # Overall
            total_abs_depth_err += sum(err_stat[0::2])
            total_abs_depth_count += sum(err_stat[1::2])
            # Bins
            for b_idx, name in enumerate(['0-2m', '2-5m', '5-10m', '10m+']):
                total_bin_abs_err[name][0] += err_stat[b_idx*2]
                total_bin_abs_err[name][1] += int(err_stat[b_idx*2 + 1])

        mve += batch_mve
        mpjpe += batch_mpjpe
        if has_kid:
            kid_total_count += sum(batch_kid_count)
            kid_total_miss_count += sum(batch_kid_miss_count)
            kid_mve += batch_kid_mve
            kid_mpjpe += batch_kid_mpjpe

        progress_bar.update(1)

    mve = np.concatenate([item[1:] for item in mve], axis=0).tolist() if len(mve) > 0 else []
    mpjpe = np.concatenate([item[1:] for item in mpjpe], axis=0).tolist() if len(mpjpe) > 0 else []
    if has_kid:
        kid_mve = np.concatenate([item[1:] for item in kid_mve], axis=0).tolist() if len(kid_mve) > 0 else []
        kid_mpjpe = np.concatenate([item[1:] for item in kid_mpjpe], axis=0).tolist() if len(kid_mpjpe) > 0 else []

    if len(mpjpe) <= 0:
        return "Failed to evaluate. Keep training!"
    if has_kid and len(kid_mpjpe) <= 0:
        return "Failed to evaluate. Keep training!"
    
    precision, recall, f1 = compute_prf1(total_count,total_miss_count,total_fp)
    error_dict = {}
    error_dict['precision'] = precision
    error_dict['recall'] = recall
    error_dict['f1'] = f1
    error_dict['MPJPE'] = round(float(sum(mpjpe)/len(mpjpe)), 1)
    
    # Report detailed PCD metrics
    pcd_total_correct = total_eq_correct + total_cd_correct + total_fd_correct
    pcd_total_pairs = total_eq_total + total_cd_total + total_fd_total
    
    if pcd_total_pairs > 0:
        error_dict['PCD_total'] = round(float(pcd_total_correct / pcd_total_pairs) * 100, 2)
    else:
        error_dict['PCD_total'] = 0.0
        
    error_dict['PCD_eq'] = round(float(total_eq_correct / total_eq_total * 100), 2) if total_eq_total > 0 else 0.0
    error_dict['PCD_cd'] = round(float(total_cd_correct / total_cd_total * 100), 2) if total_cd_total > 0 else 0.0
    error_dict['PCD_fd'] = round(float(total_fd_correct / total_fd_total * 100), 2) if total_fd_total > 0 else 0.0
    
    if total_abs_depth_count > 0:
        error_dict['AbsDepthErr'] = round(float(total_abs_depth_err / total_abs_depth_count), 3)
    else:
        error_dict['AbsDepthErr'] = 0.0
    
    # Report binned PCD for AGORA
    for name in ['0-2m', '2-5m', '5-10m', '10m+']:
        c, t = total_bin_stats[name]
        error_dict[f'PCD_{name}'] = round(float(c / t * 100), 2) if t > 0 else 0.0
    
    # Report binned Absolute Depth Error for AGORA
    for name in ['0-2m', '2-5m', '5-10m', '10m+']:
        s, c = total_bin_abs_err[name]
        error_dict[f'ADE_{name}'] = round(float(s / c), 3) if c > 0 else 0.0
    
    #error_dict['NMJE'] = round(error_dict['MPJPE'] / (f1), 1)
    if f1 == 0:
        error_dict['NMJE'] = 0.0
    else:
        error_dict['NMJE'] = round(error_dict['MPJPE'] / f1, 1)
        
    error_dict['MVE'] = round(float(sum(mve)/len(mve)), 1)
    #error_dict['NMVE'] = round(error_dict['MVE'] / (f1), 1)
    if f1 == 0:
        error_dict['NMVE'] = 0.0
    else:
        error_dict['NMVE'] = round(error_dict['MVE'] / f1, 1)


    if has_kid:
        kid_precision, kid_recall, kid_f1 = compute_prf1(kid_total_count,kid_total_miss_count,total_fp)
        error_dict['kid_precision'] = kid_precision
        error_dict['kid_recall'] = kid_recall
        error_dict['kid_f1'] = kid_f1

        error_dict['kid-MPJPE'] = round(float(sum(kid_mpjpe)/len(kid_mpjpe)), 1)
        #error_dict['kid-NMJE'] = round(error_dict['kid-MPJPE'] / (kid_f1), 1)
        error_dict['kid-MVE'] = round(float(sum(kid_mve)/len(kid_mve)), 1)
        #error_dict['kid-NMVE'] = round(error_dict['kid-MVE'] / (kid_f1), 1)
        if kid_f1 == 0:
            error_dict['kid-NMJE'] = 0.0
            error_dict['kid-NMVE'] = 0.0
        else:
            error_dict['kid-NMJE'] = round(error_dict['kid-MPJPE'] / kid_f1, 1)
            error_dict['kid-NMVE'] = round(error_dict['kid-MVE'] / kid_f1, 1)

    if accelerator.is_main_process:
        with open(os.path.join(results_save_path,'results.txt'),'w') as f:
            for k,v in error_dict.items():
                f.write(f'{k}: {v}\n')

    return error_dict


def test_agora(model, eval_dataloader, conf_thresh, 
                vis = True, vis_step = 400, results_save_path = None,
                distributed = False, accelerator = None):
    assert results_save_path is not None
    assert accelerator is not None

    os.makedirs(os.path.join(results_save_path,'predictions'),exist_ok=True)
    if vis:
        imgs_save_dir = os.path.join(results_save_path, 'imgs')
        os.makedirs(imgs_save_dir, exist_ok = True)
    step = 0
    cur_device = next(model.parameters()).device
    smpl_layer = model.human_model
    
    progress_bar = tqdm(total=len(eval_dataloader), disable=not accelerator.is_local_main_process)
    progress_bar.set_description('testing')
    for itr, (samples, targets) in enumerate(eval_dataloader):
        samples=[sample.to(device = cur_device, non_blocking = True) for sample in samples]
        with torch.no_grad():    
           outputs = model(samples, targets)
        bs = len(targets)
        for idx in range(bs):
            #gt
            img_name = targets[idx]['img_name'].split('.')[0]
            #pred
            select_queries_idx = torch.where(outputs['pred_confs'][idx] > conf_thresh)[0]
            pred_j2ds = np.array(outputs['pred_j2ds'][idx][select_queries_idx].detach().to('cpu'))[:,:24,:]*(3840/model.input_size)
            pred_j3ds = np.array(outputs['pred_j3ds'][idx][select_queries_idx].detach().to('cpu'))[:,:24,:]
            pred_verts = np.array(outputs['pred_verts'][idx][select_queries_idx].detach().to('cpu'))
            pred_poses = np.array(outputs['pred_poses'][idx][select_queries_idx].detach().to('cpu'))
            pred_betas = np.array(outputs['pred_betas'][idx][select_queries_idx].detach().to('cpu'))

            #visualization
            step+=1
            img_idx = step + accelerator.process_index*len(eval_dataloader)*bs
            if vis and (img_idx%vis_step == 0):
                ori_img = tensor_to_BGR(unNormalize(samples[idx]).cpu())
                ori_img = pad_img(ori_img, model.input_size)

                
                colors = get_colors_rgb(len(pred_verts))
                mesh_img = vis_meshes_img(img = ori_img.copy(),
                                          verts = pred_verts,
                                          smpl_faces = smpl_layer.faces,
                                          colors = colors,
                                          cam_intrinsics = outputs['pred_intrinsics'][idx].detach().cpu())
                
                
                full_img = np.vstack([np.hstack([ori_img, mesh_img]),
                                      np.hstack([np.zeros_like(ori_img), np.zeros_like(ori_img)])])
                cv2.imwrite(os.path.join(imgs_save_dir, f'{img_idx}_{img_name}.png'), full_img)

            
            # submit
            for pnum in range(len(pred_j2ds)):
                smpl_dict = {}
                # smpl_dict['age'] = 'kid'
                smpl_dict['joints'] = pred_j2ds[pnum].reshape(24,2)
                smpl_dict['params'] = {'transl': np.zeros((1,3)),
                                        'betas': pred_betas[pnum].reshape(1,10),
                                        'global_orient': pred_poses[pnum][:3].reshape(1,1,3),
                                        'body_pose': pred_poses[pnum][3:].reshape(1,23,3)}
                # smpl_dict['verts'] = pred_verts[pnum].reshape(6890,3)
                # smpl_dict['allSmplJoints3d'] = pred_j3ds[pnum].reshape(24,3)
                with open(os.path.join(results_save_path,'predictions',f'{img_name}_personId_{pnum}.pkl'), 'wb') as f:
                    pickle.dump(smpl_dict, f)
 
        progress_bar.update(1)

    if distributed:
        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        accelerator.print('Packing...')

        folder_path = os.path.join(results_save_path,'predictions')
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(results_save_path,f'pred_{timestamp}.zip')
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, os.path.dirname(folder_path))
                    zipf.write(file_path, arcname)

    return 'Results saved at: ' + os.path.join(results_save_path,'predictions')

def evaluate_3dpw(model, eval_dataloader, conf_thresh,
                        vis = True, vis_step = 40, results_save_path = None,
                        distributed = False, accelerator = None):
    assert results_save_path is not None
    assert accelerator is not None
    num_processes = accelerator.num_processes
    
    os.makedirs(results_save_path,exist_ok=True)
    if vis:
        imgs_save_dir = os.path.join(results_save_path, 'imgs')
        os.makedirs(imgs_save_dir, exist_ok = True)
    
    step = 0
    total_miss_count = 0
    total_count = 0
    total_fp = 0

    mve, mpjpe, pa_mpjpe, pa_mve = [], [], [], []
    cur_device = next(model.parameters()).device
    smpl_layer = model.human_model
    smpl2h36m_regressor = torch.from_numpy(smpl_layer.smpl2h36m_regressor).float().to(cur_device)
    
    progress_bar = tqdm(total=len(eval_dataloader), disable=not accelerator.is_local_main_process)
    progress_bar.set_description('evaluate')
    for itr, (samples, targets) in enumerate(eval_dataloader):
        samples=[sample.to(device = cur_device, non_blocking = True) for sample in samples]
        with torch.no_grad():    
           outputs = model(samples, targets)
        bs = len(targets)

        batch_count = []
        batch_miss_count = []
        batch_fp = []
        batch_mve = []
        batch_mpjpe = []
        batch_pa_mve = []
        batch_pa_mpjpe = []
        for idx in range(bs):
            batch_count.append(0)
            batch_miss_count.append(0)
            batch_fp.append(0)
            sample_mve = [float('inf')]
            sample_pa_mve = [float('inf')]
            sample_mpjpe = [float('inf')]
            sample_pa_mpjpe = [float('inf')]

            #gt 
            gt_verts = targets[idx]['verts']
            gt_transl = targets[idx]['transl']
            gt_j3ds = torch.einsum('bik,ji->bjk', [gt_verts - gt_transl[:,None,:], smpl2h36m_regressor]) + gt_transl[:,None,:]

            gt_verts = gt_verts.cpu().numpy()
            gt_j3ds = gt_j3ds.cpu().numpy()
            gt_j2ds = targets[idx]['j2ds'].cpu().numpy()[:,:24,:]

            #pred
            select_queries_idx = torch.where(outputs['pred_confs'][idx] > conf_thresh)[0]
            
            pred_verts = outputs['pred_verts'][idx][select_queries_idx].detach()
            pred_transl = outputs['pred_transl'][idx][select_queries_idx].detach()
            pred_j3ds = torch.einsum('bik,ji->bjk', [pred_verts - pred_transl[:,None,:], smpl2h36m_regressor]) + pred_transl[:,None,:]
            
            pred_verts = pred_verts.cpu().numpy()
            pred_j3ds = pred_j3ds.cpu().numpy()
            pred_j2ds = outputs['pred_j2ds'][idx][select_queries_idx].detach().cpu().numpy()[:,:24,:]


            matched_verts_idx = []
            assert len(gt_j2ds.shape) == 3 and len(pred_j2ds.shape) == 3
            gtIdxs = np.arange(len(gt_j3ds))
            if len(pred_j2ds) == 0:
                matchDict = {str(gtIdx): 'miss' for gtIdx in gtIdxs}
                falsePositive_count = 0
            else:
                greedy_match = match_2d_greedy(pred_j2ds, gt_j2ds) # tuples are (idx_pred_kps, idx_gt_kps)
                matchDict, falsePositive_count = get_matching_dict(greedy_match)

            gt_verts_list, pred_verts_list, gt_joints_list, pred_joints_list = [], [], [], []
            miss_flag = []
            for gtIdx in gtIdxs:
                gt_verts_list.append(gt_verts[gtIdx])
                gt_joints_list.append(gt_j3ds[gtIdx])
                if matchDict[str(gtIdx)] == 'miss' or matchDict[str(
                        gtIdx)] == 'invalid':
                    miss_flag.append(1)
                    pred_verts_list.append([])
                    pred_joints_list.append([])
                else:
                    miss_flag.append(0)
                    p_idx = int(matchDict[str(gtIdx)])
                    pred_joints_list.append(pred_j3ds[p_idx])
                    pred_verts_list.append(pred_verts[p_idx])
                    matched_verts_idx.append(p_idx)

            #calculating 3d errors
            for i, (gt3d, pred) in enumerate(zip(gt_joints_list, pred_joints_list)):
                batch_count[-1] += 1

                # Get corresponding ground truth and predicted 3d joints and verts
                if miss_flag[i] == 1:
                    batch_miss_count[-1] += 1
                    continue

                gt3d = gt3d.reshape(-1, 3)
                pred3d = pred.reshape(-1, 3)
                gt3d_verts = gt_verts_list[i].reshape(-1, 3)
                pred3d_verts = pred_verts_list[i].reshape(-1, 3)

                gt_pelvis = gt3d[[0],:].copy()
                pred_pelvis = pred3d[[0],:].copy()

                gt3d = (gt3d - gt_pelvis)[H36M_EVAL_JOINTS, :].copy()
                gt3d_verts = (gt3d_verts - gt_pelvis).copy()
                
                pred3d = (pred3d - pred_pelvis)[H36M_EVAL_JOINTS, :].copy()
                pred3d_verts = (pred3d_verts - pred_pelvis).copy()

                #joints
                error_j, pa_error_j = cal_3d_position_error(pred3d, gt3d)
                sample_mpjpe.append(float(error_j))
                sample_pa_mpjpe.append(float(pa_error_j))
                #vertices
                error_v, pa_error_v = cal_3d_position_error(pred3d_verts, gt3d_verts)
                sample_mve.append(float(error_v))
                sample_pa_mve.append(float(pa_error_v))


            #counting
            step += 1
            batch_fp[-1] += falsePositive_count

            batch_mve.append(np.array(sample_mve))
            batch_pa_mve.append(np.array(sample_pa_mve))
            batch_mpjpe.append(np.array(sample_mpjpe))
            batch_pa_mpjpe.append(np.array(sample_pa_mpjpe))

            img_idx = step + accelerator.process_index*len(eval_dataloader)*bs
            
            if vis and (img_idx%vis_step == 0) and len(matched_verts_idx) > 0:
                img_name = targets[idx]['img_path'].split('/')[-1].split('.')[0]
                ori_img = tensor_to_BGR(unNormalize(samples[idx]).cpu())

                gt_Tz_list = None
                try:
                    if 'depths' in targets[idx]:
                        gt_depths = targets[idx]['depths'].detach().cpu().numpy()
                        gt_Tz_list = gt_depths[:, 0]
                    else:
                        gt_Tz_list = gt_j3ds[:, 0, 2]
                except Exception:
                    gt_Tz_list = None

                gt_box_img = ori_img.copy()
                gt_boxes_xyxy = None
                if 'boxes' in targets[idx]:
                    gt_boxes = targets[idx]['boxes']  # cxcywh, normalized
                    if isinstance(gt_boxes, torch.Tensor):
                        gt_boxes = gt_boxes.detach().cpu()
                    gt_boxes_xyxy = box_cxcywh_to_xyxy(gt_boxes) * model.input_size
                    gt_box_img = vis_boxes(ori_img.copy(), gt_boxes_xyxy, color=(0, 255, 0))

                    if gt_Tz_list is not None:
                        for i_box, bbox in enumerate(gt_boxes_xyxy):
                            bbox = bbox.int().tolist()
                            x1, y1, x2, y2 = bbox
                            if i_box < len(gt_Tz_list):
                                tz = float(gt_Tz_list[i_box])
                                text = f"GTz:{tz:.2f}"
                                cv2.putText(gt_box_img, text, (x1, max(y1 - 5, 0)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

                try:
                    cam_int_gt = targets[idx]['cam_intrinsics'].reshape(3, 3).detach().cpu()
                except Exception:
                    cam_int_gt = outputs['pred_intrinsics'][idx].detach().cpu()

                gt_mesh_img = vis_meshes_img(
                    img=ori_img.copy(),
                    verts=gt_verts,
                    smpl_faces=smpl_layer.faces,
                    cam_intrinsics=cam_int_gt,
                )

                selected_verts = pred_verts[matched_verts_idx]
                colors = get_colors_rgb(len(selected_verts))
                cam_int_pred = outputs['pred_intrinsics'][idx].detach().cpu()
                pred_mesh_img = vis_meshes_img(
                    img=ori_img.copy(),
                    verts=selected_verts,
                    smpl_faces=smpl_layer.faces,
                    colors=colors,
                    cam_intrinsics=cam_int_pred,
                )

                depth_box_img = ori_img.copy()
                hmr_box_img = ori_img.copy()

                if 'bbox_prior_outputs' in outputs and outputs['bbox_prior_outputs'] is not None:
                    bbox_prior_outputs = outputs['bbox_prior_outputs']
                    prior_pred_boxes_all = bbox_prior_outputs['pred_boxes'][-1][idx]
                    prior_pred_confs_all = bbox_prior_outputs['pred_confs'][-1][idx][..., 0]

                    prior_pred_boxes_all = prior_pred_boxes_all.detach().cpu()
                    prior_pred_confs_all = prior_pred_confs_all.detach().cpu()

                    prior_keep = torch.where(prior_pred_confs_all > conf_thresh)[0]
                    if len(prior_keep) > 0:
                        prior_boxes = prior_pred_boxes_all[prior_keep]
                        prior_boxes_xyxy = box_cxcywh_to_xyxy(prior_boxes) * model.input_size
                        depth_box_img = vis_boxes(ori_img.copy(), prior_boxes_xyxy, color=(0, 255, 0))

                        if 'depth_tz_outputs' in outputs and outputs['depth_tz_outputs'] is not None:
                            depth_tz_outputs = outputs['depth_tz_outputs']
                            depth_tz_all = depth_tz_outputs['pred_tz'][idx]
                            if isinstance(depth_tz_all, torch.Tensor):
                                depth_tz_all = depth_tz_all.detach().cpu()[..., 0]
                            else:
                                depth_tz_all = torch.as_tensor(depth_tz_all, dtype=prior_pred_boxes_all.dtype)

                            for local_idx, bbox_t in enumerate(prior_boxes_xyxy):
                                global_idx = int(prior_keep[local_idx])
                                if global_idx >= len(depth_tz_all):
                                    continue
                                tz = float(depth_tz_all[global_idx])
                                x1, y1, x2, y2 = bbox_t.int().tolist()
                                text = f"Tz:{tz:.2f}"
                                cv2.putText(depth_box_img, text, (x1, max(y1 - 5, 0)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

                pred_boxes = outputs['pred_boxes'][idx][select_queries_idx].detach().cpu()
                pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes) * model.input_size
                hmr_box_img = vis_boxes(ori_img.copy(), pred_boxes_xyxy, color=(255, 0, 255))

                if 'pred_depths' in outputs:
                    pred_depths = outputs['pred_depths'][idx]  # (num_queries, 2) -> [root_z, root_z/f]
                    pred_root_all = pred_depths.detach().cpu()[..., 0]
                    pred_root_selected = pred_root_all[select_queries_idx.cpu()]

                    for p_idx, bbox_t in enumerate(pred_boxes_xyxy):
                        if p_idx >= len(pred_root_selected):
                            continue
                        tz = float(pred_root_selected[p_idx])
                        x1, y1, x2, y2 = bbox_t.int().tolist()
                        text = f"Predz:{tz:.2f}"
                        cv2.putText(hmr_box_img, text, (x1, max(y1 - 5, 0)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1, cv2.LINE_AA)

                try:
                    gt_topview_img = vis_meshes_topview(
                        verts=gt_verts,
                        smpl_faces=smpl_layer.faces,
                        out_size=model.input_size,
                        colors=[(1.0, 1.0, 0.9)] * len(gt_verts),
                        view="oblique",
                    )
                except Exception:
                    gt_topview_img = ori_img.copy()

                try:
                    pred_topview_img = vis_meshes_topview(
                        verts=selected_verts,
                        smpl_faces=smpl_layer.faces,
                        out_size=model.input_size,
                        colors=get_colors_rgb(len(selected_verts)),
                        view="oblique",
                    )
                except Exception:
                    pred_topview_img = ori_img.copy()

                ori_img_pad = pad_img(ori_img, model.input_size)
                gt_box_img_pad = pad_img(gt_box_img, model.input_size)
                gt_mesh_img_pad = pad_img(gt_mesh_img, model.input_size)
                pred_mesh_img_pad = pad_img(pred_mesh_img, model.input_size)
                depth_box_img_pad = pad_img(depth_box_img, model.input_size)
                hmr_box_img_pad = pad_img(hmr_box_img, model.input_size)
                gt_topview_img_pad = pad_img(gt_topview_img, model.input_size)
                pred_topview_img_pad = pad_img(pred_topview_img, model.input_size)

                rows = [
                    np.hstack([ori_img_pad, gt_box_img_pad]),
                    np.hstack([gt_mesh_img_pad, pred_mesh_img_pad]),
                    np.hstack([depth_box_img_pad, hmr_box_img_pad]),
                    np.hstack([gt_topview_img_pad, pred_topview_img_pad]),
                ]
                full_img = np.vstack(rows)

                cv2.imwrite(os.path.join(imgs_save_dir, f'{img_idx}_{img_name}.png'), full_img)
                
        if distributed:
            batch_count = accelerator.gather_for_metrics(batch_count)
            batch_miss_count = accelerator.gather_for_metrics(batch_miss_count)
            batch_fp = accelerator.gather_for_metrics(batch_fp)
            batch_mve = accelerator.gather_for_metrics(batch_mve)
            batch_pa_mve = accelerator.gather_for_metrics(batch_pa_mve)
            batch_mpjpe = accelerator.gather_for_metrics(batch_mpjpe)
            batch_pa_mpjpe = accelerator.gather_for_metrics(batch_pa_mpjpe)

        total_count += sum(batch_count)
        total_miss_count += sum(batch_miss_count)
        total_fp += sum(batch_fp)

        mve += batch_mve
        pa_mve += batch_pa_mve
        mpjpe += batch_mpjpe
        pa_mpjpe += batch_pa_mpjpe

        progress_bar.update(1)

    mve = np.concatenate([item[1:] for item in mve], axis=0).tolist() if len(mve) > 0 else []
    pa_mve = np.concatenate([item[1:] for item in pa_mve], axis=0).tolist() if len(pa_mve) > 0 else []
    mpjpe = np.concatenate([item[1:] for item in mpjpe], axis=0).tolist() if len(mpjpe) > 0 else []
    pa_mpjpe = np.concatenate([item[1:] for item in pa_mpjpe], axis=0).tolist() if len(pa_mpjpe) > 0 else []

    if len(mpjpe) <= 0:
        return "Failed to evaluate. Keep training!"
    
    precision, recall, f1 = compute_prf1(total_count,total_miss_count,total_fp)
    error_dict = {}
    error_dict['recall'] = recall

    error_dict['MPJPE'] = round(float(sum(mpjpe)/len(mpjpe)), 1)
    error_dict['PA-MPJPE'] = round(float(sum(pa_mpjpe)/len(pa_mpjpe)), 1)
    error_dict['MVE'] = round(float(sum(mve)/len(mve)), 1)
    error_dict['PA-MVE'] = round(float(sum(pa_mve)/len(pa_mve)), 1)

    if accelerator.is_main_process:
        with open(os.path.join(results_save_path,'results.txt'),'w') as f:
            for k,v in error_dict.items():
                f.write(f'{k}: {v}\n')

    return error_dict




def evaluate_cmu(model, eval_dataloader, conf_thresh,
                 vis=False, vis_step=40, results_save_path=None,
                 distributed=False, accelerator=None):
    """
    Evaluate CMU Panoptic dataset strictly following ROMP's evaluation_results.
    """
    assert results_save_path is not None
    assert accelerator is not None

    os.makedirs(results_save_path, exist_ok=True)
    cur_device = next(model.parameters()).device

    def _romp_get_bbx_overlap(p1, p2):
        min_p1 = np.min(p1, axis=0)
        min_p2 = np.min(p2, axis=0)
        max_p1 = np.max(p1, axis=0)
        max_p2 = np.max(p2, axis=0)

        bb1 = {}
        bb2 = {}
        bb1['x1'] = min_p1[0]
        bb1['x2'] = max_p1[0]
        bb1['y1'] = min_p1[1]
        bb1['y2'] = max_p1[1]
        bb2['x1'] = min_p2[0]
        bb2['x2'] = max_p2[0]
        bb2['y1'] = min_p2[1]
        bb2['y2'] = max_p2[1]

        x_left = max(bb1['x1'], bb2['x1'])
        y_top = max(bb1['y1'], bb2['y1'])
        x_right = min(bb1['x2'], bb2['x2'])
        y_bottom = min(bb1['y2'], bb2['y2'])

        intersection_area = max(0, x_right - x_left + 1) * max(0, y_bottom - y_top + 1)
        bb1_area = (bb1['x2'] - bb1['x1'] + 1) * (bb1['y2'] - bb1['y1'] + 1)
        bb2_area = (bb2['x2'] - bb2['x1'] + 1) * (bb2['y2'] - bb2['y1'] + 1)
        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        return iou

    def _romp_l2_error(j1, j2):
        return np.linalg.norm(j1 - j2, 2)

    def _romp_match_2d_greedy(pred_kps, gtkp, valid_mask, iou_thresh=0.1):
        from itertools import product

        predList = np.arange(len(pred_kps))
        gtList = np.arange(len(gtkp))
        combs = list(product(predList, gtList))

        errors_per_pair_list = []
        for comb in combs:
            vmask = valid_mask[comb[1]]
            assert vmask.sum() > 0, 'no valid gt keypoints'
            errors_per_pair_list.append(
                _romp_l2_error(pred_kps[comb[0]][vmask, :2], gtkp[comb[1]][vmask, :2])
            )

        gtAssigned = np.zeros((len(gtkp),), dtype=bool)
        opAssigned = np.zeros((len(pred_kps),), dtype=bool)
        errors_per_pair_list = np.array(errors_per_pair_list)

        bestMatch = []
        falsePositiveCounter = 0
        while (
            np.sum(gtAssigned) < len(gtAssigned)
            and np.sum(opAssigned) + falsePositiveCounter < len(pred_kps)
        ):
            found = False
            falsePositive = False
            while not found:
                if all(np.isinf(errors_per_pair_list)):
                    break
                minIdx = np.argmin(errors_per_pair_list)
                minComb = combs[minIdx]
                iou = _romp_get_bbx_overlap(
                    pred_kps[minComb[0]], gtkp[minComb[1]]
                )
                if (
                    not opAssigned[minComb[0]]
                    and not gtAssigned[minComb[1]]
                    and iou >= iou_thresh
                ):
                    found = True
                    errors_per_pair_list[minIdx] = np.inf
                else:
                    errors_per_pair_list[minIdx] = np.inf
                    if iou < iou_thresh:
                        found = True
                        falsePositive = True
                        falsePositiveCounter += 1

            if not found:
                break

            if not falsePositive:
                bestMatch.append(minComb)
                opAssigned[minComb[0]] = True
                gtAssigned[minComb[1]] = True

        bestMatch = np.array(bestMatch)
        opAssigned_ids = []
        gtAssigned_ids = []
        for pair in bestMatch:
            opAssigned_ids.append(pair[0])
            gtAssigned_ids.append(pair[1])
        opAssigned_ids.sort()
        gtAssigned_ids.sort()

        falsePositives = []
        misses = []

        opIds = np.arange(len(pred_kps))
        notAssignedIds = np.setdiff1d(opIds, opAssigned_ids)
        for notAssignedId in notAssignedIds:
            falsePositives.append(notAssignedId)

        gtIds = np.arange(len(gtList))
        notAssignedIdsGt = np.setdiff1d(gtIds, gtAssigned_ids)
        for notAssignedIdGt in notAssignedIdsGt:
            misses.append(notAssignedIdGt)

        return bestMatch, falsePositives, misses

    H36M17_TO_J14 = list(H36M_EVAL_JOINTS)
    action_names = ['haggling1', 'mafia2', 'ultimatum1', 'pizza1']
    missing_punish = 150

    smpl_layer = model.human_model
    smpl2h36m_regressor = torch.from_numpy(
        smpl_layer.smpl2h36m_regressor
    ).float().to(cur_device)

    dataset_obj = getattr(eval_dataloader, 'dataset', None)
    dataset_raw_persons = getattr(dataset_obj, 'total_persons_raw', None)
    dataset_visible_persons = getattr(dataset_obj, 'total_persons_visible', None)

    matched_total_sum = torch.zeros((1,), dtype=torch.float64, device=cur_device)
    matched_total_cnt = torch.zeros((1,), dtype=torch.int64, device=cur_device)
    miss_penalty_cnt = torch.zeros((1,), dtype=torch.int64, device=cur_device)
    total_gt_persons = torch.zeros((1,), dtype=torch.int64, device=cur_device)
    total_false_positives = torch.zeros((1,), dtype=torch.int64, device=cur_device)
    action_mpjpe_sum = torch.zeros((len(action_names),), dtype=torch.float64, device=cur_device)
    action_mpjpe_cnt = torch.zeros((len(action_names),), dtype=torch.int64, device=cur_device)
    action_penalty_cnt = torch.zeros((len(action_names),), dtype=torch.int64, device=cur_device)
    step = 0

    progress_bar = tqdm(
        total=len(eval_dataloader),
        disable=not accelerator.is_local_main_process,
        desc='evaluate_cmu',
    )

    for samples, targets in eval_dataloader:
        samples = [sample.to(device=cur_device, non_blocking=True) for sample in samples]
        with torch.no_grad():
            outputs = model(samples, targets)

        bs = len(targets)
        for idx in range(bs):
            img_path = targets[idx].get('img_path', '')
            img_key = img_path.replace('\\', '/')
            action_idx = None
            for aidx, aname in enumerate(action_names):
                if aname in img_key:
                    action_idx = aidx
                    break

            
            gt_kp2d = targets[idx]['gt_kp2d_j14'].cpu().numpy()
            if 'gt_kp2d_vis_j14' in targets[idx]:
                valid_mask = targets[idx]['gt_kp2d_vis_j14'].cpu().numpy().astype(bool)
            else:
                valid_mask = gt_kp2d[:, :, 0] > -2.
            kp3d_gt = targets[idx]['gt_kp3d_j14'].cpu().numpy()

            if kp3d_gt.shape[0] == 0:
                continue

            visible_kpts = kp3d_gt[:, :, 0] > -2.
            kp3d_gts = kp3d_gt.copy()

            valid_ids = valid_mask.sum(-1) != 0
            kp2d_gts = gt_kp2d[valid_ids]
            kp3d_gts = kp3d_gts[valid_ids]
            valid_mask_sel = valid_mask[valid_ids]
            visible_kpts = visible_kpts[valid_ids]
            if len(kp3d_gts) == 0:
                continue
            total_gt_persons += int(len(kp3d_gts))

            select_queries_idx = torch.where(outputs['pred_confs'][idx] > conf_thresh)[0]
            matched_person_errors = np.zeros((0,), dtype=np.float32)
            miss_count = len(kp3d_gts)

            kp3d_preds = np.zeros((0, len(H36M17_TO_J14), 3), dtype=np.float32)
            kp2d_preds = np.zeros((0, len(H36M17_TO_J14), 2), dtype=np.float32)

            if len(select_queries_idx) != 0:
                pred_verts = outputs['pred_verts'][idx][select_queries_idx].detach()
                pred_transl = outputs['pred_transl'][idx][select_queries_idx].detach()
                pred_intrinsics = (
                    outputs['pred_intrinsics'][idx].detach().cpu().numpy().reshape(3, 3)
                )

                pred_j3ds_h36m17 = torch.einsum(
                    'bik,ji->bjk',
                    [pred_verts - pred_transl[:, None, :], smpl2h36m_regressor]
                ) + pred_transl[:, None, :]
                pred_j3ds_h36m17 = pred_j3ds_h36m17.detach().cpu().numpy()
                pred_root = pred_j3ds_h36m17[:, [0]]
                kp3d_preds = pred_j3ds_h36m17[:, H36M17_TO_J14] - pred_root

                pred_j2d_homo = np.matmul(pred_j3ds_h36m17, pred_intrinsics.T)
                pred_j2ds_h36m17 = pred_j2d_homo[:, :, :2] / (
                    pred_j2d_homo[:, :, 2:3] + 1e-6
                )
                kp2d_preds = pred_j2ds_h36m17[:, H36M17_TO_J14]

            if kp2d_preds.shape[0] == 0:
                bestMatch = np.zeros((0, 2), dtype=np.int64)
                falsePositives = []
                misses = list(range(len(kp2d_gts)))
            else:
                bestMatch, falsePositives, misses = _romp_match_2d_greedy(
                    kp2d_preds, kp2d_gts, valid_mask_sel, iou_thresh=0.05
                )

            if len(bestMatch) > 0:
                bestMatch = np.array(bestMatch)
                pids, gids = bestMatch[:, 0], bestMatch[:, 1]
                per_joint_errors = (
                    np.sqrt(((kp3d_preds[pids] - kp3d_gts[gids]) ** 2).sum(-1))
                    * visible_kpts[gids]
                ) * 1000
                matched_person_errors = per_joint_errors.mean(-1)
                miss_count = len(misses)
            else:
                matched_person_errors = np.zeros((0,), dtype=np.float32)
                miss_count = len(kp3d_gts)
            total_false_positives += int(len(falsePositives))

            if matched_person_errors.size > 0:
                mpjpe_sum_val = float(matched_person_errors.sum())
                mpjpe_cnt_val = int(len(matched_person_errors))
                matched_total_sum += mpjpe_sum_val
                matched_total_cnt += mpjpe_cnt_val
                if action_idx is not None:
                    action_mpjpe_sum[action_idx] += mpjpe_sum_val
                    action_mpjpe_cnt[action_idx] += mpjpe_cnt_val

            if miss_count > 0:
                miss_penalty_cnt += int(miss_count)
                if action_idx is not None:
                    action_penalty_cnt[action_idx] += int(miss_count)

        progress_bar.update(1)

    if distributed:
        matched_total_sum = accelerator.gather_for_metrics(matched_total_sum).sum(dim=0)
        matched_total_cnt = accelerator.gather_for_metrics(matched_total_cnt).sum(dim=0)
        miss_penalty_cnt = accelerator.gather_for_metrics(miss_penalty_cnt).sum(dim=0)
        total_gt_persons = accelerator.gather_for_metrics(total_gt_persons).sum(dim=0)
        total_false_positives = accelerator.gather_for_metrics(total_false_positives).sum(dim=0)
        action_mpjpe_sum = accelerator.gather_for_metrics(action_mpjpe_sum[None, :]).sum(dim=0)
        action_mpjpe_cnt = accelerator.gather_for_metrics(action_mpjpe_cnt[None, :]).sum(dim=0)
        action_penalty_cnt = accelerator.gather_for_metrics(action_penalty_cnt[None, :]).sum(dim=0)

    matched_cnt = int(matched_total_cnt.item())
    miss_cnt_total = int(miss_penalty_cnt.item())
    penalized_total_sum = matched_total_sum + miss_penalty_cnt.to(matched_total_sum.dtype) * missing_punish
    penalized_total_cnt = matched_total_cnt + miss_penalty_cnt
    penalized_cnt = int(penalized_total_cnt.item())
    total_gt_cnt = int(total_gt_persons.item())
    total_fp_cnt = int(total_false_positives.item())
    precision = matched_cnt / max(matched_cnt + total_fp_cnt, 1) if (matched_cnt + total_fp_cnt) > 0 else 0.0
    recall = matched_cnt / max(total_gt_cnt, 1) if total_gt_cnt > 0 else 0.0
    f1 = (2 * precision * recall / max(precision + recall, 1e-8)) if (precision + recall) > 0 else 0.0


    error_dict = {
        'MPJPE_matched': round(float((matched_total_sum / matched_total_cnt).item()), 1) if matched_cnt > 0 else None,
        'MPJPE_with_penalty': round(float((penalized_total_sum / penalized_total_cnt).item()), 1) if penalized_cnt > 0 else None,
        'num_raw_persons': int(dataset_raw_persons) if dataset_raw_persons is not None else total_gt_cnt,
        'num_visible_persons': int(dataset_visible_persons) if dataset_visible_persons is not None else total_gt_cnt,
        'num_matched_persons': matched_cnt,
    }

    for aidx, aname in enumerate(action_names):
        cnt = int(action_mpjpe_cnt[aidx].item())
        if cnt > 0:
            mean_val = float((action_mpjpe_sum[aidx] / action_mpjpe_cnt[aidx]).item())
            error_dict[f'MPJPE_{aname}'] = round(mean_val, 1)
        miss_cnt = int(action_penalty_cnt[aidx].item())
        if miss_cnt > 0:
            error_dict[f'MissCount_{aname}'] = miss_cnt

    if accelerator.is_main_process:
        print("[CMU Evaluation] Summary:")
        for k, v in error_dict.items():
            print(f"  {k}: {v}")
        with open(os.path.join(results_save_path, 'results.txt'), 'w') as f:
            for k, v in error_dict.items():
                f.write(f'{k}: {v}\n')
    return error_dict



