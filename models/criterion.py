#bbox+roi-stage1/2

# Modified from DAB-DETR (https://github.com/IDEA-Research/DAB-DETR)
import os
import torch
import numpy as np
import math
from math import tan, pi
from torch import nn
import torch.nn.functional as F
from utils import box_ops
from utils.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

def focal_loss(inputs, targets, valid_mask = None, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    # prob = inputs.sigmoid()
    # ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    prob = inputs
    ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    
    return loss.mean()

class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, matcher, weight_dict, losses = ['confs','boxes', 'poses','betas', 'j3ds','j2ds', 'depths'], 
                focal_alpha=0.25, focal_gamma = 2.0, j2ds_norm_scale = 518, input_size = 672, FOV = pi/3):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
            input_size: model input image size for focal length calculation
            FOV: field of view for focal length calculation
        """
        super().__init__()
        self.matcher = matcher
        self.losses = losses
        if 'boxes' in losses and 'giou' not in weight_dict:
            weight_dict.update({'giou': weight_dict['boxes']})
        
        # Auto-add bbox_prior_giou weight if bbox_prior_boxes is present
        if 'bbox_prior_boxes' in losses and 'bbox_prior_giou' not in weight_dict:
            if 'bbox_prior_boxes' in weight_dict:
                weight_dict.update({'bbox_prior_giou': weight_dict['bbox_prior_boxes']})
        
        self.weight_dict = weight_dict
        

        self.betas_weight = torch.tensor([2.56, 1.28, 0.64, 0.64, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32]).unsqueeze(0).float()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.j2ds_norm_scale = j2ds_norm_scale
        self.device = None
        
        self.preset_focal = input_size / (2 * tan(FOV / 2))


    def loss_boxes(self, loss, outputs, targets, indices, num_instances, **kwargs):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        assert loss == 'boxes'
        idx = self._get_src_permutation_idx(indices)
        valid_idx = torch.where(torch.cat([torch.ones(len(i), dtype=bool, device = self.device)*(loss in t) for t, (_, i) in zip(targets, indices)]))[0]
        
        if len(valid_idx) == 0:
            return {loss: torch.tensor(0.).to(self.device)}

        src = outputs['pred_'+loss][idx][valid_idx]
        target = torch.cat([t[loss][i] for t, (_, i) in zip(targets, indices) if loss in t], dim=0)
        assert src.shape == target.shape
        
        src_boxes = src
        target_boxes = target

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['boxes'] = loss_bbox.sum() / num_instances

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['giou'] = loss_giou.sum() / num_instances



        return losses

    # For computing ['boxes', 'poses', 'betas', 'j3ds', 'j2ds'] losses
    def loss_L1(self, loss, outputs, targets, indices, num_instances, **kwargs):
        idx = self._get_src_permutation_idx(indices)
        valid_idx = torch.where(torch.cat([torch.ones(len(i), dtype=bool, device = self.device)*(loss in t) for t, (_, i) in zip(targets, indices)]))[0]
        
        if len(valid_idx) == 0:
            return {loss: torch.tensor(0.).to(self.device)}

        src = outputs['pred_'+loss][idx][valid_idx]
        target = torch.cat([t[loss][i] for t, (_, i) in zip(targets, indices) if loss in t], dim=0)
        assert src.shape == target.shape

        losses = {}
        loss_mask = None

        if loss == 'j3ds':
            # Root aligned
            src = src - src[...,[0],:].clone()
            target = target - target[...,[0],:].clone()
            # Use 54 smpl joints
            src = src[:,:54,:]
            target = target[:,:54,:]
        elif loss == 'j2ds':
            src = src / self.j2ds_norm_scale
            target = target / self.j2ds_norm_scale
            # Need to exclude invalid kpts in 2d datasets
            loss_mask = torch.cat([t['j2ds_mask'][i] for t, (_, i) in zip(targets, indices) if 'j2ds' in t], dim=0)
            # Use 54 smpl joints
            src = src[:,:54,:]
            target = target[:,:54,:]
            loss_mask = loss_mask[:,:54,:]
        
        valid_loss = torch.abs(src-target)

        
        if loss_mask is not None:
            valid_loss = valid_loss * loss_mask
        if loss == 'betas':
            valid_loss = valid_loss*self.betas_weight.to(src.device)
        
        losses[loss] = valid_loss.flatten(1).mean(-1).sum()/num_instances



        return losses


    def loss_confs(self, loss, outputs, targets, indices, num_instances, is_dn=False, **kwargs):
        assert loss == 'confs'
        idx = self._get_src_permutation_idx(indices)
        pred_confs = outputs['pred_'+loss]

        with torch.no_grad():
            labels = torch.zeros_like(pred_confs)
            labels[idx] = 1
            detection_valid_mask = torch.zeros_like(pred_confs,dtype=bool)
            detection_valid_mask[idx] = True
            valid_batch_idx = torch.where(torch.tensor([t['detect_all_people'] for t in targets]))[0]
            detection_valid_mask[valid_batch_idx] = True

        
        losses = {}
        if is_dn:
            losses[loss] = focal_loss(pred_confs, labels) / num_instances
        else:
            losses[loss] = focal_loss(pred_confs, labels, valid_mask = detection_valid_mask) / num_instances

        return losses

    # l1-loss*1/GT_z
    def loss_absolute_depths(self, loss, outputs, targets, indices, num_instances, **kwargs):
        assert loss == 'depths' 
        losses = {}
        idx = self._get_src_permutation_idx(indices)
        valid_idx = torch.where(torch.cat([torch.ones(len(i), dtype=bool, device = self.device)*(loss in t) for t, (_, i) in zip(targets, indices)]))[0]
        
        if len(valid_idx) == 0:
            # Keep zero loss dependent on model outputs so gradients/buckets
            # remain consistent across ranks even when no valid GT exists.
            dummy = outputs['pred_'+loss]
            losses[loss] = dummy.sum() * 0.0
            return losses

        # HMR depth predictions: use the second channel [z/f] and convert back
        # to absolute depth using the GT focal length: z_pred = (z_over_f) * f_gt.
        src = outputs['pred_'+loss][idx][valid_idx][...,[1]]  # [z/f]
        target = torch.cat([t[loss][i] for t, (_, i) in zip(targets, indices) if loss in t], dim=0)[...,[0]]  # [z_gt]
        target_focals = torch.cat([t['focals'][i] for t, (_, i) in zip(targets, indices) if loss in t], dim=0)

        # Restore absolute depth for predictions
        src = target_focals * src  # z_pred

        assert src.shape == target.shape

        # Symmetric relative depth error, matching the style of depth_tz loss:
        #   L = |z_pred - z_gt| * (1 / z_gt)
        abs_diff = torch.abs(src - target)
        weight = 1.0 / (target + 1e-6)
        weighted_loss = abs_diff * weight

        losses[loss] = weighted_loss.flatten(1).mean(-1).sum()/num_instances
        return losses
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_instances, **kwargs):
        loss_map = {
            'confs': self.loss_confs,
            'boxes': self.loss_boxes,
            'poses': self.loss_L1,
            'betas': self.loss_L1,
            'j3ds': self.loss_L1,
            'j2ds': self.loss_L1,
            'depths': self.loss_absolute_depths,
            # Bbox-only prior DETR losses
            'bbox_prior_boxes': self.loss_bbox_prior_boxes,
            'bbox_prior_conf': self.loss_bbox_prior_conf,
            # Depth-prior Tz loss (DAV2 depth-only decoder)
            'depth_tz': self.loss_depth_tz,
        }
        # assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](loss, outputs, targets, indices, num_instances, **kwargs)


    def get_valid_instances(self, targets):
        # Compute the average number of GTs accross all nodes, for normalization purposes
        num_valid_instances = {}
        for loss in self.losses:
            num_instances = 0
            for t in targets:
                num_instances += t['pnum'] if loss in t else 0
            num_instances = torch.as_tensor([num_instances], dtype=torch.float, device=self.device)
            if is_dist_avail_and_initialized():
                torch.distributed.all_reduce(num_instances)
            num_instances = torch.clamp(num_instances / get_world_size(), min=1).item()
            num_valid_instances[loss] = num_instances
        num_valid_instances['confs'] = 1.
        # Align depth-prior losses with main HMR losses
        # 1) depth_tz / bbox_prior_boxes: always use the same per-person normalization as 'boxes'
        #    Even when 'boxes' is not in self.losses (e.g., Stage-1 depth-only), we still
        #    compute the average number of persons from targets['pnum'] to keep the
        #    semantics consistent with the original boxes loss.
        if 'bbox_prior_boxes' in self.losses or 'depth_tz' in self.losses:
            if 'boxes' in num_valid_instances:
                boxes_instances = num_valid_instances['boxes']
            else:
                # Compute person count directly from targets when 'boxes' is not in losses
                boxes_instances = 0
                for t in targets:
                    boxes_instances += t.get('pnum', 0)
                boxes_instances = torch.as_tensor([boxes_instances], dtype=torch.float, device=self.device)
                if is_dist_avail_and_initialized():
                    torch.distributed.all_reduce(boxes_instances)
                boxes_instances = torch.clamp(boxes_instances / get_world_size(), min=1).item()

            # Bbox-only prior boxes follow the same normalization as 'boxes'
            if 'bbox_prior_boxes' in self.losses:
                num_valid_instances['bbox_prior_boxes'] = boxes_instances
            # Depth-prior Tz loss also follows the same normalization
            if 'depth_tz' in self.losses:
                num_valid_instances['depth_tz'] = boxes_instances
        # bbox_prior_conf also uses fixed normalization
        if 'bbox_prior_conf' in self.losses:
            num_valid_instances['bbox_prior_conf'] = 1.

        return num_valid_instances

    def prep_for_dn(self, dn_meta):
        output_known = dn_meta['output_known']
        num_dn_groups, pad_size = dn_meta['num_dn_group'], dn_meta['pad_size']
        assert pad_size % num_dn_groups == 0
        single_pad = pad_size//num_dn_groups

        return output_known, single_pad, num_dn_groups

    def forward(self, outputs, targets, step=-1):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
             step: training step number for debugging
        """
        # remove invalid information in targets
        for t in targets:
            if not t['3d_valid']:
                for key in ['betas', 'poses', 'j3ds', 'depths', 'focals']:
                    if key in t:
                        del t[key]

        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs' and k != 'sat'}
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        
        # Retrieve independent matching for bbox-only prior outputs if present
        bbox_prior_indices = None
        if 'bbox_prior_outputs' in outputs and outputs['bbox_prior_outputs'] is not None:
            prior_out = outputs['bbox_prior_outputs']
            prior_matcher_input = {
                'pred_boxes': prior_out['pred_boxes'][-1],
                'pred_confs': prior_out['pred_confs'][-1],
            }
            bbox_prior_indices = self.matcher(prior_matcher_input, targets)

        if 'pred_poses' in outputs:
            self.device = outputs['pred_poses'].device
        else:
            self.device = outputs['pred_boxes'].device
        num_valid_instances = self.get_valid_instances(targets)

        # Compute all the requested losses
        losses = {}
        
        # prepare for dn loss
        if 'dn_meta' in outputs:
            dn_meta = outputs['dn_meta']
            output_known, single_pad, scalar = self.prep_for_dn(dn_meta)

            dn_pos_idx = []
            dn_neg_idx = []
            for i in range(len(targets)):
                assert len(targets[i]['boxes']) > 0
                # t = torch.range(0, len(targets[i]['labels']) - 1).long().to(self.device)
                t = torch.arange(0, len(targets[i]['labels'])).long().to(self.device)
                t = t.unsqueeze(0).repeat(scalar, 1)
                tgt_idx = t.flatten()
                output_idx = (torch.tensor(range(scalar)) * single_pad).long().to(self.device).unsqueeze(1) + t
                output_idx = output_idx.flatten()

                dn_pos_idx.append((output_idx, tgt_idx))
                dn_neg_idx.append((output_idx + single_pad // 2, tgt_idx))

            l_dict = {}
            # Only apply DN losses to main HMR branch losses
            hmr_dn_losses = {"confs", "boxes", "poses", "betas", "j3ds", "j2ds", "depths"}
            for loss in self.losses:
                if loss not in hmr_dn_losses:
                    continue
                l_dict.update(self.get_loss(loss, output_known, targets, dn_pos_idx, num_valid_instances[loss]*scalar, is_dn=True))

            l_dict = {k + f'_dn': v for k, v in l_dict.items()}
            losses.update(l_dict)
        
        for loss in self.losses:
            # Default: use main HMR matching indices
            curr_indices = indices
            # Route bbox-prior and depth-prior Tz losses to bbox_prior_indices
            if loss in ['bbox_prior_boxes', 'bbox_prior_conf', 'depth_tz']:
                if bbox_prior_indices is not None:
                    curr_indices = bbox_prior_indices
            
            losses.update(self.get_loss(loss, outputs, targets, curr_indices, num_valid_instances[loss], step=step))
        
        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_valid_instances[loss], step=step)
                    l_dict = {f'{k}.{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

                if 'dn_meta' in outputs:
                    aux_outputs_known = output_known['aux_outputs'][i]
                    l_dict={}
                    for loss in self.losses:
                        if loss not in hmr_dn_losses:
                            continue
                        l_dict.update(self.get_loss(loss, aux_outputs_known, targets, dn_pos_idx, num_valid_instances[loss]*scalar, is_dn=True))
                    l_dict = {k + f'_dn.{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # Auxiliary losses for bbox-only prior DETR decoder layers
        if (
            bbox_prior_indices is not None
            and 'bbox_prior_outputs' in outputs
            and outputs['bbox_prior_outputs'] is not None
        ):
            prior_out = outputs['bbox_prior_outputs']
            num_prior_layers = prior_out['pred_boxes'].shape[0]

            # Follow DETR convention: last layer is main, previous layers are auxiliary
            if num_prior_layers > 1:
                for i in range(num_prior_layers - 1):
                    if 'bbox_prior_boxes' in self.losses:
                        l_dict_boxes = self.loss_bbox_prior_boxes(
                            'bbox_prior_boxes',
                            outputs,
                            targets,
                            bbox_prior_indices,
                            num_valid_instances['bbox_prior_boxes'],
                            layer_idx=i,
                        )
                        l_dict_boxes = {f'{k}.prior.{i}': v for k, v in l_dict_boxes.items()}
                        losses.update(l_dict_boxes)

                    if 'bbox_prior_conf' in self.losses:
                        l_dict_conf = self.loss_bbox_prior_conf(
                            'bbox_prior_conf',
                            outputs,
                            targets,
                            bbox_prior_indices,
                            num_valid_instances['bbox_prior_conf'],
                            layer_idx=i,
                        )
                        l_dict_conf = {f'{k}.prior.{i}': v for k, v in l_dict_conf.items()}
                        losses.update(l_dict_conf)

        return losses    

    # ========== Bbox-only Prior DETR Losses ==========
    def loss_bbox_prior_boxes(self, loss, outputs, targets, indices, num_instances, layer_idx=-1, **kwargs):
        assert loss == 'bbox_prior_boxes'
        losses = {}

        if 'bbox_prior_outputs' not in outputs or outputs['bbox_prior_outputs'] is None:
            return {loss: torch.tensor(0.).to(self.device), 'bbox_prior_giou': torch.tensor(0.).to(self.device)}

        prior_out = outputs['bbox_prior_outputs']
        # Select specific decoder layer (default: last)
        src_boxes_layer = prior_out['pred_boxes'][layer_idx]

        # Reuse existing loss_boxes implementation via a fake outputs dict
        fake_outputs = {'pred_boxes': src_boxes_layer}
        base_losses = self.loss_boxes('boxes', fake_outputs, targets, indices, num_instances)

        losses['bbox_prior_boxes'] = base_losses['boxes']
        losses['bbox_prior_giou'] = base_losses['giou']
        return losses

    def loss_bbox_prior_conf(self, loss, outputs, targets, indices, num_instances, layer_idx=-1, **kwargs):
        assert loss == 'bbox_prior_conf'
        losses = {}

        if 'bbox_prior_outputs' not in outputs or outputs['bbox_prior_outputs'] is None:
            return {loss: torch.tensor(0.).to(self.device)}

        prior_out = outputs['bbox_prior_outputs']
        pred_confs_layer = prior_out['pred_confs'][layer_idx]

        fake_outputs = {'pred_confs': pred_confs_layer}
        base_losses = self.loss_confs('confs', fake_outputs, targets, indices, num_instances)

        losses['bbox_prior_conf'] = base_losses['confs']
        return losses

    # ========== Depth-prior Tz Loss (DAV2 depth-only decoder) ==========
    def loss_depth_tz(self, loss, outputs, targets, indices, num_instances, layer_idx=-1, **kwargs):
        """Depth-prior Tz loss for root depth, using DAV2-based decoder outputs."""
        assert loss == 'depth_tz'
        losses = {}

        if 'depth_tz_outputs' not in outputs or outputs['depth_tz_outputs'] is None:
            return {loss: torch.tensor(0.).to(self.device)}

        idx = self._get_src_permutation_idx(indices)
        valid_idx = torch.where(torch.cat([
            torch.ones(len(i), dtype=bool, device=self.device) * ('depths' in t)
            for t, (_, i) in zip(targets, indices)
        ]))[0]

        if len(valid_idx) == 0:
            # Keep the graph dependent on depth-prior Tz outputs even when
            # this rank has no valid depth GT.
            dummy = outputs['depth_tz_outputs']['pred_tz']
            losses[loss] = dummy.sum() * 0.0
            return losses

        # Get predictions from depth-prior Tz decoder
        pred_tz = outputs['depth_tz_outputs']['pred_tz'][idx][valid_idx]  # [N, 1]

        # Get GT Tz from existing depths target (SMPL root depth)
        gt_depths = torch.cat([
            t['depths'][i] for t, (_, i) in zip(targets, indices)
            if 'depths' in t
        ], dim=0)  # [N, 2]
        gt_tz = gt_depths[..., [0]]  # [N, 1]

        # BLADE-style absolute depth L1 loss with 1/gt_z weighting
        abs_diff = torch.abs(pred_tz - gt_tz)  # [N, 1]
        # Per-instance weight: 1 / gt_z, clipped to avoid division by zero
        weight = 1.0 / (gt_tz + 1e-6)         # [N, 1]

        # Apply per-instance weighting, then follow the same per-person
        # normalization convention as other losses: sum over instances
        # divided by num_instances (average per person across the batch).
        weighted_loss = abs_diff * weight    # [N, 1]
        losses[loss] = weighted_loss.flatten(1).mean(-1).sum() / num_instances

        return losses
