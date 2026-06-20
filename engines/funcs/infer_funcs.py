import os
import pickle
from tqdm.auto import tqdm
import torch
import numpy as np
from utils.transforms import unNormalize
from utils.visualization import tensor_to_BGR, pad_img
from utils.visualization import vis_meshes_img, get_colors_rgb, vis_meshes_topview
from utils.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
import time
import cv2
import trimesh

def inference(model, infer_dataloader, conf_thresh, results_save_path = None,
                        distributed = False, accelerator = None):
    assert results_save_path is not None
    assert accelerator is not None

    accelerator.print(f'Results will be saved at: {results_save_path}')
    os.makedirs(results_save_path,exist_ok=True)
    cur_device = next(model.parameters()).device
    smpl_layer = model.human_model

    progress_bar = tqdm(total=len(infer_dataloader), disable=not accelerator.is_local_main_process)
    progress_bar.set_description('inference')
    
    total_inference_time = 0
    total_process_time = 0
    total_samples = 0
    
    if accelerator.is_local_main_process:
        detail_time_file = os.path.join(results_save_path, 'detailed_time_stats.txt')
        with open(detail_time_file, 'w') as f:
            f.write('image_name,inference_time_ms,process_time_ms\n')
    
    for itr, (samples, targets) in enumerate(infer_dataloader):
        process_start_time = time.time()
        samples=[sample.to(device = cur_device, non_blocking = True) for sample in samples]
        
        with torch.no_grad():    
           inference_start_time = time.time()
           outputs = model(samples, targets)
           inference_end_time = time.time()
           batch_inference_time = inference_end_time - inference_start_time
           total_inference_time += batch_inference_time
           
        bs = len(targets)
        total_samples += bs
        
        for idx in range(bs):
            img_size = targets[idx]['img_size'].detach().cpu().int().numpy()
            img_name = targets[idx]['img_path'].split('/')[-1].split('.')[0]

            #pred
            select_queries_idx = torch.where(outputs['pred_confs'][idx] > conf_thresh)[0]
            pred_verts = outputs['pred_verts'][idx][select_queries_idx].detach().cpu().numpy()

            ori_img = tensor_to_BGR(unNormalize(samples[idx]).cpu())
            ori_img[img_size[0]:,:,:] = 255
            ori_img[:,img_size[1]:,:] = 255
            ori_img[img_size[0]:,img_size[1]:,:] = 255
            ori_img = pad_img(ori_img, model.input_size, pad_color_offset=255)


            colors = get_colors_rgb(len(pred_verts))
            pred_mesh_img = vis_meshes_img(img = ori_img.copy(),
                                        verts = pred_verts,
                                        smpl_faces = smpl_layer.faces,
                                        cam_intrinsics = outputs['pred_intrinsics'][idx].reshape(3,3).detach().cpu(),
                                        colors=colors)[:img_size[0],:img_size[1]]
            

            ori_canvas = pad_img(ori_img, model.input_size, pad_color_offset=255)

            mesh_overlay = vis_meshes_img(
                img=ori_canvas.copy(),
                verts=pred_verts,
                smpl_faces=smpl_layer.faces,
                cam_intrinsics=outputs['pred_intrinsics'][idx].reshape(3, 3).detach().cpu(),
                colors=colors,
            )
            mesh_canvas = pad_img(mesh_overlay, model.input_size, pad_color_offset=255)

            # top view (bird-eye)
            top_img = vis_meshes_topview(
                verts=pred_verts,
                smpl_faces=smpl_layer.faces,
                out_size=model.input_size,
                colors=colors,
                view="top",
            )

            side_img = vis_meshes_topview(
                verts=pred_verts,
                smpl_faces=smpl_layer.faces,
                out_size=model.input_size,
                colors=colors,
                view="side",
            )

            row1 = np.hstack([ori_canvas, mesh_canvas])
            row2 = np.hstack([top_img, side_img])
            full_img = np.vstack([row1, row2])

            cv2.imwrite(os.path.join(results_save_path, f'{img_name}.png'), full_img)

        process_end_time = time.time()
        batch_process_time = process_end_time - process_start_time
        total_process_time += batch_process_time
        
        if accelerator.is_local_main_process:
            for idx in range(bs):
                img_name = targets[idx]['img_path'].split('/')[-1].split('.')[0]
                with open(detail_time_file, 'a') as f:
                    f.write(f'{img_name},{(batch_inference_time/bs)*1000:.2f},{(batch_process_time/bs)*1000:.2f}\n')
        progress_bar.update(1)
            
    if accelerator.is_local_main_process:
        avg_inference_time = total_inference_time / total_samples
        avg_process_time = total_process_time / total_samples
        accelerator.print(f'\nAverage inference time per image: {avg_inference_time*1000:.2f} ms')
        accelerator.print(f'Average total processing time per image: {avg_process_time*1000:.2f} ms')
        
        time_stats = {
            'avg_inference_time_ms': avg_inference_time * 1000,
            'avg_process_time_ms': avg_process_time * 1000,
            'total_samples': total_samples
        }
        with open(os.path.join(results_save_path, 'time_stats.txt'), 'w') as f:
            for key, value in time_stats.items():
                f.write(f'{key}: {value:.2f}\n')
    








