import os
import io
import torch
import numpy as np
from termcolor import colored
try:
    #os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    import pyrender
except:
    print(colored('pyrender is not correctly imported.', 'red'))
    raise

import matplotlib
from matplotlib import colormaps
from matplotlib.colors import LightSource
import matplotlib.pyplot as plt
import math
import cv2
import trimesh
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R
import torchvision
from .transforms import adjust_colors

BASE_COLORS = np.loadtxt(os.path.abspath(os.path.join(__file__, "../colors.txt")), skiprows=0)/255.
BASE_COLORS = adjust_colors(BASE_COLORS,
                            saturation_threshold = 0.3, 
                            brightness_threshold = 0.8)


def get_colors_rgb(size):
    # np.random.seed(131)
    return BASE_COLORS[np.random.choice(BASE_COLORS.shape[0], size=size, replace=False)]

def tensor_to_BGR(img_tensor):
    img = img_tensor.numpy()*255
    img = img.astype(np.uint8).transpose((1,2,0))[:,:,::-1].copy()
    return img

def pad_img(img, pad_size = None, pad_color_offset = 127):
    if not isinstance(img, np.ndarray):
        img = tensor_to_BGR(img.detach().cpu())
    if pad_size is None:
        pad_size = max(img.shape[0],img.shape[1])

    pad = np.zeros((pad_size,pad_size,img.shape[-1]), dtype=img.dtype) + pad_color_offset
    pad[:img.shape[0], :img.shape[1]] = img.copy()
    return pad



def vis_meshes_img(img, verts, smpl_faces, cam_intrinsics, colors = None, padding = True):
    if not isinstance(img, np.ndarray):
        img = tensor_to_BGR(img.detach().cpu())

    if padding:
        pad_size = max(img.shape[0],img.shape[1])
        img = pad_img(img, pad_size)

    if colors is not None:
        assert len(colors) == len(verts)

    if len(cam_intrinsics.flatten()) == 9:
        cam_intrinsics = cam_intrinsics.reshape(3,3)
        rgb, depth = render_mesh(img.shape[0],img.shape[1],verts,smpl_faces,cam_intrinsics,colors)
        valid_mask = (depth > 0)[:,:,None] 
        visible_weight = 1.
        rendered_img = rgb[:,:,::-1] * valid_mask * visible_weight +\
                        img * valid_mask * (1-visible_weight)+\
                        img * (1-valid_mask)
    else:
        rendered_img = img
        for i, cam_int in enumerate(cam_intrinsics):
            rgb, depth = render_mesh(img.shape[0],img.shape[1],[verts[i]],smpl_faces,cam_int,colors)
            valid_mask = (depth > 0)[:,:,None] 
            visible_weight = 0.8
            rendered_img = rgb[:,:,::-1] * valid_mask * visible_weight +\
                            rendered_img * valid_mask * (1-visible_weight)+\
                            rendered_img * (1-valid_mask)
    rendered_img = rendered_img.astype(np.uint8)

    return rendered_img


def vis_joints_img(img, j2ds, colors=None):
    if not isinstance(img, np.ndarray):
        img = tensor_to_BGR(img.detach().cpu())
    
    img = img.copy()
    
    # SMPL Joint Connectivity (Simplified for visualization)
    # This covers the main body structure
    skeleton = [
        (0, 1), (0, 2), (0, 3), # Pelvis to hips and spine
        (1, 4), (4, 7), (7, 10), # Left leg
        (2, 5), (5, 8), (8, 11), # Right leg
        (3, 6), (6, 9), (9, 12), (12, 15), # Spine to head
        (9, 13), (13, 16), (16, 18), (18, 20), # Left arm
        (9, 14), (14, 17), (17, 19), (19, 21)  # Right arm
    ]
    
    if colors is None:
        colors = [(0, 255, 255)] * len(j2ds) # Default yellow
    elif not isinstance(colors[0], (list, tuple, np.ndarray)):
        colors = [colors] * len(j2ds)

    for person_idx, person_j2d in enumerate(j2ds):
        color = colors[person_idx]
        if isinstance(color, np.ndarray):
            color = tuple((color * 255).astype(int).tolist())
        
        # Draw joints
        for j in person_j2d:
            cv2.circle(img, (int(j[0]), int(j[1])), 4, color, -1)
        
        # Draw skeleton
        for start, end in skeleton:
            if start < len(person_j2d) and end < len(person_j2d):
                p1 = (int(person_j2d[start][0]), int(person_j2d[start][1]))
                p2 = (int(person_j2d[end][0]), int(person_j2d[end][1]))
                cv2.line(img, p1, p2, color, 2)
                
    return img


def vis_boxes(img, boxes, padding=True, color = (0,0,255)):
    if not isinstance(img, np.ndarray):
        img = tensor_to_BGR(img.detach().cpu())
    if padding:
        pad_size = max(img.shape[0],img.shape[1])
        img = pad_img(img, pad_size)
    
    for bbox in boxes:
        bbox = bbox.int().tolist()
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
            color=color, thickness = 2 )
    
    return img


def get_img_from_fig(fig, dpi=120):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, transparent=False, bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img



def vis_depth_map(depth_map, depth_range=(0.1, 50.0), colormap='viridis'):
    """Render a depth map as a BGR color image."""
    if hasattr(depth_map, 'detach'):
        depth_map = depth_map.detach().cpu().numpy()
    
    if depth_map.ndim == 3:
        depth_map = depth_map.squeeze()
    
    valid_mask = (depth_map > 1e-6) & (depth_map < 1000)
    depth_map_clean = depth_map.copy()
    
    min_depth, max_depth = depth_range
    depth_normalized = np.clip(
        (depth_map_clean - min_depth) / (max_depth - min_depth), 
        0, 1
    )
    
    depth_normalized[~valid_mask] = 0
    
    import matplotlib.cm as cm
    depth_cmap = cm.get_cmap(colormap)
    
    depth_colored = depth_cmap(depth_normalized)[:, :, :3]  # [H, W, 3] RGB
    
    depth_vis = (depth_colored[:, :, ::-1] * 255).astype(np.uint8)  # RGB -> BGR
    
    return depth_vis


def vis_depth_map_dav2(depth_map, grayscale=False):
    """Render depth using Depth-Anything-V2-style per-image normalization."""
    # to numpy float32
    if hasattr(depth_map, 'detach'):
        depth_map = depth_map.detach().cpu().numpy()
    if depth_map.ndim == 3:
        depth_map = depth_map.squeeze()
    depth_map = depth_map.astype(np.float32, copy=False)

    valid = depth_map > 0
    if np.any(valid):
        vmin = float(depth_map[valid].min())
        vmax = float(depth_map[valid].max())
    else:
        vmin, vmax = float(depth_map.min()), float(depth_map.max())

    if vmax > vmin:
        norm = (depth_map - vmin) / (vmax - vmin)
    else:
        norm = np.zeros_like(depth_map, dtype=np.float32)

    norm[~valid] = 0.0

    u8 = (norm * 255.0).clip(0, 255).astype(np.uint8)

    if grayscale:
        vis = np.repeat(u8[..., None], 3, axis=-1)
        return vis

    import matplotlib
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    vis_rgb = (cmap(u8.astype(np.float32) / 255.0)[..., :3] * 255.0).astype(np.uint8)
    vis_bgr = vis_rgb[..., ::-1]
    return vis_bgr

def render_mesh(height, width, meshes, face, cam_intrinsics, colors = None):
    
    # renderer
    scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3))
    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height, point_size=1.0)

    # light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
    light_pose = np.eye(4)
    light_pose[:3, 3] = np.array([0, -1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([0, 1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([1, 1, 2])
    scene.add(light, pose=light_pose)

    # mesh
    if colors is None:
        colors = get_colors_rgb(len(meshes))

    for i, mesh in enumerate(meshes):
        mesh = trimesh.Trimesh(mesh, face)
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode='OPAQUE', baseColorFactor=(*colors[i], 1.0))
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)

        scene.add(mesh, f'mesh_{i}')


    # camera
    f=np.array([cam_intrinsics[0,0],cam_intrinsics[1,1]])
    c=cam_intrinsics[0:2,2]
    camera = pyrender.camera.IntrinsicsCamera(fx=f[0], fy=f[1], cx=c[0], cy=c[1])
    scene.add(camera)

    # render
    rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    rgb = rgb[:,:,:3].astype(np.float32)
    renderer.delete()
    return rgb, depth


def _create_camera_pose(eye, target, up):
    """Create a right-handed camera pose matrix given eye, target and up vectors.

    This utility is used for fixed oblique/top-view renderings that do not rely
    on dataset camera extrinsics. It follows the OpenGL convention where the
    camera looks toward the -Z axis in its local coordinates.
    """
    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)

    forward = target - eye
    norm = np.linalg.norm(forward)
    if norm < 1e-6:
        forward = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        forward /= norm

    right = np.cross(forward, up)
    r_norm = np.linalg.norm(right)
    if r_norm < 1e-6:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        right /= r_norm

    up = np.cross(right, forward)

    pose = np.eye(4, dtype=np.float32)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = -forward
    pose[:3, 3] = eye
    return pose

def render_mesh_topview(height, width, meshes, face,
                        colors=None, view="oblique",
                        center=None, radius=None):
    """Render meshes from a fixed oblique/top view.

    This function is independent of dataset camera intrinsics/extrinsics and is
    intended for qualitative visualization of global spatial consistency (e.g.,
    multi-person depth ordering) in evaluation scripts.

    Args:
        height: output image height in pixels.
        width: output image width in pixels.
        meshes: list of (V, 3) vertex arrays in the same coordinate frame.
        face: SMPL faces array.
        colors: optional list of RGB tuples in [0, 1] per person.
        view: view type, currently supports "oblique" or "top".
    Returns:
        rgb: (H, W, 3) float32 array in [0, 255].
        depth: (H, W) float32 depth map.
    """
    scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3))
    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height, point_size=1.0)

    # lights
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
    light_pose = np.eye(4)
    light_pose[:3, 3] = np.array([0, -1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([0, 1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([1, 1, 2])
    scene.add(light, pose=light_pose)

    if colors is None:
        colors = get_colors_rgb(len(meshes))

    # Add meshes (same base rotation as render_mesh to keep orientation consistent).
    # IMPORTANT: accumulate points *after* applying the base rotation so that
    # the camera center and radius are computed in the same frame as rendered meshes.
    all_points = []
    for i, mesh in enumerate(meshes):
        mesh = np.asarray(mesh)
        if mesh.ndim != 2 or mesh.shape[1] != 3:
            continue

        tm = trimesh.Trimesh(mesh, face)
        base_rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        tm.apply_transform(base_rot)

        all_points.append(np.asarray(tm.vertices))
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode="OPAQUE",
            baseColorFactor=(*colors[i], 1.0),
        )
        tm = pyrender.Mesh.from_trimesh(tm, material=material, smooth=True)
        scene.add(tm, f"mesh_topview_{i}")

    if len(all_points) == 0:
        # Degenerate case: no valid vertices, return a blank image.
        rgb = np.zeros((height, width, 3), dtype=np.float32)
        depth = np.zeros((height, width), dtype=np.float32)
        renderer.delete()
        return rgb, depth

    all_points = np.concatenate(all_points, axis=0)
    if center is None:
        center = all_points.mean(axis=0)
    if radius is None:
        radius = np.linalg.norm(all_points - center, axis=1).max()
        if radius < 1e-3:
            radius = 1.0

    # Radius-based camera placement (close to the original implementation).
    # Slightly more top-down oblique view by shortening the front distance.
    if view == "top":
        eye = center + np.array([0.0, 2.5 * radius, 1e-3], dtype=np.float32)
        up = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    elif view == "side":
        eye = center + np.array([2.5 * radius, 0.0, 0.0], dtype=np.float32)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    else:  # oblique
        eye = center + np.array([0.0, 1.5 * radius, 2.0 * radius], dtype=np.float32)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    target = center
    cam_pose = _create_camera_pose(eye, target, up)

    # Slightly narrower FOV to further enlarge the meshes in the image
    camera = pyrender.PerspectiveCamera(
        yfov=np.deg2rad(45.0),
        aspectRatio=float(width) / float(height),
    )
    scene.add(camera, pose=cam_pose)

    # Composite RGBA render on a light gray background for better contrast
    rgba, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    rgb = rgba[:, :, :3].astype(np.float32)
    alpha = (rgba[:, :, 3:4].astype(np.float32) / 255.0)
    bg_color = np.full_like(rgb, 235.0, dtype=np.float32)
    rgb = rgb * alpha + bg_color * (1.0 - alpha)

    renderer.delete()
    return rgb, depth

def vis_meshes_topview(verts, smpl_faces, out_size,
                       colors=None, view="oblique",
                       center=None, radius=None):
    """Render SMPL meshes from an oblique/top view for qualitative inspection.

    Args:
        verts: list of (V, 3) vertex arrays or a single (N, V, 3) array.
        smpl_faces: SMPL faces array.
        out_size: int or (H, W) specifying output resolution.
        colors: optional list of RGB tuples in [0, 1] per person.
        view: "oblique" or "top".
    Returns:
        rendered_img: (H, W, 3) uint8 BGR image.
    """
    if hasattr(verts, "shape") and verts.ndim == 3:
        meshes = [v for v in verts]
    else:
        meshes = list(verts)

    if isinstance(out_size, int):
        height = width = int(out_size)
    else:
        height, width = out_size

    rgb, _ = render_mesh_topview(height, width, meshes, smpl_faces,
                                 colors=colors, view=view,
                                 center=center, radius=radius)
    rendered_img = rgb.astype(np.uint8)
    return rendered_img

