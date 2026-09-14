import os
from pathlib import Path
import torch
import numpy as np
from PIL import Image, ImageDraw
import json

def get_map_polygons(map_path):
    path = Path(map_path)
    if not path.exists():
        # Fallback to centralized maps directory on the mounted disk
        central_path = Path("/mnt/kinexar-data/maps") / path.name
        if central_path.exists():
            path = central_path
        else:
            raise FileNotFoundError(f"map not found at {map_path} or {central_path}")

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    polys = []
    for val in raw["drivable_areas"].values():
        bnd = val.get("area_boundary", [])
        if len(bnd) < 3:
            continue
        pts = [[float(x["x"]), float(x["y"])] for x in bnd]
        polys.append(np.array(pts))
    return polys

def visualize_sample(
    scenario_id,
    map_path,
    input_traj,  # (N, T_hist, 2) - we only care about the focal agent (index 0)
    target_traj, # (T_fut, 2)
    pred_trajs,  # (K, T_fut, 2)
    output_path,
    res=0.2,     # higher resolution for visualization
    padding=40.0
):
    polys = get_map_polygons(map_path)
    
    # Focal agent input
    focal_input = input_traj[0].cpu().numpy() # (T_hist, 2)
    focal_target = target_traj.cpu().numpy() # (T_fut, 2)
    focal_preds = pred_trajs.cpu().numpy()   # (K, T_fut, 2)
    
    # Calculate bounding box for the view
    all_pts = []
    for p in polys: all_pts.append(p)
    all_pts.append(focal_input)
    all_pts.append(focal_target)
    for k in range(focal_preds.shape[0]):
        all_pts.append(focal_preds[k])
    
    pts_concat = np.concatenate(all_pts, axis=0)
    
    # Focus on the focal agent's immediate surroundings
    cx, cy = focal_input[-1]
    x0 = cx - padding
    x1 = cx + padding
    y0 = cy - padding
    y1 = cy + padding
    
    w = int(ceil_div(x1 - x0, res))
    h = int(ceil_div(y1 - y0, res))
    
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    def to_px(x, y):
        px = (x - x0) / res
        py = (y1 - y) / res
        return px, py

    # Draw Map
    for poly in polys:
        px_poly = [to_px(x, y) for x, y in poly]
        draw.polygon(px_poly, fill=(240, 240, 240), outline=(200, 200, 200))
        
    # Draw Input (Blue)
    input_px = [to_px(x, y) for x, y in focal_input]
    draw.line(input_px, fill=(0, 0, 255), width=3)
    
    # Draw Target (Green)
    target_px = [to_px(x, y) for x, y in focal_target]
    draw.line(target_px, fill=(0, 200, 0), width=4)
    
    # Draw Predictions (Red/Orange)
    colors = [(255, 0, 0), (255, 100, 0), (255, 150, 0), (255, 200, 0), (200, 100, 0), (150, 50, 0)]
    for k in range(min(focal_preds.shape[0], len(colors))):
        pred_px = [to_px(x, y) for x, y in focal_preds[k]]
        draw.line(pred_px, fill=colors[k], width=2)
        
    img.save(output_path)

def ceil_div(a, b):
    return int(-(a // -b))
