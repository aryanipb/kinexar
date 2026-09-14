import json
import logging
import math
import os
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt

from kinexar.dataset.preprocessing.graph_builder import prebuild


col = [
    "track_id",
    "timestep",
    "position_x",
    "position_y",
    "heading",
    "velocity_x",
    "velocity_y",
    "focal_track_id",
]

res = 0.3
pad = 5.0
CACHE_WAIT_SEC = 5.0
CACHE_POLL_SEC = 0.05
SAFE_FORWARD_METERS = 80.0
SAFE_BACKWARD_METERS = 40.0
SAFE_LATERAL_METERS = 40.0
SAFE_MIN_INTERSECTION_RATIO = 0.6
MAX_NEIGHBORS = 9

logger = logging.getLogger(__name__)


def sid(path):
    return Path(path).stem.replace("scenario_", "", 1)


def mappath(path):
    p = Path(path)
    return p.with_name(f"log_map_archive_{sid(path)}.json")


def polysjson(map_path):
    with open(map_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out = []
    for val in raw["drivable_areas"].values():
        bnd = val.get("area_boundary", [])
        if len(bnd) < 3:
            continue
        pts = [[float(x["x"]), float(x["y"])] for x in bnd]
        out.append(np.asarray(pts, dtype=np.float16))
    if not out:
        raise ValueError("no drivable polygons")
    return out


def _bbox_area(bbox):
    x0, y0, x1, y1 = bbox
    return max(0.0, float(x1) - float(x0)) * max(0.0, float(y1) - float(y0))


def _bbox_intersection(a, b):
    x0 = max(float(a[0]), float(b[0]))
    y0 = max(float(a[1]), float(b[1]))
    x1 = min(float(a[2]), float(b[2]))
    y1 = min(float(a[3]), float(b[3]))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _drivable_bbox(polys):
    pts = np.concatenate(polys, axis=0)
    return (
        float(pts[:, 0].min() - pad),
        float(pts[:, 1].min() - pad),
        float(pts[:, 0].max() + pad),
        float(pts[:, 1].max() + pad),
    )


def _window_bbox(cx, cy, heading):
    dx = math.cos(float(heading))
    dy = math.sin(float(heading))
    right_x = -dy
    right_y = dx
    half_length = 0.5 * (SAFE_FORWARD_METERS + SAFE_BACKWARD_METERS)
    half_width = SAFE_LATERAL_METERS
    center_x = float(cx) + dx * 0.5 * (SAFE_FORWARD_METERS - SAFE_BACKWARD_METERS)
    center_y = float(cy) + dy * 0.5 * (SAFE_FORWARD_METERS - SAFE_BACKWARD_METERS)
    corners = []
    for s_long in (-1.0, 1.0):
        for s_lat in (-1.0, 1.0):
            x = center_x + s_long * half_length * dx + s_lat * half_width * right_x
            y = center_y + s_long * half_length * dy + s_lat * half_width * right_y
            corners.append((x, y))
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    return (min(xs), min(ys), max(xs), max(ys))


def _pick_agent_pose(cur, hist_steps):
    if not cur:
        return None
    target = hist_steps - 1
    if target in cur:
        state = cur[target]
    else:
        prior = [t for t in cur if t <= target]
        if prior:
            state = cur[max(prior)]
        else:
            state = cur[min(cur)]
    return float(state[0]), float(state[1]), float(state[2])


def _safe_crop_bbox(polys, cx, cy, heading):
    drivable_bbox = _drivable_bbox(polys)
    window_bbox = _window_bbox(cx, cy, heading)
    final_bbox = _bbox_intersection(drivable_bbox, window_bbox)
    window_area = _bbox_area(window_bbox)
    final_area = _bbox_area(final_bbox) if final_bbox is not None else 0.0
    if final_bbox is None or final_area < SAFE_MIN_INTERSECTION_RATIO * window_area:
        final_bbox = window_bbox
    if _bbox_area(final_bbox) <= 0.0:
        raise ValueError(f"invalid zero-area crop: {final_bbox}")
    return drivable_bbox, window_bbox, final_bbox


def _log_crop(debug_name, map_path, drivable_bbox, final_bbox):
    before_w = float(drivable_bbox[2] - drivable_bbox[0])
    before_h = float(drivable_bbox[3] - drivable_bbox[1])
    after_w = float(final_bbox[2] - final_bbox[0])
    after_h = float(final_bbox[3] - final_bbox[1])
    before_area = _bbox_area(drivable_bbox)
    after_area = _bbox_area(final_bbox)
    reduction = 100.0 * (1.0 - (after_area / before_area)) if before_area > 0.0 else 0.0
    logger.debug(
        "map crop sample=%s map=%s before_wh=(%.2f,%.2f)m after_wh=(%.2f,%.2f)m reduction=%.2f%% before_bbox=(%.2f,%.2f,%.2f,%.2f) after_bbox=(%.2f,%.2f,%.2f,%.2f)",
        debug_name,
        map_path,
        before_w,
        before_h,
        after_w,
        after_h,
        reduction,
        drivable_bbox[0],
        drivable_bbox[1],
        drivable_bbox[2],
        drivable_bbox[3],
        final_bbox[0],
        final_bbox[1],
        final_bbox[2],
        final_bbox[3],
    )


def _rasterize_polys(polys, bbox, map_path="", debug_name=""):
    x0, y0, x1, y1 = [float(v) for v in bbox]
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"invalid crop bbox for {debug_name or map_path}: {(x0, y0, x1, y1)}")
    w = int(math.ceil((x1 - x0) / res)) + 1
    h = int(math.ceil((y1 - y0) / res)) + 1
    if w <= 0 or h <= 0:
        raise ValueError(f"non-positive raster shape for {debug_name or map_path}: {(h, w)}")
    img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)
    for poly in polys:
        xy = []
        for x, y in poly:
            px = (float(x) - x0) / res
            py = (y1 - float(y)) / res
            xy.append((px, py))
        draw.polygon(xy, fill=1)
    msk = np.array(img, dtype=np.uint8) > 0
    dist = distance_transform_edt(~msk, sampling=res).astype(np.float16)
    dist = np.flipud(dist).copy()
    return {
        "distance_map": torch.from_numpy(dist),
        "map_origin": torch.tensor([x0, y0], dtype=torch.float16),
        "map_resolution": torch.tensor(res, dtype=torch.float16),
    }


def _build_cropped_map(polys, map_path, crop_bbox=None, debug_name=""):
    drivable_bbox = _drivable_bbox(polys)
    final_bbox = drivable_bbox if crop_bbox is None else tuple(float(v) for v in crop_bbox)
    if _bbox_area(final_bbox) <= 0.0:
        raise ValueError(f"invalid zero-area crop for {debug_name or map_path}: {final_bbox}")
    _log_crop(debug_name or "-", map_path, drivable_bbox, final_bbox)
    return _rasterize_polys(polys, final_bbox, map_path=map_path, debug_name=debug_name)


def buildmapjson(map_path, crop_bbox=None, debug_name=""):
    return _build_cropped_map(polysjson(map_path), map_path, crop_bbox=crop_bbox, debug_name=debug_name)


def buildmap(path):
    return buildmapjson(mappath(path))


def validmap(dat):
    if not isinstance(dat, dict):
        return False
    if set(["distance_map", "map_origin", "map_resolution"]) - set(dat):
        return False
    if dat["distance_map"].ndim != 2:
        return False
    if tuple(dat["map_origin"].shape) != (2,):
        return False
    if dat["map_resolution"].numel() != 1:
        return False
    return True


def normmap(dat):
    if not validmap(dat):
        raise ValueError("bad map payload")
    out = {
        "distance_map": dat["distance_map"].to(torch.float16),
        "map_origin": dat["map_origin"].to(torch.float16),
        "map_resolution": dat["map_resolution"].to(torch.float16),
    }
    if out["distance_map"].ndim != 2:
        raise ValueError("bad distance_map shape")
    if tuple(out["map_origin"].shape) != (2,):
        raise ValueError("bad map_origin shape")
    if tuple(out["map_resolution"].shape) != ():
        raise ValueError("bad map_resolution shape")
    if out["distance_map"].dtype != torch.float16:
        raise ValueError("bad distance_map dtype")
    if out["map_origin"].dtype != torch.float16:
        raise ValueError("bad map_origin dtype")
    if out["map_resolution"].dtype != torch.float16:
        raise ValueError("bad map_resolution dtype")
    return out


def maproot(map_path):
    root = os.environ.get("KINEXAR_MAP_DIR", "").strip() or os.environ.get("ROADWEAVER_MAP_DIR", "").strip()
    if root:
        return Path(root)
    return Path(map_path).resolve().parent


def cachepaths(scenario_id, map_path):
    root = maproot(map_path)
    cache_dir = root / "cache"
    cache_file = cache_dir / f"{scenario_id}.pt"
    tmp_file = cache_dir / f"{scenario_id}.pt.tmp"
    return cache_dir, cache_file, tmp_file


def row_cachepaths(scenario_id, start_t, map_path):
    root = maproot(map_path)
    cache_dir = root / "cache_rows"
    stem = f"{scenario_id}__{int(start_t):06d}"
    cache_file = cache_dir / f"{stem}.pt"
    tmp_file = cache_dir / f"{stem}.pt.tmp"
    return cache_dir, cache_file, tmp_file


def requirecache():
    raw = os.environ.get("KINEXAR_REQUIRE_MAP_CACHE", "").strip()
    if not raw:
        raw = os.environ.get("ROADWEAVER_REQUIRE_MAP_CACHE", "").strip()
    return raw == "1"


def loadcache(path):
    try:
        return normmap(torch.load(path, map_location="cpu"))
    except Exception:
        path.unlink(missing_ok=True)
        raise


def waitcache(path, tmp_path):
    end = time.time() + CACHE_WAIT_SEC
    while time.time() < end:
        if path.exists():
            try:
                return loadcache(path)
            except Exception:
                continue
        if not tmp_path.exists():
            return None
        time.sleep(CACHE_POLL_SEC)
    return None


def loadmapjsoncached(scenario_id, map_path):
    cache_dir, cache_file, tmp_file = cachepaths(scenario_id, map_path)
    if cache_file.exists():
        try:
            return loadcache(cache_file)
        except Exception:
            if requirecache():
                raise RuntimeError(f"corrupted map cache for scenario_id={scenario_id}: {cache_file}")
            pass
    if requirecache():
        raise FileNotFoundError(f"missing map cache for scenario_id={scenario_id}: {cache_file}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(tmp_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        dat = waitcache(cache_file, tmp_file)
        if dat is not None:
            return dat
        if cache_file.exists():
            return loadcache(cache_file)
        return normmap(buildmapjson(map_path))
    try:
        dat = normmap(buildmapjson(map_path))
        with os.fdopen(fd, "wb") as f:
            torch.save(dat, f)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.replace(tmp_file, cache_file)
        except FileNotFoundError:
            pass
        except OSError:
            if cache_file.exists():
                return loadcache(cache_file)
            raise
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        tmp_file.unlink(missing_ok=True)
        raise
    return loadcache(cache_file) if cache_file.exists() else dat


def loadmap(path):
    return loadmapjsoncached(sid(path), mappath(path))


def loadmaprowcached(row, cur=None):
    scenario_id = row.get("scenario_id", sid(row["file_path"]))
    map_path = mappath(row["file_path"])
    cache_dir, cache_file, tmp_file = row_cachepaths(scenario_id, int(row["start_t"]), str(map_path))
    if cache_file.exists():
        try:
            return loadcache(cache_file)
        except Exception:
            if requirecache():
                raise RuntimeError(f"corrupted row map cache for scenario_id={scenario_id} start_t={row['start_t']}: {cache_file}")
            pass
    if requirecache():
        raise FileNotFoundError(f"missing row map cache for scenario_id={scenario_id} start_t={row['start_t']}: {cache_file}")
    if cur is None:
        tab = pq.read_table(row["file_path"], columns=col)
        lst = tab.to_pylist()
        start = int(row["start_t"])
        stop = start + int(row.get("window_steps", int(row.get("history_steps", 50)) + int(row.get("future_steps", 40))))
        root = lst[0]["focal_track_id"]
        raw = {}
        for item in lst:
            t = item["timestep"]
            if item["track_id"] != root or t < start or t >= stop:
                continue
            raw[t - start] = [
                item["position_x"],
                item["position_y"],
                item["velocity_x"],
                item["velocity_y"],
            ]
        cur = ang(raw)
    pose = _pick_agent_pose(cur, int(row.get("history_steps", 50)))
    if pose is None:
        return loadmapjsoncached(scenario_id, str(map_path))
    polys = polysjson(map_path)
    _, _, final_bbox = _safe_crop_bbox(polys, pose[0], pose[1], pose[2])
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(tmp_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        dat = waitcache(cache_file, tmp_file)
        if dat is not None:
            return dat
        if cache_file.exists():
            return loadcache(cache_file)
        return normmap(_build_cropped_map(polys, str(map_path), crop_bbox=final_bbox, debug_name=f"{scenario_id}:{row['start_t']}"))
    try:
        dat = normmap(_build_cropped_map(polys, str(map_path), crop_bbox=final_bbox, debug_name=f"{scenario_id}:{row['start_t']}"))
        with os.fdopen(fd, "wb") as f:
            torch.save(dat, f)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.replace(tmp_file, cache_file)
        except FileNotFoundError:
            pass
        except OSError:
            if cache_file.exists():
                return loadcache(cache_file)
            raise
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        tmp_file.unlink(missing_ok=True)
        raise
    return loadcache(cache_file) if cache_file.exists() else dat


def ang(cur, eps=1e-3):
    out = {}
    last = 0.0
    for t in sorted(cur):
        x = cur[t]
        vx = x[2]
        vy = x[3]
        v = (vx * vx + vy * vy) ** 0.5
        if v >= eps:
            last = torch.atan2(torch.tensor(vy), torch.tensor(vx)).item()
        out[t] = [x[0], x[1], last, vx, vy]
    return out


def _history_xy(cur, hist_steps):
    prior = [t for t in cur if t < hist_steps]
    if not prior:
        return None
    state = cur[max(prior)]
    return float(state[0]), float(state[1])


def select_nearest_agent_ids(dat, root, hist_steps, max_neighbors=MAX_NEIGHBORS):
    """Keep focal plus nearest observed neighbors at the history horizon."""
    root_xy = _history_xy(dat.get(root, {}), hist_steps)
    others = []
    for track_id, cur in dat.items():
        if track_id == root:
            continue
        xy = _history_xy(cur, hist_steps)
        if xy is None:
            continue
        if root_xy is None:
            dist2 = 0.0
        else:
            dx = xy[0] - root_xy[0]
            dy = xy[1] - root_xy[1]
            dist2 = dx * dx + dy * dy
        others.append((dist2, str(track_id), track_id))
    others.sort(key=lambda item: (item[0], item[1]))
    return [root] + [track_id for _, _, track_id in others[: int(max_neighbors)]]


def extract_map_feat(map_path):
    with open(map_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    type_map = {"VEHICLE": 1, "BIKE": 2, "BUS": 3, "PEDESTRIAN": 4}
    all_segments = []
    for lane in data.get("lane_segments", {}).values():
        centerline = lane.get("centerline", [])
        ltype = type_map.get(lane.get("lane_type", "VEHICLE"), 0)
        is_int = 1.0 if lane.get("is_intersection", False) else 0.0
        for i in range(len(centerline)):
            pt = centerline[i]
            x, y = float(pt["x"]), float(pt["y"])
            if i < len(centerline) - 1:
                next_pt = centerline[i+1]
                dx, dy = float(next_pt["x"]) - x, float(next_pt["y"]) - y
            elif i > 0:
                prev_pt = centerline[i-1]
                dx, dy = x - float(prev_pt["x"]), y - float(prev_pt["y"])
            else:
                dx, dy = 1.0, 0.0
            mag = (dx*dx + dy*dy)**0.5
            cos_th, sin_th = (dx / mag, dy / mag) if mag > 1e-6 else (1.0, 0.0)
            all_segments.append([x, y, cos_th, sin_th, float(ltype), is_int])
    if not all_segments:
        return torch.zeros((1, 6), dtype=torch.float16)
    return torch.tensor(all_segments, dtype=torch.float16)


def load(row):
    if os.environ.get("KINEXAR_NO_PARQUET") == "1" or os.environ.get("ROADWEAVER_NO_PARQUET") == "1":
        raise Exception("Parquet access not allowed in training")
    tab = pq.read_table(row["file_path"], columns=col)
    lst = tab.to_pylist()
    hist_steps = int(row.get("history_steps", 50))
    future_steps = int(row.get("future_steps", 40))
    window_steps = int(row.get("window_steps", hist_steps + future_steps))
    start = int(row["start_t"])
    stop = start + window_steps
    root = lst[0]["focal_track_id"]
    dat = {}
    for x in lst:
        t = x["timestep"]
        if t < start or t >= stop:
            continue
        k = x["track_id"]
        if k not in dat:
            dat[k] = {}
        dat[k][t - start] = [
            x["position_x"],
            x["position_y"],
            x["velocity_x"],
            x["velocity_y"],
        ]
    for k in list(dat):
        dat[k] = ang(dat[k])
    ids = select_nearest_agent_ids(dat, root, hist_steps)
    n = len(ids)
    inp = torch.zeros((n, hist_steps, 5), dtype=torch.float16)
    out = torch.zeros((future_steps, 2), dtype=torch.float16)
    msk = torch.zeros((n, hist_steps), dtype=torch.bool)
    for i, k in enumerate(ids):
        cur = dat.get(k, {})
        for t in range(hist_steps):
            if t in cur:
                inp[i, t] = torch.tensor(cur[t], dtype=torch.float16)
                msk[i, t] = True
    cur = dat.get(root, {})
    for t in range(future_steps):
        j = t + hist_steps
        if j in cur:
            out[t] = torch.tensor(cur[j][:2], dtype=torch.float16)
    return {
        "feat": inp,
        "future": out,
        "input": inp,
        "output": out,
        "mask": msk,
        "graph": prebuild(inp, msk, topk=9, lag=1, span=1),
        "scenario_id": row["scenario_id"],
        "file_path": row["file_path"],
    }
