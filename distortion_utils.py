"""
Shared helpers for bounding-box distortion experiments.
"""

import os.path as osp

try:
    import pickle5 as pickle
except ImportError:
    import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_LEVELS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
PEDESTRIAN_NOT_PRESENT = -10
DEFAULT_MODE = "uniform_scale"
VALID_MODES = ("uniform_scale", "aspect_ratio")

"""
    uniform_scale: the width and height are multiplied by the same amount
    aspect_ratio: the width and height are changed in opposite directions so that one dimension grows while the other shrinks
"""


def format_scale(scale):
    return f"{scale:.2f}"


def format_mode(mode):
    return mode.replace("_", " ")


def load_combined_sample(base_dir, video_id, frame_id, pid):
    """
    Load a combined sample from the jaadpose dataset. Where Pedestrians actually exist, and selects a specific pedestrian if pid is provided.
    """
    combined_fp = osp.join(
        base_dir,
        "jaadpie_pose",
        "sequences",
        "jaad_all_all",
        "test",
        "combined",
        f"{video_id}.pkl",
    )
    with open(combined_fp, "rb") as f:
        data = pickle.load(f)

    if frame_id < 0 or frame_id >= len(data):
        raise ValueError(f"Frame {frame_id} is out of range for {video_id}.")

    ped_data = data[frame_id].get("ped_data", [])
    candidates = [p for p in ped_data if p.get("actions") != PEDESTRIAN_NOT_PRESENT]
    if not candidates:
        raise ValueError(f"No valid pedestrians found for {video_id} frame {frame_id}.")

    selected = None
    if pid:
        for ped in candidates:
            if ped.get("pid") == pid:
                selected = ped
                break
        if selected is None:
            raise ValueError(
                f"Pedestrian {pid} not found in {video_id} frame {frame_id}."
            )
    else:
        selected = max(
            candidates,
            key=lambda ped: (
                (ped["bbox"][2] - ped["bbox"][0]) * (ped["bbox"][3] - ped["bbox"][1])
            ),
        )

    return {
        "combined_fp": combined_fp,
        "frame_path_hint": data[frame_id].get("path", ""),
        "frame_id": frame_id,
        "video_id": video_id,
        "pid": selected.get("pid"),
        "bbox": np.asarray(selected.get("bbox"), dtype=np.float32),
    }


def load_combined_sequence(base_dir, video_id):
    """
    Load a combined sequence from the jaadpose dataset.
    """
    combined_fp = osp.join(
        base_dir,
        "jaadpie_pose",
        "sequences",
        "jaad_all_all",
        "test",
        "combined",
        f"{video_id}.pkl",
    )
    with open(combined_fp, "rb") as f:
        return pickle.load(f)


def extract_pid_frame_records(sequence_data, pid):
    """
    Create a list of frame records for a specific pedestrian from the sequence data.
    """
    records = []
    for frame_id, frame_entry in enumerate(sequence_data):
        ped_data = frame_entry.get("ped_data", [])
        for ped in ped_data:
            if ped.get("actions") == PEDESTRIAN_NOT_PRESENT or ped.get("pid") != pid:
                continue
            records.append(
                {
                    "frame_id": frame_id,
                    "bbox": np.asarray(ped.get("bbox"), dtype=np.float32),
                    "image_path_hint": frame_entry.get("path", ""),
                }
            )
            break
    return records


def select_window_records(records, center_frame, window_size):
    """
    Select a window of records centered around a given frame, for visualizations.
    """
    if window_size <= 0:
        raise ValueError("window_size must be positive.")

    center_idx = None
    for idx, record in enumerate(records):
        if record["frame_id"] == center_frame:
            center_idx = idx
            break

    if center_idx is None:
        raise ValueError(
            f"Frame {center_frame} was not found in the selected pedestrian track."
        )

    half_window = window_size // 2
    start_idx = max(0, center_idx - half_window)
    end_idx = min(len(records), start_idx + window_size)
    start_idx = max(0, end_idx - window_size)
    return records[start_idx:end_idx]


def read_clip_frame(base_dir, video_id, frame_id):
    """
    Read a frame from the JAAD clip video.
    """
    clip_fp = osp.join(base_dir, "JAAD_clips", f"{video_id}.mp4")
    cap = cv2.VideoCapture(clip_fp)
    if not cap.isOpened():
        raise ValueError(f"Unable to open clip {clip_fp}.")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise ValueError(f"Unable to read frame {frame_id} from {clip_fp}.")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def bbox_dimensions(bbox_xyxy):
    """
    Calculate the width and height of a bounding box.
    """
    x1, y1, x2, y2 = np.asarray(bbox_xyxy, dtype=np.float32)
    width = max(1.0, float(x2 - x1))
    height = max(1.0, float(y2 - y1))
    return width, height


def bbox_aspect_ratio(bbox_xyxy):
    """
    Calculate the aspect ratio of a bounding box.
    """
    width, height = bbox_dimensions(bbox_xyxy)
    return width / max(height, 1.0)


def bbox_center(bbox_xyxy):
    """
    Calculate the center of a bounding box.
    """
    x1, y1, x2, y2 = np.asarray(bbox_xyxy, dtype=np.float32)
    return (float((x1 + x2) / 2.0), float((y1 + y2) / 2.0))


def draw_bbox(ax, bbox, color, label=None, linewidth=2.5, linestyle="-", alpha=1.0):
    """
    Draws an unfilled bounding box using Matplotlib.
    """
    x1, y1, x2, y2 = bbox
    rect = plt.Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        fill=False,
        edgecolor=color,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha,
    )
    ax.add_patch(rect)
    if label:
        ax.text(
            x1,
            max(15, y1 - 10),
            label,
            color=color,
            fontsize=10,
            fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.45, pad=2, edgecolor="none"),
        )


def build_sample_dir(output_dir, mode, sample):
    return osp.join(
        output_dir,
        mode,
        f"{sample['video_id']}_frame_{sample['frame_id']:05d}_{sample['pid']}",
    )


def apply_bbox_distortion_xyxy(bbox_xyxy, mode, level, image_shape, rng):
    """
    Create a bounding box with two different distortion modes: uniform_scale and aspect_ratio.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"Unsupported distortion mode: {mode}")

    x1, y1, x2, y2 = np.asarray(bbox_xyxy, dtype=np.float32)
    width = max(1.0, float(x2 - x1))
    height = max(1.0, float(y2 - y1))
    center_x = float((x1 + x2) / 2.0)
    center_y = float((y1 + y2) / 2.0)

    sampled_delta = float(rng.normal(loc=0.0, scale=level))
    if mode == "uniform_scale":
        sampled_width_scale = max(0.05, 1.0 + sampled_delta)
        sampled_height_scale = max(0.05, 1.0 + sampled_delta)
    else:
        sampled_width_scale = float(np.exp(sampled_delta))
        sampled_height_scale = float(np.exp(-sampled_delta))

    desired_width = width * sampled_width_scale
    desired_height = height * sampled_height_scale

    image_h, image_w = image_shape[:2]
    max_width = max(1.0, 2.0 * min(center_x, image_w - center_x))
    max_height = max(1.0, 2.0 * min(center_y, image_h - center_y))
    applied_width = float(np.clip(desired_width, 1.0, max_width))
    applied_height = float(np.clip(desired_height, 1.0, max_height))

    distorted_bbox = np.array(
        [
            center_x - applied_width / 2.0,
            center_y - applied_height / 2.0,
            center_x + applied_width / 2.0,
            center_y + applied_height / 2.0,
        ],
        dtype=np.float32,
    )
    clipped = (not np.isclose(desired_width, applied_width)) or (
        not np.isclose(desired_height, applied_height)
    )

    return distorted_bbox, {
        "sampled_delta": sampled_delta,
        "sampled_scales": (float(sampled_width_scale), float(sampled_height_scale)),
        "applied_scales": (applied_width / width, applied_height / height),
        "original_center": (center_x, center_y),
        "distorted_center": (center_x, center_y),
        "original_size": (width, height),
        "distorted_size": (applied_width, applied_height),
        "original_aspect_ratio": width / height,
        "distorted_aspect_ratio": applied_width / applied_height,
        "clipped": clipped,
    }


def prepare_distortion_samples(bbox, image_shape, levels, seed, mode):
    cmap = plt.get_cmap("turbo", len(levels))
    samples = []
    for idx, level in enumerate(levels):
        rng = np.random.default_rng(seed + idx)
        distorted_bbox, distortion_info = apply_bbox_distortion_xyxy(
            bbox, mode, level, image_shape, rng
        )
        samples.append(
            {
                "level": level,
                "bbox": distorted_bbox,
                "sampled_delta": distortion_info["sampled_delta"],
                "sampled_scales": distortion_info["sampled_scales"],
                "applied_scales": distortion_info["applied_scales"],
                "center": distortion_info["distorted_center"],
                "original_size": distortion_info["original_size"],
                "distorted_size": distortion_info["distorted_size"],
                "original_aspect_ratio": distortion_info["original_aspect_ratio"],
                "distorted_aspect_ratio": distortion_info["distorted_aspect_ratio"],
                "clipped": distortion_info["clipped"],
                "color": cmap(idx),
            }
        )
    return samples
