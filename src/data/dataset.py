import json
import os
import random
from pathlib import Path
from typing import Sequence, Optional, List
from tqdm import tqdm

import einops
import mediapy
import numpy as np
import torch
import einops
from torch.utils.data import Dataset
from torchvision import transforms


def _load_encoded_video_cls():
    try:
        from pytorchvideo.data.encoded_video import EncodedVideo
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pytorchvideo is required to load MP4 datasets. Install the "
            "requirements or install pytorchvideo before constructing dataset items."
        ) from exc
    return EncodedVideo


class OpenXMP4VideoDataset(Dataset):
    def __init__(
        self,
        args,
        split: str = "train",
        max_videos: int | None = None,
    ) -> None:
        super().__init__()

        subset_names = args.subset_names

        if split not in {"train", "test"}:
            raise ValueError(f"Unknown split: {split}")

        if isinstance(subset_names, str):
            subset_names = subset_names.split(",")
        self.save_dir = Path(args.dataset_dir)
        if subset_names is None:
            subset_names = [p.name for p in self.save_dir.iterdir() if p.is_dir()]
        self.subset_names = list(subset_names)

        self.n_frames = int(args.n_frames)
        self.num_history = int(args.num_history)

        self.frame_skip = int(args.frame_skip)
        self.clip_len = self.n_frames * self.frame_skip
        self.action_dim = int(args.action_dim)
        self.variable_history_sampling = args.variable_history_sampling

        self.transform = transforms.Resize((int(args.input_h), int(args.input_w)))

        self.video_paths: list[Path] = []
        self.video_lengths: list[int] = []
        for name in self.subset_names:
            subset_dir = self.save_dir / name / split
            mp4_files = sorted(subset_dir.glob("*.mp4"))
            if max_videos is not None:
                mp4_files = mp4_files[:max_videos]
            for mp4 in tqdm(mp4_files, desc=f"Loading {name} {split} videos"):
                action_path = mp4.with_suffix(".npz")
                if not action_path.exists():
                    continue
                try:
                    npz = np.load(action_path)
                    arr = npz["actions"] if "actions" in npz else npz["arr_0"]
                    length = int(arr.shape[0])
                except Exception:
                    continue
                if length >= self.clip_len:
                    self.video_paths.append(mp4)
                    self.video_lengths.append(length)

        if not self.video_paths:
            raise RuntimeError(
                f"No valid videos found in {self.save_dir} for subsets {self.subset_names}"
            )

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        video_path = self.video_paths[idx]
        length = self.video_lengths[idx]
        action_path = video_path.with_suffix(".npz")

        # Variable history/future sampling
        n_history = self.num_history
        n_future = self.n_frames - n_history  # current + future frames

        if self.variable_history_sampling:
            # Random skip for future frames (1 or 2)
            skip_future = np.random.randint(1, 3) * self.frame_skip
            # Random skip for history frames (4x or 8x future skip)
            skip_history = skip_future * 4
            # 15% chance of no history (all history frames = current frame)
            if np.random.random() < 0.15:
                skip_history = 0
        else:
            skip_future = self.frame_skip
            skip_history = self.frame_skip

        # Calculate required span
        history_span = (
            n_history * skip_history if (n_history > 0 and skip_history > 0) else 0
        )
        future_span = (n_future - 1) * skip_future + 1

        # Pick current_step ensuring we can fit history before and future after
        max_start = length - future_span
        min_start = history_span
        if max_start < min_start:
            current_step = min_start
        else:
            current_step = np.random.randint(min_start, max_start + 1)

        # Build step indices: [history..., current, future...]
        step_indices = []
        for i in range(n_history, 0, -1):
            step_indices.append(max(0, current_step - i * skip_history))
        step_indices.append(current_step)
        for i in range(1, n_future):
            step_indices.append(min(current_step + i * skip_future, length - 1))

        # Load video clip covering the needed range
        load_start = min(step_indices)
        load_end = max(step_indices)
        EncodedVideo = _load_encoded_video_cls()
        video = EncodedVideo.from_path(video_path, decode_audio=False)
        fps = video._container.streams.video[0].guessed_rate
        start_sec = load_start / fps
        end_sec = (load_end + 1) / fps
        clip = video.get_clip(start_sec=start_sec, end_sec=end_sec)["video"]
        clip = einops.rearrange(clip, "c t h w -> t h w c")

        # Index into loaded clip using relative positions
        rel_indices = [min(si - load_start, clip.shape[0] - 1) for si in step_indices]
        clip = clip[rel_indices]

        # Load actions at the same step indices
        npz_data = np.load(action_path)
        actions_all = (
            npz_data["actions"] if "actions" in npz_data else npz_data["arr_0"]
        )
        safe_indices = [min(si, len(actions_all) - 1) for si in step_indices]
        actions = actions_all[safe_indices]
        assert (
            actions.shape[1] == self.action_dim
        ), f"Unexpected action dim: {actions.shape[1]} != {self.action_dim}"

        assert len(clip) == self.n_frames
        assert len(actions) == self.n_frames

        clip = clip.float() / 255.0
        clip = einops.rearrange(clip, "t h w c -> t c h w")
        clip = self.transform(clip)
        clip = einops.rearrange(clip, "t c h w -> t h w c")
        actions = torch.from_numpy(actions).float()
        return clip, actions


class MultiViewMP4VideoDataset(OpenXMP4VideoDataset):
    """Loads multi-view Bridge V2 episodes stored as *_view0.mp4, *_view1.mp4, etc.

    Returns ``(V, T, H, W, C)`` frames and ``(T, action_dim)`` actions so the
    training loop can reshape views into the batch dimension for encoding.
    """

    def __init__(
        self, args, split: str = "train", max_videos: int | None = None
    ) -> None:
        self.num_views: int = int(getattr(args, "num_views", 3))
        # Temporarily redirect subset_names so the parent scans *_view0.mp4 files.
        # We override the file-scanning logic below.
        Dataset.__init__(self)

        subset_names = args.subset_names
        if split not in {"train", "test"}:
            raise ValueError(f"Unknown split: {split}")
        if isinstance(subset_names, str):
            subset_names = subset_names.split(",")
        self.save_dir = Path(args.dataset_dir)
        if subset_names is None:
            subset_names = [p.name for p in self.save_dir.iterdir() if p.is_dir()]
        self.subset_names = list(subset_names)

        self.n_frames = int(args.n_frames)
        self.num_history = int(args.num_history)
        self.frame_skip = int(args.frame_skip)
        self.clip_len = self.n_frames * self.frame_skip
        self.action_dim = int(args.action_dim)
        self.variable_history_sampling = args.variable_history_sampling
        self.transform = transforms.Resize((int(args.input_h), int(args.input_w)))

        # Scan for multi-view episodes: look for *_view0.mp4, ensure all views exist
        self.video_paths: list[Path] = []  # stores the *_view0.mp4 path as the key
        self.video_lengths: list[int] = []
        for name in self.subset_names:
            subset_dir = self.save_dir / name / split
            view0_files = sorted(subset_dir.glob("*_view0.mp4"))
            if max_videos is not None:
                view0_files = view0_files[:max_videos]
            for view0_mp4 in tqdm(
                view0_files, desc=f"Loading {name} {split} multi-view"
            ):
                # Verify all views exist
                base = str(view0_mp4).replace("_view0.mp4", "")
                all_views_exist = all(
                    Path(f"{base}_view{v}.mp4").exists() for v in range(self.num_views)
                )
                if not all_views_exist:
                    continue
                action_path = Path(f"{base}.npz")
                if not action_path.exists():
                    continue
                try:
                    npz = np.load(action_path)
                    arr = npz["actions"] if "actions" in npz else npz["arr_0"]
                    length = int(arr.shape[0])
                except Exception:
                    continue
                if length >= self.clip_len:
                    self.video_paths.append(view0_mp4)
                    self.video_lengths.append(length)

        if not self.video_paths:
            raise RuntimeError(
                f"No valid multi-view videos found in {self.save_dir} for subsets {self.subset_names}"
            )

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        view0_path = self.video_paths[idx]
        length = self.video_lengths[idx]
        base = str(view0_path).replace("_view0.mp4", "")
        action_path = Path(f"{base}.npz")

        # Use the same sampling logic as the parent class
        n_history = self.num_history
        n_future = self.n_frames - n_history

        if self.variable_history_sampling:
            skip_future = np.random.randint(1, 3) * self.frame_skip
            skip_history = skip_future * 4
            if np.random.random() < 0.15:
                skip_history = 0
        else:
            skip_future = self.frame_skip
            skip_history = self.frame_skip

        history_span = (
            n_history * skip_history if (n_history > 0 and skip_history > 0) else 0
        )
        future_span = (n_future - 1) * skip_future + 1
        max_start = length - future_span
        min_start = history_span
        if max_start < min_start:
            current_step = min_start
        else:
            current_step = np.random.randint(min_start, max_start + 1)

        step_indices = []
        for i in range(n_history, 0, -1):
            step_indices.append(max(0, current_step - i * skip_history))
        step_indices.append(current_step)
        for i in range(1, n_future):
            step_indices.append(min(current_step + i * skip_future, length - 1))

        load_start = min(step_indices)
        load_end = max(step_indices)

        # Load all views with the same frame indices
        view_clips = []
        EncodedVideo = _load_encoded_video_cls()
        for v in range(self.num_views):
            vpath = Path(f"{base}_view{v}.mp4")
            video = EncodedVideo.from_path(vpath, decode_audio=False)
            fps = video._container.streams.video[0].guessed_rate
            start_sec = load_start / fps
            end_sec = (load_end + 1) / fps
            clip = video.get_clip(start_sec=start_sec, end_sec=end_sec)["video"]
            clip = einops.rearrange(clip, "c t h w -> t h w c")
            rel_indices = [
                min(si - load_start, clip.shape[0] - 1) for si in step_indices
            ]
            clip = clip[rel_indices]
            clip = clip.float() / 255.0
            clip = einops.rearrange(clip, "t h w c -> t c h w")
            clip = self.transform(clip)
            clip = einops.rearrange(clip, "t c h w -> t h w c")
            view_clips.append(clip)

        # Stack views: (V, T, H, W, C)
        frames = torch.stack(view_clips, dim=0)

        # Load actions
        npz_data = np.load(action_path)
        actions_all = (
            npz_data["actions"] if "actions" in npz_data else npz_data["arr_0"]
        )
        safe_indices = [min(si, len(actions_all) - 1) for si in step_indices]
        actions = actions_all[safe_indices]
        assert (
            actions.shape[1] == self.action_dim
        ), f"Unexpected action dim: {actions.shape[1]} != {self.action_dim}"
        actions = torch.from_numpy(actions).float()
        return frames, actions
