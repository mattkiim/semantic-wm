
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


def _tactile_enabled(args) -> bool:
    return bool(getattr(args, "use_tactile", False) or getattr(args, "tactile_dim", 0) > 0)


def _npz_array(npz_data, key: str):
    if key in npz_data:
        return npz_data[key]
    if key == "tactile" and "touch" in npz_data:
        return npz_data["touch"]
    raise KeyError(key)


def _prepare_tactile_array(tactile) -> np.ndarray:
    tactile = np.array(tactile)
    if tactile.ndim > 2:
        tactile = tactile.mean(axis=tuple(range(1, tactile.ndim - 1)))
    return tactile


class H5EmbeddingDataset(Dataset):
    """Loads pre-computed backbone patch embeddings from HDF5 (combined_v3 format).

    Returns ``(embeddings, actions)`` where embeddings are
    ``(T, patch_h, patch_w, D)`` float32 — ready to feed directly into the
    adapter / DiT without any encoder forward pass.

    Args:
        args: Namespace with h5_train_path, h5_val_path, h5_embedding_key,
              patch_h, patch_w, n_frames, num_history, frame_skip, action_dim,
              variable_history_sampling.
        split: ``"train"`` or ``"test"``/``"val"``.
    """

    def __init__(self, args, split: str = "train") -> None:
        super().__init__()

        if split == "train":
            h5_path = getattr(args, "h5_train_path", None)
        else:
            h5_path = getattr(args, "h5_val_path", None)

        if not h5_path:
            raise ValueError(f"h5_{'train' if split == 'train' else 'val'}_path must be set")

        self.h5_path = Path(h5_path)
        self.embedding_key = str(getattr(args, "h5_embedding_key", "cam_0_patch_embd"))
        self.patch_h = int(getattr(args, "patch_h", 14))
        self.patch_w = int(getattr(args, "patch_w", 14))
        self.n_frames = int(args.n_frames)
        self.num_history = int(args.num_history)
        self.frame_skip = int(args.frame_skip)
        self.clip_len = self.n_frames * self.frame_skip
        self.action_dim = int(args.action_dim)
        self.use_tactile = _tactile_enabled(args)
        self.tactile_dim = int(getattr(args, "tactile_dim", 0))
        self.tactile_key = str(getattr(args, "h5_tactile_key", "cam_tactile_patch_embd"))
        self.variable_history_sampling = args.variable_history_sampling

        self._h5_file = None

        import h5py
        with h5py.File(self.h5_path, "r") as f:
            all_keys = sorted(f.keys())
            valid_keys, valid_lengths = [], []
            for k in all_keys:
                length = int(f[k]["actions"].shape[0])
                if self.use_tactile:
                    if self.tactile_key not in f[k]:
                        continue
                    length = min(length, int(f[k][self.tactile_key].shape[0]))
                if length >= self.clip_len:
                    valid_keys.append(k)
                    valid_lengths.append(length)

        if not valid_keys:
            raise RuntimeError(
                f"No trajectories with length >= {self.clip_len} in {self.h5_path}"
            )
        self.traj_keys = valid_keys
        self.traj_lengths = valid_lengths

    def __len__(self) -> int:
        return len(self.traj_keys)

    def _get_h5(self):
        if self._h5_file is None:
            import h5py
            self._h5_file = h5py.File(self.h5_path, "r")
        return self._h5_file

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        key = self.traj_keys[idx]
        length = self.traj_lengths[idx]

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

        f = self._get_h5()
        traj = f[key]
        idx_arr = np.array(step_indices)
        unique_sorted, inverse = np.unique(idx_arr, return_inverse=True)

        # embeddings: (n_frames, N_patches, D) → reshape to (n_frames, patch_h, patch_w, D)
        emb = traj[self.embedding_key][unique_sorted.tolist()][inverse]
        actions = traj["actions"][unique_sorted.tolist()][inverse]
        tactile = (
            traj[self.tactile_key][unique_sorted.tolist()][inverse]
            if self.use_tactile
            else None
        )

        assert (
            actions.shape[1] == self.action_dim
        ), f"Unexpected action dim: {actions.shape[1]} != {self.action_dim}"

        emb = torch.from_numpy(np.array(emb)).float()
        emb = emb.reshape(self.n_frames, self.patch_h, self.patch_w, -1)

        actions = torch.from_numpy(np.array(actions)).float()
        if tactile is None:
            return emb, actions

        tactile = torch.from_numpy(_prepare_tactile_array(tactile)).float()
        if self.tactile_dim > 0:
            assert (
                tactile.shape[1] == self.tactile_dim
            ), f"Unexpected tactile dim: {tactile.shape[1]} != {self.tactile_dim}"
        return emb, actions, tactile


class H5TrajectoryDataset(Dataset):
    """Dataset backed by a consolidated HDF5 file (combined_v3 format).

    Each trajectory group must contain:
      - ``{camera_key}``: (T, H, W, C) float32 in [0, 1]
      - ``actions``:      (T, action_dim) float

    Args:
        args: Namespace with fields:
            h5_train_path, h5_val_path, n_frames, num_history,
            frame_skip, action_dim, input_h, input_w,
            variable_history_sampling, h5_camera_key (optional).
        split: ``"train"`` or ``"test"``/``"val"``.
    """

    def __init__(self, args, split: str = "train") -> None:
        super().__init__()

        if split == "train":
            h5_path = getattr(args, "h5_train_path", None)
        else:
            h5_path = getattr(args, "h5_val_path", None)

        if not h5_path:
            raise ValueError(
                f"'h5_{'train' if split == 'train' else 'val'}_path' must be set in args"
            )

        self.h5_path = Path(h5_path)
        self.n_frames = int(args.n_frames)
        self.num_history = int(args.num_history)
        self.frame_skip = int(args.frame_skip)
        self.clip_len = self.n_frames * self.frame_skip
        self.action_dim = int(args.action_dim)
        self.use_tactile = _tactile_enabled(args)
        self.tactile_dim = int(getattr(args, "tactile_dim", 0))
        self.tactile_key = str(getattr(args, "h5_tactile_key", "cam_tactile_patch_embd"))
        self.variable_history_sampling = args.variable_history_sampling
        self.camera_key = str(getattr(args, "h5_camera_key", "camera_0"))
        self.transform = transforms.Resize((int(args.input_h), int(args.input_w)))

        # Per-worker lazy file handle (h5py handles can't be pickled across workers)
        self._h5_file = None

        import h5py
        with h5py.File(self.h5_path, "r") as f:
            all_keys = sorted(f.keys())
            valid_keys, valid_lengths = [], []
            for k in all_keys:
                length = int(f[k]["actions"].shape[0])
                if self.use_tactile:
                    if self.tactile_key not in f[k]:
                        continue
                    length = min(length, int(f[k][self.tactile_key].shape[0]))
                if length >= self.clip_len:
                    valid_keys.append(k)
                    valid_lengths.append(length)

        if not valid_keys:
            raise RuntimeError(
                f"No trajectories with length >= {self.clip_len} found in {self.h5_path}"
            )

        self.traj_keys = valid_keys
        self.traj_lengths = valid_lengths

    def __len__(self) -> int:
        return len(self.traj_keys)

    def _get_h5(self):
        if self._h5_file is None:
            import h5py
            self._h5_file = h5py.File(self.h5_path, "r")
        return self._h5_file

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        key = self.traj_keys[idx]
        length = self.traj_lengths[idx]

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

        f = self._get_h5()
        traj = f[key]
        # h5py requires strictly increasing indices; use unique sorted indices then remap.
        idx_arr = np.array(step_indices)
        unique_sorted, inverse = np.unique(idx_arr, return_inverse=True)
        frames = traj[self.camera_key][unique_sorted.tolist()][inverse]
        actions = traj["actions"][unique_sorted.tolist()][inverse]
        tactile = (
            traj[self.tactile_key][unique_sorted.tolist()][inverse]
            if self.use_tactile
            else None
        )

        assert (
            actions.shape[1] == self.action_dim
        ), f"Unexpected action dim: {actions.shape[1]} != {self.action_dim}"

        frames = torch.from_numpy(frames).float()
        frames = einops.rearrange(frames, "t h w c -> t c h w")
        frames = self.transform(frames)
        frames = einops.rearrange(frames, "t c h w -> t h w c")

        actions = torch.from_numpy(np.array(actions)).float()
        if tactile is None:
            return frames, actions

        tactile = torch.from_numpy(_prepare_tactile_array(tactile)).float()
        if self.tactile_dim > 0:
            assert (
                tactile.shape[1] == self.tactile_dim
            ), f"Unexpected tactile dim: {tactile.shape[1]} != {self.tactile_dim}"
        return frames, actions, tactile


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
        self.use_tactile = _tactile_enabled(args)
        self.tactile_dim = int(getattr(args, "tactile_dim", 0))
        self.tactile_key = str(getattr(args, "tactile_npz_key", "tactile"))
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
                    if self.use_tactile:
                        tactile_arr = _npz_array(npz, self.tactile_key)
                        length = min(int(arr.shape[0]), int(tactile_arr.shape[0]))
                    else:
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
        tactile = None
        if self.use_tactile:
            tactile_all = _npz_array(npz_data, self.tactile_key)
            tactile = tactile_all[safe_indices]
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
        if tactile is None:
            return clip, actions

        tactile = torch.from_numpy(_prepare_tactile_array(tactile)).float()
        if self.tactile_dim > 0:
            assert (
                tactile.shape[1] == self.tactile_dim
            ), f"Unexpected tactile dim: {tactile.shape[1]} != {self.tactile_dim}"
        return clip, actions, tactile


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
        self.use_tactile = _tactile_enabled(args)
        self.tactile_dim = int(getattr(args, "tactile_dim", 0))
        self.tactile_key = str(getattr(args, "tactile_npz_key", "tactile"))
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
                    if self.use_tactile:
                        tactile_arr = _npz_array(npz, self.tactile_key)
                        length = min(int(arr.shape[0]), int(tactile_arr.shape[0]))
                    else:
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
        tactile = None
        if self.use_tactile:
            tactile_all = _npz_array(npz_data, self.tactile_key)
            tactile = tactile_all[safe_indices]
        assert (
            actions.shape[1] == self.action_dim
        ), f"Unexpected action dim: {actions.shape[1]} != {self.action_dim}"
        actions = torch.from_numpy(actions).float()
        if tactile is None:
            return frames, actions

        tactile = torch.from_numpy(_prepare_tactile_array(tactile)).float()
        if self.tactile_dim > 0:
            assert (
                tactile.shape[1] == self.tactile_dim
            ), f"Unexpected tactile dim: {tactile.shape[1]} != {self.tactile_dim}"
        return frames, actions, tactile
