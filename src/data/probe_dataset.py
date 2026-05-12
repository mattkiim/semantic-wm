"""Episode-level dataset for trajectory success probing.

Loads full episodes, subsamples frames according to a chosen strategy,
and returns the success label alongside frames and actions.
"""

from pathlib import Path
from typing import List, Tuple

import einops
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm


def _load_encoded_video_cls():
    try:
        from pytorchvideo.data.encoded_video import EncodedVideo
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pytorchvideo is required to load MP4 probe datasets. Install the "
            "requirements or install pytorchvideo before constructing dataset items."
        ) from exc
    return EncodedVideo


class TrajectoryProbeDataset(Dataset):
    """Load full episodes, subsample frames, return success label.

    Parameters
    ----------
    dataset_dir : str or Path
        Root directory containing subset folders.
    subset_names : str or list[str]
        Comma-separated or list of dataset subset names (e.g. "soar").
    split : str
        "train" or "test".
    n_sample_frames : int
        Number of frames to sample per episode.
    sampling_strategy : str
        "uniform" or "final_heavy".
    input_h, input_w : int
        Target frame resolution.
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        subset_names: str | List[str],
        split: str = "train",
        n_sample_frames: int = 8,
        sampling_strategy: str = "uniform",
        input_h: int = 256,
        input_w: int = 256,
    ) -> None:
        super().__init__()

        if isinstance(subset_names, str):
            subset_names = subset_names.split(",")

        self.save_dir = Path(dataset_dir)
        self.n_sample_frames = n_sample_frames
        self.sampling_strategy = sampling_strategy
        self.transform = transforms.Resize((input_h, input_w))

        self.video_paths: List[Path] = []
        self.success_labels: List[int] = []
        self.video_lengths: List[int] = []
        self.language_instructions: List[str] = []

        for name in subset_names:
            subset_dir = self.save_dir / name / split
            if not subset_dir.exists():
                continue
            mp4_files = sorted(subset_dir.glob("*.mp4"))
            for mp4 in tqdm(mp4_files, desc=f"Loading {name}/{split} for probe"):
                npz_path = mp4.with_suffix(".npz")
                if not npz_path.exists():
                    continue
                try:
                    npz = np.load(npz_path, allow_pickle=True)
                    arr = npz["actions"] if "actions" in npz else npz["arr_0"]
                    length = int(arr.shape[0])
                except Exception:
                    continue

                # Require success label
                if "success" not in npz:
                    continue
                # Need enough frames to sample
                if length < n_sample_frames:
                    continue

                self.video_paths.append(mp4)
                self.video_lengths.append(length)
                self.success_labels.append(int(npz["success"]))
                lang = str(npz["language_instruction"]) if "language_instruction" in npz else ""
                self.language_instructions.append(lang)

        if not self.video_paths:
            raise RuntimeError(
                f"No valid episodes with success labels found in {self.save_dir} "
                f"for subsets {subset_names}, split={split}"
            )

        n_success = sum(self.success_labels)
        n_failure = len(self.success_labels) - n_success
        print(f"ProbeDataset: {len(self)} episodes ({n_success} success, {n_failure} failure)")

    def __len__(self) -> int:
        return len(self.video_paths)

    def _sample_frame_indices(self, episode_length: int) -> List[int]:
        """Return frame indices based on sampling strategy."""
        n = self.n_sample_frames
        L = episode_length

        if self.sampling_strategy == "uniform":
            return np.linspace(0, L - 1, n).astype(int).tolist()

        elif self.sampling_strategy == "fps_1":
            # Sample 1 frame per second (videos are at ~20fps)
            fps = getattr(self, "_video_fps", 20)
            step = fps  # 1 frame per second
            indices = list(range(0, L, step))
            if len(indices) > n:
                sel = np.linspace(0, len(indices) - 1, n).astype(int)
                indices = [indices[i] for i in sel]
            elif len(indices) < n:
                indices = np.linspace(0, L - 1, n).astype(int).tolist()
            return indices

        elif self.sampling_strategy == "bookend":
            # More frames from start and end, fewer from middle
            # 3 early (0-20%) + 2 middle (20-80%) + 3 late (80-100%)
            n_early = n // 3
            n_late = n // 3
            n_mid = n - n_early - n_late
            early = np.linspace(0, L * 0.2, n_early, endpoint=False).astype(int)
            mid = np.linspace(L * 0.2, L * 0.8, n_mid + 2)[1:-1].astype(int)
            late = np.linspace(L * 0.8, L - 1, n_late).astype(int)
            indices = np.concatenate([early, mid, late]).tolist()
            return indices

        elif self.sampling_strategy == "final_heavy":
            # 2 early (0-25%) + 2 middle (25-50%) + 4 late (50-100%)
            n_early, n_mid, n_late = 2, 2, n - 4
            early = np.linspace(0, L * 0.25, n_early, endpoint=False).astype(int)
            mid = np.linspace(L * 0.25, L * 0.5, n_mid, endpoint=False).astype(int)
            late = np.linspace(L * 0.5, L - 1, n_late).astype(int)
            indices = np.concatenate([early, mid, late]).tolist()
            return indices

        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """Return (frames, actions, success, episode_length).

        frames:  (n_sample_frames, H, W, 3) float [0, 1]
        actions: (n_sample_frames, action_dim) float
        success: int (0 or 1)
        episode_length: int
        """
        video_path = self.video_paths[idx]
        length = self.video_lengths[idx]
        action_path = video_path.with_suffix(".npz")
        success = self.success_labels[idx]

        # Determine which frames to sample
        frame_indices = self._sample_frame_indices(length)

        # Load video
        EncodedVideo = _load_encoded_video_cls()
        video = EncodedVideo.from_path(video_path, decode_audio=False)
        fps = video._container.streams.video[0].guessed_rate

        load_start = min(frame_indices)
        load_end = max(frame_indices)
        start_sec = load_start / fps
        end_sec = (load_end + 1) / fps
        clip = video.get_clip(start_sec=start_sec, end_sec=end_sec)["video"]
        clip = einops.rearrange(clip, "c t h w -> t h w c")

        # Index into loaded clip
        rel_indices = [min(fi - load_start, clip.shape[0] - 1) for fi in frame_indices]
        clip = clip[rel_indices]

        # Load actions
        npz = np.load(action_path)
        actions_all = npz["actions"] if "actions" in npz else npz["arr_0"]
        safe_indices = [min(fi, len(actions_all) - 1) for fi in frame_indices]
        actions = actions_all[safe_indices]

        # Resize and normalize
        clip = clip.float() / 255.0
        clip = einops.rearrange(clip, "t h w c -> t c h w")
        clip = self.transform(clip)
        clip = einops.rearrange(clip, "t c h w -> t h w c")

        actions = torch.from_numpy(actions).float()

        return clip, actions, success, length
