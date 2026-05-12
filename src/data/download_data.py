"""
Derived from https://github.com/google-deepmind/open_x_embodiment/blob/main/colabs/Open_X_Embodiment_Datasets.ipynb
"""

from typing import Any, Callable, Dict, Tuple
from pathlib import Path
from tqdm import tqdm

import numpy as np
import tensorflow_datasets as tfds
import fire


import tensorflow as tf
import functools

import torch
from torchvision.io import write_video

BRIDGE_V2_PATH = None  # TODO: add bridge_v2 dataset path when available
SOAR_PATH = None  # TODO: add SOAR dataset path when available


def map_observation(
    to_step: Dict[str, Any],
    from_step: Dict[str, Any],
    from_image_feature_names: Tuple[str, ...] = ("image",),
    to_image_feature_names: Tuple[str, ...] = ("image",),
) -> None:
    for from_feature_name, to_feature_name in zip(
        from_image_feature_names, to_image_feature_names
    ):
        to_step["observation"][to_feature_name] = from_step["observation"][
            from_feature_name
        ]


def terminate_bool_to_act(terminate_episode: np.ndarray) -> np.ndarray:
    if terminate_episode == 1.0:
        return np.array([1, 0, 0], dtype=np.int32)
    else:
        return np.array([0, 1, 0], dtype=np.int32)


def rescale_action_with_bound(
    actions: np.ndarray,
    low: float,
    high: float,
    safety_margin: float = 0,
    post_scaling_max: float = 1.0,
    post_scaling_min: float = -1.0,
) -> np.ndarray:
    resc_actions = (actions - low) / (high - low) * (
        post_scaling_max - post_scaling_min
    ) + post_scaling_min
    return tf.clip_by_value(
        resc_actions,
        post_scaling_min + safety_margin,
        post_scaling_max - safety_margin,
    )


def _rescale_action(
    action: Dict[str, np.ndarray],
    wv_lo: float = -0.05,
    wv_hi: float = 0.05,
    rd_lo: float = -0.25,
    rd_hi: float = 0.25,
) -> Dict[str, np.ndarray]:
    # Rescale world vector
    action["world_vector"] = rescale_action_with_bound(
        action["world_vector"],
        low=wv_lo,
        high=wv_hi,
        safety_margin=0.01,
        post_scaling_max=1.75,
        post_scaling_min=-1.75,
    )

    # Rescale rotation delta
    action["rotation_delta"] = rescale_action_with_bound(
        action["rotation_delta"],
        low=rd_lo,
        high=rd_hi,
        safety_margin=0.01,
        post_scaling_max=1.4,
        post_scaling_min=-1.4,
    )

    return action


# Action mapping functions for different datasets


def rt_1_map_action(to_step: Dict[str, Any], from_step: Dict[str, Any]) -> None:
    to_step["action"] = from_step["action"]


def bridge_map_action(
    to_step: Dict[str, Any],
    from_step: Dict[str, Any],
) -> None:
    to_step["action"]["world_vector"] = from_step["action"]["world_vector"]
    to_step["action"]["terminate_episode"] = terminate_bool_to_act(
        from_step["action"]["terminate_episode"]
    )
    to_step["action"]["rotation_delta"] = from_step["action"]["rotation_delta"]

    open_gripper = from_step["action"]["open_gripper"]
    possible_values = tf.constant([True, False], dtype=tf.bool)
    eq = tf.equal(possible_values, open_gripper)
    assert_op = tf.Assert(tf.reduce_any(eq), [open_gripper])

    with tf.control_dependencies([assert_op]):
        to_step["action"]["gripper_closedness_action"] = tf.cond(
            open_gripper,
            lambda: tf.constant([-1.0], dtype=tf.float32),  # Open gripper
            lambda: tf.constant([1.0], dtype=tf.float32),  # Close gripper
        )

    to_step["action"] = _rescale_action(to_step["action"])


def libero_map_action(
    to_step: Dict[str, Any],
    from_step: Dict[str, Any],
) -> None:
    to_step["action"]["terminate_episode"] = terminate_bool_to_act(
        from_step["is_terminal"]
    )
    to_step["action"]["world_vector"] = from_step["action"][0:3]
    to_step["action"]["rotation_delta"] = from_step["action"][3:6]
    to_step["action"]["gripper_closedness_action"] = from_step["action"][6:7]

    to_step["action"] = _rescale_action(
        to_step["action"], wv_lo=-1.0, wv_hi=+1.0, rd_lo=-0.4, rd_hi=+0.4
    )


def bridge_v2_map_action(
    to_step: Dict[str, Any],
    from_step: Dict[str, Any],
) -> None:
    to_step["action"]["terminate_episode"] = terminate_bool_to_act(
        from_step["is_terminal"]
    )
    to_step["action"]["world_vector"] = from_step["action"][0:3]
    to_step["action"]["rotation_delta"] = from_step["action"][3:6]

    open_gripper = from_step["action"][6:7]
    open_gripper = tf.round(open_gripper)
    open_gripper = -(open_gripper * 2 - 1)
    to_step["action"]["gripper_closedness_action"] = open_gripper

    to_step["action"] = _rescale_action(to_step["action"])


bridge_v2_map_observation = functools.partial(
    map_observation,
    from_image_feature_names=("image_0",),
    to_image_feature_names=("image",),
)

bridge_v2_map_observation_multiview = functools.partial(
    map_observation,
    from_image_feature_names=("image_0", "image_1", "image_2"),
    to_image_feature_names=("image_0", "image_1", "image_2"),
)


def taco_play_rescale_actions_by_bounds(
    actions: np.ndarray,
    lows: np.ndarray,
    highs: np.ndarray,
    safety_margin: float = 0.01,
) -> np.ndarray:
    resc_actions = (actions - lows) / (highs - lows) * 2 - 1
    return tf.clip_by_value(resc_actions, -1 + safety_margin, 1 - safety_margin)


def taco_play_rescale_action(action: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    rd_lows = tf.constant([-3.2, -0.8, -1.8])
    rd_highs = tf.constant([3.2, 0.2, 2.5])
    action["rotation_delta"] = taco_play_rescale_actions_by_bounds(
        action["rotation_delta"], lows=rd_lows, highs=rd_highs
    )

    wv_lows = tf.constant([0.0, -0.5, 0.0])
    wv_highs = tf.constant([0.8, 0.7, 0.6])
    action["world_vector"] = taco_play_rescale_actions_by_bounds(
        action["world_vector"], lows=wv_lows, highs=wv_highs
    )

    return action


def taco_play_map_action(to_step: Dict[str, Any], from_step: Dict[str, Any]) -> None:
    actions = from_step["action"]["actions"]
    to_step["action"]["world_vector"] = actions[:3]
    to_step["action"]["terminate_episode"] = terminate_bool_to_act(
        from_step["action"]["terminate_episode"]
    )
    to_step["action"]["rotation_delta"] = actions[3:6]
    to_step["action"]["gripper_closedness_action"] = tf.expand_dims(actions[6], axis=-1)

    to_step["action"] = _rescale_action(to_step["action"])


taco_play_map_observation = functools.partial(
    map_observation,
    from_image_feature_names=("rgb_static",),
    to_image_feature_names=("image",),
)


def _normalize(value: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (value - mean) / std


def jaco_play_map_action(to_step: Dict[str, Any], from_step: Dict[str, Any]) -> None:
    to_step["action"]["world_vector"] = _normalize(
        from_step["action"]["world_vector"],
        mean=tf.constant([0.00096585, -0.00580069, -0.00395066], dtype=tf.float32),
        std=tf.constant([0.12234575, 0.09676983, 0.11155209], dtype=tf.float32),
    )
    to_step["action"]["gripper_closedness_action"] = from_step["action"][
        "gripper_closedness_action"
    ]
    to_step["action"]["terminate_episode"] = from_step["action"]["terminate_episode"]


def berkeley_cable_routing_map_action(
    to_step: Dict[str, Any], from_step: Dict[str, Any]
) -> None:
    to_step["action"]["world_vector"] = from_step["action"]["world_vector"]
    to_step["action"]["rotation_delta"] = from_step["action"]["rotation_delta"]
    to_step["action"]["terminate_episode"] = terminate_bool_to_act(
        from_step["action"]["terminate_episode"]
    )


def roboturk_map_action(to_step: Dict[str, Any], from_step: Dict[str, Any]) -> None:
    """Map RoboTurk actions to standardized format."""
    to_step["action"]["world_vector"] = from_step["action"]["world_vector"]
    to_step["action"]["gripper_closedness_action"] = from_step["action"][
        "gripper_closedness_action"
    ]
    to_step["action"]["rotation_delta"] = from_step["action"]["rotation_delta"]
    to_step["action"]["terminate_episode"] = terminate_bool_to_act(
        from_step["action"]["terminate_episode"]
    )


roboturk_map_observation = functools.partial(
    map_observation,
    from_image_feature_names=("front_rgb",),
    to_image_feature_names=("image",),
)


def nyu_door_opening_surprising_effectiveness_map_action(
    to_step: Dict[str, Any], from_step: Dict[str, Any]
) -> None:
    # Scale world vector from [-0.07, 0.07] to span [-2.0, 2.0]
    to_step["action"]["world_vector"] = from_step["action"]["world_vector"] * 20.0

    # Scale rotation delta from [-0.07, 0.07] to span [-pi/2, pi/2]
    to_step["action"]["rotation_delta"] = from_step["action"]["rotation_delta"] * 15.0

    to_step["action"]["gripper_closedness_action"] = from_step["action"][
        "gripper_closedness_action"
    ]
    to_step["action"]["terminate_episode"] = terminate_bool_to_act(
        from_step["action"]["terminate_episode"]
    )


def viola_map_action(to_step: Dict[str, Any], from_step: Dict[str, Any]) -> None:
    # Scale world vector from [-1.0, 1.0] to better span [-2.0, 2.0]
    to_step["action"]["world_vector"] = from_step["action"]["world_vector"] * 1.75
    to_step["action"]["terminate_episode"] = terminate_bool_to_act(
        from_step["action"]["terminate_episode"]
    )

    # Scale rotation delta from [-0.4, 0.4] to span [-pi/2, pi/2]
    to_step["action"]["rotation_delta"] = from_step["action"]["rotation_delta"] * 3.0

    gripper_closedness_action = from_step["action"]["gripper_closedness_action"]

    # Validate gripper action values
    possible_values = np.array([-1.0, 1.0, 0.0], dtype=np.float32)
    eq = possible_values == gripper_closedness_action
    assert eq.any(), f"Invalid gripper action: {gripper_closedness_action}"

    gripper_closedness_action = np.expand_dims(gripper_closedness_action, axis=-1)
    to_step["action"]["gripper_closedness_action"] = gripper_closedness_action


viola_map_observation = functools.partial(
    map_observation,
    from_image_feature_names=("agentview_rgb",),
    to_image_feature_names=("image",),
)


def berkeley_autolab_ur5_map_action(
    to_step: Dict[str, Any], from_step: Dict[str, Any]
) -> None:
    # Scale world vector from [-0.02, 0.02] to span [-2.0, 2.0]
    to_step["action"]["world_vector"] = from_step["action"]["world_vector"] * 100.0
    to_step["action"]["terminate_episode"] = terminate_bool_to_act(
        from_step["action"]["terminate_episode"]
    )

    # Scale rotation delta from [-0.07, 0.07] to span [-pi/2, pi/2]
    to_step["action"]["rotation_delta"] = from_step["action"]["rotation_delta"] * 15.0
    to_step["action"]["gripper_closedness_action"] = tf.expand_dims(
        from_step["action"]["gripper_closedness_action"], axis=0
    )


def toto_map_action(to_step: Dict[str, Any], from_step: Dict[str, Any]) -> None:
    # Scale world vector from [-0.7, 0.7] to better span [-2.0, 2.0]
    to_step["action"]["world_vector"] = from_step["action"]["world_vector"] * 2.0
    to_step["action"]["terminate_episode"] = terminate_bool_to_act(
        from_step["action"]["terminate_episode"]
    )

    to_step["action"]["rotation_delta"] = from_step["action"]["rotation_delta"]
    to_step["action"]["gripper_closedness_action"] = tf.expand_dims(
        from_step["action"]["open_gripper"], axis=0
    )
    to_step["action"]["gripper_closedness_action"] = tf.cast(
        to_step["action"]["gripper_closedness_action"], tf.float32
    )


def soar_map_action(
    to_step: Dict[str, Any],
    from_step: Dict[str, Any],
) -> None:
    """Map SOAR actions (7D) to standardized format.

    SOAR actions: [xyz_delta(3), rpy_delta(3), gripper(1)]
    XYZ range ~[-0.06, 0.06], RPY range ~[-6.2, 6.2], gripper [0, 1].
    """
    to_step["action"]["world_vector"] = from_step["action"][0:3]
    to_step["action"]["rotation_delta"] = from_step["action"][3:6]
    # SOAR gripper is [0, 1]; convert to closedness [-1, 1]
    gripper = from_step["action"][6:7]
    gripper = gripper * 2 - 1  # [0,1] -> [-1,1]
    to_step["action"]["gripper_closedness_action"] = gripper

    to_step["action"] = _rescale_action(
        to_step["action"], wv_lo=-0.06, wv_hi=0.06, rd_lo=-6.3, rd_hi=6.3
    )


soar_map_observation = functools.partial(
    map_observation,
    from_image_feature_names=("image_0",),
    to_image_feature_names=("image",),
)


def soar_extract_metadata(episode: Dict[str, Any]) -> Dict[str, Any]:
    """Extract episode-level metadata from SOAR dataset."""
    success = bool(episode["episode_metadata"]["success"])
    steps = list(episode["steps"])
    lang = steps[0]["language_instruction"]
    if isinstance(lang, bytes):
        lang = lang.decode("utf-8")
    return {"success": success, "language_instruction": lang}


def bridge_v2_extract_metadata(episode: Dict[str, Any]) -> Dict[str, Any]:
    """Extract episode-level metadata from bridge_v2 dataset."""
    steps = list(episode["steps"])
    # bridge_v2: reward=1.0 at last step for success
    reward_last = float(steps[-1].get("reward", 1.0))
    lang = steps[0].get("language_instruction", b"")
    if isinstance(lang, bytes):
        lang = lang.decode("utf-8")
    return {"success": reward_last > 0.5, "language_instruction": lang}


def episode_map_fn(
    episode: Dict[str, Any],
    map_step: Callable[[Dict[str, Any]], Dict[str, Any]],
    extract_metadata: Callable[[Dict[str, Any]], Dict[str, Any]] = None,
) -> Dict[str, Any]:
    steps = list(map(map_step, episode["steps"]))
    frames = np.stack([s["observation"]["image"] for s in steps], axis=0)
    result = {
        "video": frames,
        "action": np.stack([s["action"] for s in steps]),
    }
    if extract_metadata is not None:
        result.update(extract_metadata(episode))
    return result


def episode_map_fn_multiview(
    episode: Dict[str, Any],
    map_step: Callable[[Dict[str, Any]], Dict[str, Any]],
    view_keys: Tuple[str, ...] = ("image_0", "image_1", "image_2"),
    extract_metadata: Callable[[Dict[str, Any]], Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Like episode_map_fn but extracts multiple camera views."""
    steps = list(map(map_step, episode["steps"]))
    result = {
        "action": np.stack([s["action"] for s in steps]),
    }
    for key in view_keys:
        frames = np.stack([s["observation"][key] for s in steps], axis=0)
        result[f"video_{key}"] = frames
    if extract_metadata is not None:
        result.update(extract_metadata(episode))
    return result


def step_map_fn(
    step: Dict[str, Any], map_observation: Callable, map_action: Callable
) -> Dict[str, Any]:
    transformed_step = {}

    transformed_step["observation"] = {}
    transformed_step["action"] = {
        "gripper_closedness_action": np.zeros(1, dtype=np.float32),
        "rotation_delta": np.zeros(3, dtype=np.float32),
        "terminate_episode": np.zeros(3, dtype=np.int32),
        "world_vector": np.zeros(3, dtype=np.float32),
        "base_displacement_vertical_rotation": np.zeros(1, dtype=np.float32),
        "base_displacement_vector": np.zeros(2, dtype=np.float32),
    }

    # Apply dataset-specific mappings
    map_observation(transformed_step, step)
    map_action(transformed_step, step)

    # Concatenate action components into single vector
    action = np.concatenate(
        [
            transformed_step["action"]["world_vector"],
            transformed_step["action"]["rotation_delta"],
            transformed_step["action"]["gripper_closedness_action"],
            transformed_step["action"]["base_displacement_vector"],
            transformed_step["action"]["base_displacement_vertical_rotation"],
        ],
        axis=0,
    )
    transformed_step["action"] = action

    return transformed_step


def get_dataset_configs(dataset_home: str) -> Dict[str, Dict[str, Any]]:
    return {
        # RT-1
        "rt_1": {
            "builder_dir": f"{dataset_home}/fractal20220817_data/0.1.0",
            "step_map_fn": functools.partial(
                step_map_fn, map_observation=map_observation, map_action=rt_1_map_action
            ),
        },
        # Bridge
        "bridge": {
            "builder_dir": f"{dataset_home}/bridge/0.1.0",
            "step_map_fn": functools.partial(
                step_map_fn,
                map_observation=map_observation,
                map_action=functools.partial(bridge_map_action),
            ),
        },
        "bridge_v2": {
            "builder_dir": BRIDGE_V2_PATH,
            "step_map_fn": functools.partial(
                step_map_fn,
                map_observation=bridge_v2_map_observation,
                map_action=functools.partial(bridge_v2_map_action),
            ),
            "extract_metadata": bridge_v2_extract_metadata,
            "splits": ["train", "val"],
            "split_rename": {"val": "test"},
        },
        "bridge_v2_multiview": {
            "builder_dir": BRIDGE_V2_PATH,
            "step_map_fn": functools.partial(
                step_map_fn,
                map_observation=bridge_v2_map_observation_multiview,
                map_action=functools.partial(bridge_v2_map_action),
            ),
            "extract_metadata": bridge_v2_extract_metadata,
            "multiview": True,
            "view_keys": ("image_0", "image_1", "image_2"),
            "splits": ["train", "val"],
        },
        # SOAR (success/failure bridge-robot dataset)
        "soar": {
            "builder_dir": SOAR_PATH,
            "step_map_fn": functools.partial(
                step_map_fn,
                map_observation=soar_map_observation,
                map_action=soar_map_action,
            ),
            "extract_metadata": soar_extract_metadata,
            "splits": ["success", "failure"],
            "custom_splits": True,
        },
        # LIBERO
        "libero_10": {
            "builder_dir": f"{dataset_home}/LIBERO/libero/modified_libero_rlds/libero_10_no_noops/1.0.0",
            "step_map_fn": functools.partial(
                step_map_fn,
                map_observation=map_observation,
                map_action=functools.partial(libero_map_action),
            ),
        },
        "libero_object": {
            "builder_dir": f"{dataset_home}/LIBERO/libero/modified_libero_rlds/libero_object_no_noops/1.0.0",
            "step_map_fn": functools.partial(
                step_map_fn,
                map_observation=map_observation,
                map_action=functools.partial(libero_map_action),
            ),
        },
        "libero_goal": {
            "builder_dir": f"{dataset_home}/LIBERO/libero/modified_libero_rlds/libero_goal_no_noops/1.0.0",
            "step_map_fn": functools.partial(
                step_map_fn,
                map_observation=map_observation,
                map_action=functools.partial(libero_map_action),
            ),
        },
        "libero_spatial": {
            "builder_dir": f"{dataset_home}/LIBERO/libero/modified_libero_rlds/libero_spatial_no_noops/1.0.0",
            "step_map_fn": functools.partial(
                step_map_fn,
                map_observation=map_observation,
                map_action=functools.partial(libero_map_action),
            ),
        },
        # Task Agnostic Robot Play
        "taco_play": {
            "builder_dir": f"{dataset_home}/taco_play/0.1.0",
            "step_map_fn": functools.partial(
                step_map_fn,
                map_observation=taco_play_map_observation,
                map_action=functools.partial(taco_play_map_action),
            ),
        },
        # Jaco Play
        "jaco_play": {
            "builder_dir": f"{dataset_home}/jaco_play/0.1.0",
            "step_map_fn": functools.partial(
                step_map_fn,
                map_observation=map_observation,
                map_action=jaco_play_map_action,
            ),
        },
        # Cable Routing
        "berkeley_cable_routing": {
            "builder_dir": f"{dataset_home}/berkeley_cable_routing/0.1.0",
            "step_map_fn": functools.partial(
                step_map_fn,
                map_observation=map_observation,
                map_action=berkeley_cable_routing_map_action,
            ),
        },
        # Roboturk
        "roboturk": {
            "builder_dir": f"{dataset_home}/roboturk/0.1.0",
            "step_map_fn": functools.partial(
                step_map_fn,
                map_observation=roboturk_map_observation,
                map_action=roboturk_map_action,
            ),
        },
        # NYU VINN
        "nyu_door_opening_surprising_effectiveness": {
            "builder_dir": f"{dataset_home}/nyu_door_opening_surprising_effectiveness/0.1.0",
            "step_map_fn": functools.partial(
                step_map_fn,
                map_observation=map_observation,
                map_action=nyu_door_opening_surprising_effectiveness_map_action,
            ),
        },
        # Austin VIOLA
        "viola": {
            "builder_dir": f"{dataset_home}/viola/0.1.0",
            "step_map_fn": functools.partial(
                step_map_fn,
                map_observation=viola_map_observation,
                map_action=viola_map_action,
            ),
        },
        # Berkeley Autolab UR5
        "berkeley_autolab_ur5": {
            "builder_dir": f"{dataset_home}/berkeley_autolab_ur5/0.1.0",
            "step_map_fn": functools.partial(
                step_map_fn,
                map_observation=map_observation,
                map_action=berkeley_autolab_ur5_map_action,
            ),
        },
        # TOTO
        "toto": {
            "builder_dir": f"{dataset_home}/toto/0.1.0",
            "step_map_fn": functools.partial(
                step_map_fn, map_observation=map_observation, map_action=toto_map_action
            ),
        },
    }


def _save_episode(
    episode: Dict[str, Any],
    base_path: Path,
    fps: int,
) -> None:
    """Save a processed episode as MP4 + NPZ.

    For multi-view episodes (keys like ``video_image_0``, ``video_image_1``, ...),
    each view is saved as a separate MP4: ``{base_path}_view0.mp4``, etc.
    """
    action = episode["action"]
    view_keys = sorted(k for k in episode if k.startswith("video_image_"))

    if view_keys:
        # Multi-view: save each view as a separate MP4
        for i, vk in enumerate(view_keys):
            video = torch.from_numpy(episode[vk])
            write_video(f"{base_path}_view{i}.mp4", video, fps=fps)
    else:
        # Single-view (original behaviour)
        video = torch.from_numpy(episode["video"])
        write_video(str(base_path) + ".mp4", video, fps=fps)

    save_dict = {"actions": action}
    if "success" in episode:
        save_dict["success"] = np.array(episode["success"], dtype=np.bool_)
    if "language_instruction" in episode:
        save_dict["language_instruction"] = np.array(episode["language_instruction"])
    np.savez(str(base_path) + ".npz", **save_dict)


def convert_dataset(
    dataset_name: str,
    dataset_config: Dict[str, Any],
    output_dir: str,
    fps: int,
    test_ratio: float = 0.1,
    test_only: bool = False,
) -> None:
    print(f"Converting {dataset_name}...")

    # Create output directories
    dataset_output_dir = Path(output_dir) / dataset_name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    # Build dataset
    print("Building dataset...")
    try:
        dataset_builder = tfds.builder_from_directory(
            builder_dir=dataset_config["builder_dir"]
        )
    except Exception as e:
        print(f"Failed to build dataset {dataset_name}: {e}")
        return

    print("Dataset built successfully.")

    extract_metadata = dataset_config.get("extract_metadata", None)
    is_multiview = dataset_config.get("multiview", False)
    view_keys = dataset_config.get("view_keys", ())
    splits = dataset_config.get("splits", ["train", "test"])
    if test_only:
        # Keep only splits that map to "test" output dir, or are named "test"
        split_rename = dataset_config.get("split_rename", {})
        splits = [s for s in splits if split_rename.get(s, s) == "test"]
    # If dataset has custom splits (e.g. SOAR: success/failure), merge into train/test
    has_custom_splits = dataset_config.get("custom_splits", False)

    if has_custom_splits:
        # Merge all custom splits, then split into train/test by ratio
        train_dir = dataset_output_dir / "train"
        test_dir = dataset_output_dir / "test"
        train_dir.mkdir(exist_ok=True)
        test_dir.mkdir(exist_ok=True)

        train_idx = 0
        test_idx = 0

        for split_name in splits:
            try:
                dataset = dataset_builder.as_dataset(
                    split=split_name, shuffle_files=True
                )
            except Exception as e:
                print(f"Split {split_name} not available for {dataset_name}: {e}")
                continue

            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            dataset = tfds.as_numpy(dataset)

            print(f"Processing {split_name} split...")
            for i, episode in tqdm(
                enumerate(dataset), desc=f"{dataset_name}-{split_name}"
            ):
                try:
                    episode = episode_map_fn(
                        episode,
                        map_step=dataset_config["step_map_fn"],
                        extract_metadata=extract_metadata,
                    )

                    # Deterministic train/test assignment
                    is_test = (hash(f"{split_name}_{i}") % 1000) < int(
                        test_ratio * 1000
                    )
                    if is_test:
                        base_path = test_dir / f"{test_idx:09d}"
                        test_idx += 1
                    else:
                        base_path = train_dir / f"{train_idx:09d}"
                        train_idx += 1

                    _save_episode(episode, base_path, fps)

                except Exception as e:
                    print(
                        f"Failed to process episode {i} in {dataset_name}-{split_name}: {e}"
                    )
                    continue

        print(f"Saved {train_idx} train, {test_idx} test episodes for {dataset_name}")

    else:
        # Standard train/test splits
        split_rename = dataset_config.get("split_rename", {})
        for split in splits:
            # Use renamed directory if configured (e.g. TFDS "val" -> output "test")
            output_split = split_rename.get(split, split)
            split_output_dir = dataset_output_dir / output_split
            split_output_dir.mkdir(exist_ok=True)

            try:
                dataset = dataset_builder.as_dataset(split=split, shuffle_files=True)
            except Exception as e:
                print(f"Split {split} not available for {dataset_name}: {e}")
                continue

            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            dataset = tfds.as_numpy(dataset)

            print(f"Processing {split} split...")
            saved_idx = 0
            skipped_multiview = 0
            for i, episode in tqdm(enumerate(dataset), desc=f"{dataset_name}-{split}"):
                try:
                    # For multi-view datasets, skip episodes missing required views
                    if is_multiview:
                        steps_list = list(episode["steps"])
                        obs = steps_list[0].get("observation", {})
                        if not all(vk in obs for vk in view_keys):
                            skipped_multiview += 1
                            continue
                        episode["steps"] = steps_list
                        episode = episode_map_fn_multiview(
                            episode,
                            map_step=dataset_config["step_map_fn"],
                            view_keys=view_keys,
                            extract_metadata=extract_metadata,
                        )
                    else:
                        episode = episode_map_fn(
                            episode,
                            map_step=dataset_config["step_map_fn"],
                            extract_metadata=extract_metadata,
                        )

                    base_path = split_output_dir / f"{saved_idx:09d}"
                    _save_episode(episode, base_path, fps)
                    saved_idx += 1

                except Exception as e:
                    print(
                        f"Failed to process episode {i} in {dataset_name}-{split}: {e}"
                    )
                    continue

            if is_multiview:
                print(
                    f"  {split}: saved {saved_idx}, skipped {skipped_multiview} (missing views)"
                )


def main(
    dataset_name: str,
    output_dir: str = "converted_datasets",
    dataset_home: str = "gs://gresearch/robotics",
    fps: int = 20,
    test_only: bool = False,
) -> None:
    dataset_configs = get_dataset_configs(dataset_home=dataset_home)

    if dataset_name == "all":
        selected_configs = dataset_configs
    else:
        if dataset_name not in dataset_configs.keys():
            raise ValueError(
                f"Invalid dataset name: {dataset_name}."
                "Please choose from one of {dataset_configs.keys()} or 'all' for all datasets."
            )
        selected_configs = {dataset_name: dataset_configs[dataset_name]}

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    for dataset_name, dataset_config in selected_configs.items():
        convert_dataset(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            output_dir=output_dir,
            fps=fps,
            test_only=test_only,
        )


if __name__ == "__main__":
    fire.Fire(main)
