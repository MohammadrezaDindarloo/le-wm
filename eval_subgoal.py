"""
Hierarchical Subgoal Planning Evaluation for LeWM
==================================================

Idea
----
The base LeWM evaluation plans toward a goal image that is only ~25 dataset
steps ahead, which the authors identify as a short-horizon limitation.

Here we address this WITHOUT touching any training or model code:
  1. Take a full expert episode (start → end) from the dataset.
  2. Extract intermediate subgoal images every `subgoal.spacing` steps.
  3. Chain the LeWM planner: reach subgoal_1, then subgoal_2, ..., final_goal.

Each segment uses the same WorldModelPolicy (CEM) as the original eval.
Switching a subgoal flushes the action buffer so the planner re-optimises
immediately toward the new target.

Usage
-----
  python eval_subgoal.py policy=pusht/lewm
  python eval_subgoal.py policy=pusht/lewm subgoal.spacing=50 eval.num_eval=5
"""

import os
os.environ["MUJOCO_GL"] = "egl"

import time
from copy import deepcopy
from pathlib import Path

import hydra
import imageio
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm


# ── Image transform (identical to eval.py) ────────────────────────────────────

def img_transform(cfg):
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=cfg.eval.img_size),
    ])


# ── Dataset helpers ───────────────────────────────────────────────────────────

def get_dataset(cfg):
    cache = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    return swm.data.HDF5Dataset(
        cfg.eval.dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=cache,
    )


def fit_preprocessors(dataset, keys_to_cache):
    """Fit a StandardScaler for each non-pixel column (action, proprio, state).

    The scalers are used by WorldModelPolicy._prepare_info() to normalise
    inputs before passing them to the world model.
    """
    process = {}
    for col in keys_to_cache:
        if col == "pixels":
            continue
        scaler = preprocessing.StandardScaler()
        data = dataset.get_col_data(col)
        data = data[~np.isnan(data).any(axis=1)]
        scaler.fit(data)
        process[col] = scaler
        process[f"goal_{col}"] = scaler   # same scaler for the goal version of the key
    return process


# ── Subgoal schedule ──────────────────────────────────────────────────────────

def build_subgoal_schedule(ep_length: int, spacing: int) -> list:
    """Return the list of dataset frame indices that serve as subgoals.

    Example: ep_length=109, spacing=25 → [25, 50, 75, 100, 108]
    The last entry is always ep_length-1 (the true final goal frame).
    """
    indices = list(range(spacing, ep_length, spacing))
    # always include the very last frame as the final goal
    if not indices or indices[-1] != ep_length - 1:
        indices.append(ep_length - 1)
    return indices


# ── Policy helpers ────────────────────────────────────────────────────────────

def flush_plan(policy):
    """Clear the WorldModelPolicy action buffer to force an immediate replan.

    Without this, the planner would keep executing the old buffered actions
    even after we switch the goal image to the next subgoal.
    """
    if hasattr(policy, "_action_buffer"):
        policy._action_buffer.clear()
    if hasattr(policy, "_next_init"):
        policy._next_init = None


# ── Environment helpers ────────────────────────────────────────────────────────

def get_raw_env(world):
    """Traverse wrapper layers to reach the bare PushT env instance."""
    return world.envs.unwrapped.envs[0].unwrapped


def init_episode(world, ep_data):
    """Reset the environment to the start state of the given expert episode.

    Steps
    -----
    1. Reset the env (initialises StackedWrapper buffers inside the World).
    2. Set the physical state to the episode's first frame via _set_state.
    3. Set the success-check target to the episode's *last* frame via
       _set_goal_state.  This stays fixed for the whole episode, so the env's
       `terminated` signal always fires when the T-block reaches the true goal.
    4. Re-render the pixels for the new physical position and write them into
       world.infos so the policy sees the correct initial observation.

    Returns
    -------
    final_goal_px : np.ndarray, shape (H, W, C), uint8
        The rendered goal image (last frame), used for video generation.
    """
    world.reset()

    raw_env = get_raw_env(world)
    init_state  = ep_data["state"][0].numpy()   # shape (7,)
    final_state = ep_data["state"][-1].numpy()  # shape (7,)

    # Set physics to the episode's starting configuration.
    raw_env._set_state(init_state)

    # Fix the success target to the final configuration of the episode.
    # The env's step() calls eval_state(self.goal_state, cur_state), so this
    # is what determines when `terminated` becomes True.
    raw_env._set_goal_state(final_state)

    # Re-render pixels at the new position (_set_state does NOT auto-render).
    init_pixels = raw_env.render()   # (H, W, C) uint8

    # Inject into world.infos so the policy reads the correct initial frame.
    # Shape must be (num_envs=1, history_size=1, H, W, C).
    shape_prefix = world.infos["pixels"].shape[:2]
    world.infos["pixels"] = np.broadcast_to(
        init_pixels[None, None], shape_prefix + init_pixels.shape
    ).copy()

    # The goal image (last frame), stored for video generation.
    final_goal_px = ep_data["pixels"][-1].permute(1, 2, 0).numpy()  # (H, W, C)

    # Disable auto-reset so the env does not reset on its own after termination.
    world.envs.unwrapped._autoreset_envs = np.zeros((world.num_envs,), dtype=bool)

    return final_goal_px


def inject_goal(world, subgoal_px):
    """Overwrite world.infos['goal'] with the given subgoal image.

    This must be called before every world.step() call because world.step()
    internally calls the env's _get_info(), which writes the env's own reset-
    time goal back into infos, clobbering our subgoal.  The fix (matching
    evaluate_from_dataset) is to re-inject before each step.

    Parameters
    ----------
    subgoal_px : np.ndarray, shape (H, W, C), uint8
    """
    shape_prefix = world.infos["pixels"].shape[:2]   # (1, 1)
    world.infos["goal"] = np.broadcast_to(
        subgoal_px[None, None], shape_prefix + subgoal_px.shape
    ).copy()


# ── Segment runner ────────────────────────────────────────────────────────────

def run_segment(world, subgoal_px, budget, obs_frames, subgoal_frames):
    """Run the planner toward one subgoal for up to `budget` env steps.

    The env's success check is always against the FINAL goal_state (set in
    init_episode), not the intermediate subgoal image.  So `terminated=True`
    means the agent has actually reached the true final goal, not just the
    current subgoal.

    Parameters
    ----------
    world       : swm.World with policy already set
    subgoal_px  : np.ndarray (H, W, C) uint8 — current subgoal image
    budget      : int — max env steps for this segment
    obs_frames  : list — appended with each current observation frame (for video)
    subgoal_frames : list — appended with the active subgoal frame (for video)

    Returns
    -------
    reached : bool  — True if env terminated (final goal reached)
    steps   : int   — how many env steps were actually taken
    """
    for t in range(budget):
        # Inject goal before every step (env.step clobbers world.infos['goal']).
        inject_goal(world, subgoal_px)

        # Record the current observation for the video.
        obs_frames.append(world.infos["pixels"][0, -1].copy())   # (H, W, C)
        subgoal_frames.append(subgoal_px)

        world.step()

        # Prevent auto-reset: keep the env alive even after termination.
        world.envs.unwrapped._autoreset_envs = np.zeros((world.num_envs,), dtype=bool)

        if world.terminateds[0]:
            return True, t + 1

    return False, budget


# ── Video builder ─────────────────────────────────────────────────────────────

def save_episode_video(path, obs_frames, subgoal_frames, final_goal_px, fps=15):
    """Save a 4-panel MP4 video for one episode.

    Panel layout (matches the original eval.py style):

        ┌─────────────┬─────────────┐
        │ current obs │ final goal  │   ← final_goal fixed throughout
        ├─────────────┼─────────────┤
        │current sgoa │ final goal  │   ← subgoal changes each segment
        └─────────────┴─────────────┘
    """
    writer = imageio.get_writer(str(path), fps=fps, codec="libx264")
    goals_col = np.vstack([final_goal_px, final_goal_px])   # right column (fixed)

    for obs, sg in zip(obs_frames, subgoal_frames):
        left_col = np.vstack([obs, sg])
        frame = np.hstack([left_col, goals_col])
        writer.append_data(frame)

    writer.close()


# ── Per-episode evaluation ────────────────────────────────────────────────────

def evaluate_episode(world, policy, ep_data, cfg):
    """Run the full hierarchical plan for one expert episode.

    Returns a dict with:
      success          : bool   — reached the final goal at any point
      total_steps      : int    — total env steps taken
      num_subgoals     : int    — number of subgoal segments attempted
      steps_per_segment: list   — steps used in each segment
      obs_frames       : list   — raw pixel frames for video
      subgoal_frames   : list   — active subgoal image at each frame (for video)
      final_goal_px    : ndarray
    """
    ep_length = ep_data["state"].shape[0]

    # Build the list of subgoal frame indices for this episode.
    subgoal_indices = build_subgoal_schedule(ep_length, cfg.subgoal.spacing)

    # Initialise env at episode start, get final goal image.
    final_goal_px = init_episode(world, ep_data)

    # Flush any leftover actions from a previous episode.
    flush_plan(policy)

    obs_frames     = []
    subgoal_frames = []
    steps_per_seg  = []
    success        = False

    for seg_idx, frame_idx in enumerate(subgoal_indices):
        # Extract the subgoal image from the dataset (channel-first → channel-last).
        subgoal_px = ep_data["pixels"][frame_idx].permute(1, 2, 0).numpy()

        # Inject the new subgoal and force an immediate replan.
        inject_goal(world, subgoal_px)
        flush_plan(policy)

        reached, steps = run_segment(
            world,
            subgoal_px,
            cfg.subgoal.segment_budget,
            obs_frames,
            subgoal_frames,
        )
        steps_per_seg.append(steps)

        if reached:
            success = True
            break
        # If budget exhausted, move on to the next subgoal anyway.
        # The agent continues from wherever it ended up.

    return {
        "success":          success,
        "total_steps":      sum(steps_per_seg),
        "num_subgoals":     len(subgoal_indices),
        "steps_per_segment": steps_per_seg,
        "obs_frames":       obs_frames,
        "subgoal_frames":   subgoal_frames,
        "final_goal_px":    final_goal_px,
    }


# ── Main entry point ──────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht_subgoal")
def run(cfg: DictConfig):

    # ── Assertion: planning params must be consistent ─────────────────────────
    assert cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.subgoal.segment_budget, (
        "horizon * action_block must be <= subgoal.segment_budget "
        f"(got {cfg.plan_config.horizon * cfg.plan_config.action_block} > {cfg.subgoal.segment_budget})"
    )

    # ── World setup ───────────────────────────────────────────────────────────
    # num_envs is forced to 1 so each episode has its own subgoal schedule.
    # max_episode_steps must be large enough for all segments.
    max_ep_steps = (len(build_subgoal_schedule(200, cfg.subgoal.spacing))
                    * cfg.subgoal.segment_budget + 10)
    cfg.world.max_episode_steps = max_ep_steps
    world = swm.World(**cfg.world, image_shape=(224, 224))

    # ── Transforms and dataset ────────────────────────────────────────────────
    transform = {
        "pixels": img_transform(cfg),
        "goal":   img_transform(cfg),
    }
    dataset = get_dataset(cfg)
    process = fit_preprocessors(dataset, cfg.dataset.keys_to_cache)

    # ── Policy ────────────────────────────────────────────────────────────────
    model = swm.policy.AutoCostModel(cfg.policy)
    model = model.to("cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    plan_config = swm.PlanConfig(**cfg.plan_config)
    solver      = hydra.utils.instantiate(cfg.solver, model=model)
    policy      = swm.policy.WorldModelPolicy(
        solver=solver, config=plan_config, process=process, transform=transform
    )
    world.set_policy(policy)

    # ── Episode selection ─────────────────────────────────────────────────────
    # dataset.lengths[i] = number of steps in episode i (positional index).
    # In pusht_expert_train, episode_idx column == positional index (0-based).
    ep_lengths = dataset.lengths
    valid_ep_ids = np.where(ep_lengths >= cfg.eval.min_ep_length)[0]
    print(f"Episodes in dataset: {len(ep_lengths)}, "
          f"valid (>= {cfg.eval.min_ep_length} steps): {len(valid_ep_ids)}")

    rng = np.random.default_rng(cfg.seed)
    sampled_ep_ids = rng.choice(valid_ep_ids, size=cfg.eval.num_eval, replace=False)
    sampled_ep_ids = np.sort(sampled_ep_ids)
    print(f"Evaluating {len(sampled_ep_ids)} episodes: {sampled_ep_ids}")

    # ── Output paths ─────────────────────────────────────────────────────────
    # Must NOT use <cache>/<policy> directly — AutoCostModel checks that path
    # for a directory and globs for *_object.ckpt inside it → IndexError.
    policy_parent = Path(cfg.policy).parent   # e.g. "pusht"
    results_dir = Path(swm.data.utils.get_cache_dir(), policy_parent, "subgoal_evals", cfg.output.get("run_name", "default"))
    results_dir.mkdir(parents=True, exist_ok=True)
    video_dir = results_dir / "videos"
    if cfg.output.save_video:
        video_dir.mkdir(exist_ok=True)

    # ── Evaluation loop ───────────────────────────────────────────────────────
    all_results = []
    start_time  = time.time()

    for i, ep_id in enumerate(sampled_ep_ids):
        print(f"\n── Episode {i+1}/{len(sampled_ep_ids)}  (dataset ep_id={ep_id}, "
              f"length={ep_lengths[ep_id]}) ──")

        ep_data = dataset.load_episode(int(ep_id))
        sg_schedule = build_subgoal_schedule(ep_lengths[ep_id], cfg.subgoal.spacing)
        print(f"   Subgoal schedule ({len(sg_schedule)} segments): "
              f"frames {sg_schedule}")

        ep_result = evaluate_episode(world, policy, ep_data, cfg)
        all_results.append(ep_result)

        status = "SUCCESS" if ep_result["success"] else "fail"
        print(f"   {status} | total_steps={ep_result['total_steps']} | "
              f"steps_per_seg={ep_result['steps_per_segment']}")

        if cfg.output.save_video:
            video_path = video_dir / f"rollout_{i:03d}_ep{ep_id}.mp4"
            save_episode_video(
                video_path,
                ep_result["obs_frames"],
                ep_result["subgoal_frames"],
                ep_result["final_goal_px"],
            )

    elapsed = time.time() - start_time

    # ── Metrics ───────────────────────────────────────────────────────────────
    successes    = [r["success"] for r in all_results]
    success_rate = sum(successes) / len(successes) * 100.0
    avg_steps    = np.mean([r["total_steps"] for r in all_results])

    print(f"\n{'='*60}")
    print(f"SUCCESS RATE : {success_rate:.1f}%  ({sum(successes)}/{len(successes)})")
    print(f"AVG STEPS    : {avg_steps:.1f}")
    print(f"EVAL TIME    : {elapsed:.1f}s")
    print(f"{'='*60}")

    # ── Save results ──────────────────────────────────────────────────────────
    results_path = results_dir / cfg.output.filename
    with results_path.open("a") as f:
        f.write("\n")
        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n==== RESULTS ====\n")
        f.write(f"success_rate: {success_rate}\n")
        f.write(f"num_successes: {sum(successes)}\n")
        f.write(f"num_episodes: {len(successes)}\n")
        f.write(f"avg_total_steps: {avg_steps:.1f}\n")
        f.write(f"evaluation_time: {elapsed:.1f}s\n")
        f.write("\n-- Per-episode --\n")
        for i, (ep_id, r) in enumerate(zip(sampled_ep_ids, all_results)):
            f.write(
                f"ep{i:03d} id={ep_id} success={r['success']} "
                f"total_steps={r['total_steps']} "
                f"steps_per_seg={r['steps_per_segment']}\n"
            )

    print(f"\nResults written to: {results_path}")


if __name__ == "__main__":
    run()
