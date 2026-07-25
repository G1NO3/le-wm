#!/usr/bin/env python
"""Render a qualitative FetchPush commitment comparison from an exact fork.

The default scene is a fresh confirmation context where the action selected
from the learned residual mean misses under both hidden friction outcomes and
the action selected from the learned stochastic modes succeeds under both.
Selection is replayed from the saved evaluator indices; this script only
reconstructs and visualizes the exact simulator outcomes.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import numpy as np
import stable_worldmodel  # noqa: F401 - registers swm environments
from PIL import Image, ImageDraw, ImageFont

from fetch_push_expert import FetchPushExpertPolicy
from scripts.eval.evaluate_fetch_push_commitment import (
    collect_push_context,
    push_pulse_candidates,
)
from scripts.eval.generate_fetch_push_fork_samples import fixed_profile, render, set_mode
from stochastic_physics import FetchPushHiddenFriction


MODES = (("LOW FRICTION", 0.2), ("HIGH FRICTION", 3.0))
METHODS = ("Residual mean", "Stochastic residual")
PANEL_SIZE = 360
MARGIN = 22
GAP = 14
HEADER_HEIGHT = 122
COLUMN_HEADER_HEIGHT = 64
ROW_HEADER_HEIGHT = 34
PANEL_FOOTER_HEIGHT = 54
BOTTOM_HEIGHT = 46
FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--poster", type=Path)
    parser.add_argument("--seed", type=int, default=1803092)
    parser.add_argument("--mean-index", type=int, default=22)
    parser.add_argument("--stochastic-index", type=int, default=92)
    parser.add_argument("--raw-horizon", type=int, default=20)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--initial-hold", type=int, default=6)
    parser.add_argument("--final-hold", type=int, default=12)
    parser.add_argument("--success-threshold", type=float, default=0.05)
    return parser.parse_args()


def font(size, *, bold=False):
    path = FONT_BOLD if bold else FONT_REGULAR
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def centered(draw, box, text, text_font, fill):
    left, top, right, bottom = box
    bounds = draw.textbbox((0, 0), text, font=text_font)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    draw.text(
        ((left + right - width) / 2, (top + bottom - height) / 2 - bounds[1]),
        text,
        font=text_font,
        fill=fill,
    )


def replay(
    env,
    fork_state,
    actions,
    multiplier,
    threshold,
    *,
    initial_hold,
    final_hold,
):
    env.set_fork_state(set_mode(fork_state, multiplier))
    frames = [render(env)]
    distances = [float(np.linalg.norm(env._object_position() - env._goal_position()))]
    for action in actions:
        env.step(action)
        frames.append(render(env))
        distances.append(
            float(np.linalg.norm(env._object_position() - env._goal_position()))
        )
    frames = [frames[0]] * initial_hold + frames + [frames[-1]] * final_hold
    distances = [distances[0]] * initial_hold + distances + [distances[-1]] * final_hold
    running_minimum = np.minimum.accumulate(distances)
    successes = running_minimum <= threshold
    return frames, np.asarray(distances), running_minimum, successes


def panel_frame(frame, distance, running_minimum, success, final, threshold):
    image = Image.fromarray(frame).resize((PANEL_SIZE, PANEL_SIZE), Image.Resampling.LANCZOS)
    footer = Image.new("RGB", (PANEL_SIZE, PANEL_FOOTER_HEIGHT), "#111821")
    footer_draw = ImageDraw.Draw(footer)
    status = "SUCCESS" if success else ("MISS" if final else "PUSHING")
    status_color = "#48d17a" if success else ("#ff6b6b" if final else "#f5c451")
    footer_draw.text((12, 7), status, font=font(19, bold=True), fill=status_color)
    footer_draw.text(
        (12, 31),
        f"distance {distance * 100:.1f} cm  |  closest {running_minimum * 100:.1f} cm",
        font=font(13),
        fill="#e4e9ef",
    )
    border_color = status_color if (success or final) else "#617082"
    combined = Image.new("RGB", (PANEL_SIZE, PANEL_SIZE + PANEL_FOOTER_HEIGHT))
    combined.paste(image, (0, 0))
    combined.paste(footer, (0, PANEL_SIZE))
    draw = ImageDraw.Draw(combined)
    draw.rectangle((1, 1, combined.width - 2, combined.height - 2), outline=border_color, width=4)
    if final:
        banner_color = "#217a45" if success else "#9f3434"
        draw.rounded_rectangle((105, 150, 255, 205), radius=9, fill=banner_color)
        centered(draw, (105, 150, 255, 205), status, font(24, bold=True), "white")
    return combined


def compose(index, trajectories, parameters, threshold, seed):
    width = MARGIN * 2 + PANEL_SIZE * 2 + GAP
    row_height = ROW_HEADER_HEIGHT + PANEL_SIZE + PANEL_FOOTER_HEIGHT
    height = (
        HEADER_HEIGHT
        + COLUMN_HEADER_HEIGHT
        + row_height * 2
        + GAP
        + BOTTOM_HEIGHT
        + MARGIN
    )
    canvas = Image.new("RGB", (width, height), "#080d13")
    draw = ImageDraw.Draw(canvas)

    centered(
        draw,
        (MARGIN, 10, width - MARGIN, 48),
        "Friction-varying FetchPush — fresh held-out scene",
        font(25, bold=True),
        "#f6f8fb",
    )
    centered(
        draw,
        (MARGIN, 48, width - MARGIN, 78),
        "Stochastic residual selects a push that succeeds in both hidden modes",
        font(16, bold=True),
        "#83d9ff",
    )
    centered(
        draw,
        (MARGIN, 79, width - MARGIN, 112),
        "1,024-context result: 52.3% → 56.2%  (+3.8 pp; 95% CI +1.8 to +5.9)",
        font(15),
        "#cad4df",
    )

    column_y = HEADER_HEIGHT
    for column, method in enumerate(METHODS):
        x = MARGIN + column * (PANEL_SIZE + GAP)
        param = parameters[column]
        centered(
            draw,
            (x, column_y, x + PANEL_SIZE, column_y + 31),
            method,
            font(20, bold=True),
            "#f6f8fb",
        )
        centered(
            draw,
            (x, column_y + 31, x + PANEL_SIZE, column_y + COLUMN_HEADER_HEIGHT),
            f"selected pulse: speed {param['speed']:.1f}, duration {param['duration']} steps",
            font(13),
            "#aab7c5",
        )

    final_index = len(next(iter(trajectories.values()))[0]) - 1
    for row, (mode_name, multiplier) in enumerate(MODES):
        y = HEADER_HEIGHT + COLUMN_HEADER_HEIGHT + row * (row_height + GAP)
        centered(
            draw,
            (MARGIN, y, width - MARGIN, y + ROW_HEADER_HEIGHT),
            f"{mode_name}  ({multiplier:g}× surface friction)",
            font(16, bold=True),
            "#f5c451" if row == 0 else "#c3a2ff",
        )
        for column in range(2):
            x = MARGIN + column * (PANEL_SIZE + GAP)
            frames, distances, minima, successes = trajectories[(column, row)]
            panel = panel_frame(
                frames[index],
                distances[index],
                minima[index],
                bool(successes[index]),
                index == final_index,
                threshold,
            )
            canvas.paste(panel, (x, y + ROW_HEADER_HEIGHT))

    footer_y = height - BOTTOM_HEIGHT - MARGIN
    centered(
        draw,
        (MARGIN, footer_y, width - MARGIN, footer_y + BOTTOM_HEIGHT),
        f"Success = within {threshold * 100:g} cm of goal  •  scene seed {seed}  •  friction is hidden from the selector",
        font(13),
        "#91a0ae",
    )
    return canvas


def encode_video(frames, output, fps):
    with tempfile.TemporaryDirectory(prefix="fetch-push-demo-") as temporary:
        temporary_path = Path(temporary)
        for index, frame in enumerate(frames):
            frame.save(temporary_path / f"frame{index:04d}.png")
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-framerate",
                str(fps),
                "-i",
                str(temporary_path / "frame%04d.png"),
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output),
            ],
            check=True,
        )


def main():
    args = parse_args()
    for path in (args.output, args.poster):
        if path is not None and path.exists():
            raise SystemExit(f"Refusing to overwrite existing artifact: {path}")
    if args.fps <= 0:
        raise SystemExit("--fps must be positive")

    env = FetchPushHiddenFriction(
        gym.make("swm/FetchPush-v3", max_episode_steps=100, render_mode="rgb_array"),
        profile=fixed_profile(1.0),
        terminate_at_goal=False,
    )
    try:
        context = collect_push_context(env, FetchPushExpertPolicy(seed=args.seed), args.seed)
        if context is None:
            raise SystemExit(f"Could not reach push onset for seed {args.seed}")
        candidates, parameters = push_pulse_candidates(
            context["state"],
            raw_horizon=args.raw_horizon,
            speed_min=0.1,
            speed_max=1.0,
            speed_count=10,
            duration_min=2,
            duration_stride=2,
        )
        selected = (args.mean_index, args.stochastic_index)
        if min(selected) < 0 or max(selected) >= len(candidates):
            raise SystemExit(f"Candidate index must be in [0, {len(candidates) - 1}]")

        trajectories = {}
        for column, candidate_index in enumerate(selected):
            for row, (_, multiplier) in enumerate(MODES):
                trajectories[(column, row)] = replay(
                    env,
                    context["fork_state"],
                    candidates[candidate_index],
                    multiplier,
                    args.success_threshold,
                    initial_hold=args.initial_hold,
                    final_hold=args.final_hold,
                )
        length = len(trajectories[(0, 0)][0])
        composed = [
            compose(
                index,
                trajectories,
                (parameters[selected[0]], parameters[selected[1]]),
                args.success_threshold,
                args.seed,
            )
            for index in range(length)
        ]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        encode_video(composed, args.output, args.fps)
        if args.poster is not None:
            args.poster.parent.mkdir(parents=True, exist_ok=True)
            composed[-1].save(args.poster)

        print(f"video={args.output}")
        if args.poster is not None:
            print(f"poster={args.poster}")
        for column, method in enumerate(METHODS):
            for row, (mode_name, multiplier) in enumerate(MODES):
                minimum = trajectories[(column, row)][2][-1]
                status = "success" if minimum <= args.success_threshold else "miss"
                print(
                    f"{method} | {mode_name} ({multiplier:g}x): "
                    f"{status}, closest={minimum * 100:.2f} cm"
                )
    finally:
        env.close()


if __name__ == "__main__":
    main()
