import csv
import os

import numpy as np
import pymunk

from render_utils import render_frame


def generate_single_video(drag_coeff, initial_conditions, num_frames=64, width=224, height=224):
    space = pymunk.Space()
    space.gravity = (0, 0)

    mass = 1.0
    radius = 15
    moment = pymunk.moment_for_circle(mass, 0, radius)
    body = pymunk.Body(mass, moment)
    body.position = (float(initial_conditions["x"]), float(initial_conditions["y"]))
    body.velocity = (float(initial_conditions["vx"]), float(initial_conditions["vy"]))
    init_position = (float(body.position.x), float(body.position.y))
    init_velocity = (float(body.velocity.x), float(body.velocity.y))

    shape = pymunk.Circle(body, radius)
    shape.friction = 0.0
    shape.elasticity = 1.0
    space.add(body, shape)

    dt = 1.0 / 240.0
    frames = []
    x, y = body.position
    frames.append(render_frame(x, y, radius, width=width, height=height, draw_floor=False))
    for _ in range(num_frames - 1):
        for _ in range(4):
            space.step(dt)
            s = 1.0 - drag_coeff * dt
            body.velocity = (body.velocity.x * s, body.velocity.y * s)
        x, y = body.position
        frames.append(render_frame(x, y, radius, width=width, height=height, draw_floor=False))

    return np.stack(frames, axis=0), init_position, init_velocity


def generate_dataset(
    num_scenarios=150,
    values_per_scenario=5,
    property_range=(0.3, 4.0),
    output_dir="data/generated6.0/drag",
    dataset_seed=1337,
):
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(dataset_seed)
    num_videos = num_scenarios * values_per_scenario
    width, height = 224, 224
    radius = 15
    margin = radius + 6.0

    scenarios = []
    for _ in range(num_scenarios):
        phi = float(rng.uniform(0, 2 * np.pi))
        v0 = float(rng.uniform(60.0, 450.0))
        scenarios.append(
            {
                "x": float(rng.uniform(margin, width - margin)),
                "y": float(rng.uniform(margin, height - margin)),
                "vx": float(v0 * np.cos(phi)),
                "vy": float(v0 * np.sin(phi)),
            }
        )

    all_frames = np.empty((num_videos, 64, height, width, 3), dtype=np.uint8)
    labels = np.empty(num_videos, dtype=np.float32)
    scenario_ids = np.empty(num_videos, dtype=np.int32)
    init_positions = np.empty((num_videos, 2), dtype=np.float32)
    init_velocities = np.empty((num_videos, 2), dtype=np.float32)
    manifest_rows = []
    video_idx = 0
    for scenario_idx, ic in enumerate(scenarios):
        edges = np.linspace(property_range[0], property_range[1], values_per_scenario + 1)
        prop_values = rng.uniform(edges[:-1], edges[1:])
        for val in prop_values:
            frames, init_pos, init_vel = generate_single_video(float(val), ic)
            all_frames[video_idx] = frames
            labels[video_idx] = float(val)
            scenario_ids[video_idx] = scenario_idx
            init_positions[video_idx] = init_pos
            init_velocities[video_idx] = init_vel
            manifest_rows.append(
                {
                    "filename": f"dataset.npz[{video_idx}]",
                    "property_value": float(val),
                    "scenario_id": scenario_idx,
                }
            )
            video_idx += 1
            if video_idx % 50 == 0:
                print(f"Generated {video_idx}/{num_videos} videos")

    shuffle_idx = rng.permutation(num_videos)
    all_frames = all_frames[shuffle_idx]
    labels = labels[shuffle_idx]
    scenario_ids = scenario_ids[shuffle_idx]
    init_positions = init_positions[shuffle_idx]
    init_velocities = init_velocities[shuffle_idx]

    out_path = os.path.join(output_dir, "dataset.npz")
    np.savez_compressed(
        out_path,
        frames=all_frames,
        drag_coefficients=labels,
        initial_positions=init_positions,
        initial_velocities=init_velocities,
        scenario_id=scenario_ids,
        dataset_seed=dataset_seed,
        num_scenarios=num_scenarios,
        values_per_scenario=values_per_scenario,
        design="matched_initial_conditions_v4",
        mass=1.0,
        radius=radius,
        gravity_y=0.0,
        num_frames=64,
        width=width,
        height=height,
    )

    manifest_path = os.path.join(output_dir, "manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "property_value", "scenario_id"])
        writer.writeheader()
        writer.writerows(manifest_rows)


if __name__ == "__main__":
    generate_dataset()
