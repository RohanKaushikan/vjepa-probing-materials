import csv
import os

import numpy as np
import pymunk

from render_utils import render_frame


def generate_single_video(mass_ratio_m, initial_conditions, num_frames=64, width=224, height=224):
    space = pymunk.Space()
    space.gravity = (0, 0)

    radius = 15
    mass1 = float(mass_ratio_m)
    mass2 = 1.0 / mass1

    c = float(initial_conditions["cos_phi"])
    s = float(initial_conditions["sin_phi"])
    v0 = float(initial_conditions["v0"])
    d = float(initial_conditions["d"])
    cx, cy = width / 2.0, height / 2.0

    body1 = pymunk.Body(mass1, pymunk.moment_for_circle(mass1, 0, radius))
    body1.position = (cx - d * c, cy - d * s)
    body1.velocity = (v0 * c, v0 * s)

    body2 = pymunk.Body(mass2, pymunk.moment_for_circle(mass2, 0, radius))
    body2.position = (cx + d * c, cy + d * s)
    body2.velocity = (-v0 * c, -v0 * s)

    shape1 = pymunk.Circle(body1, radius)
    shape1.friction = 0.0
    shape1.elasticity = 1.0

    shape2 = pymunk.Circle(body2, radius)
    shape2.friction = 0.0
    shape2.elasticity = 1.0

    space.add(body1, shape1, body2, shape2)

    init_p1 = (float(body1.position.x), float(body1.position.y))
    init_p2 = (float(body2.position.x), float(body2.position.y))
    init_v1 = (float(body1.velocity.x), float(body1.velocity.y))
    init_v2 = (float(body2.velocity.x), float(body2.velocity.y))

    frames = []
    x1, y1 = body1.position
    x2, y2 = body2.position
    frames.append(
        render_frame(
            x1,
            y1,
            radius,
            width=width,
            height=height,
            ball2_x=x2,
            ball2_y=y2,
            ball2_radius=radius,
            ball2_color=(255, 255, 255),
            draw_floor=False,
        )
    )
    for _ in range(num_frames - 1):
        for _ in range(4):
            space.step(1 / 240)
        x1, y1 = body1.position
        x2, y2 = body2.position
        frames.append(
            render_frame(
                x1,
                y1,
                radius,
                width=width,
                height=height,
                ball2_x=x2,
                ball2_y=y2,
                ball2_radius=radius,
                draw_floor=False,
            )
        )

    return (
        np.stack(frames, axis=0),
        init_p1,
        init_v1,
        init_p2,
        init_v2,
    )


def generate_dataset(
    num_scenarios=150,
    values_per_scenario=5,
    property_range=(0.25, 4.0),
    output_dir="data/generated6.0/mass",
    dataset_seed=1337,
):
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(dataset_seed)
    num_videos = num_scenarios * values_per_scenario
    radius = 15

    width, height = 224, 224
    all_frames = np.empty((num_videos, 64, height, width, 3), dtype=np.uint8)
    labels = np.empty(num_videos, dtype=np.float32)
    scenario_ids = np.empty(num_videos, dtype=np.int32)
    init_positions_ball1 = np.empty((num_videos, 2), dtype=np.float32)
    init_velocities_ball1 = np.empty((num_videos, 2), dtype=np.float32)
    init_positions_ball2 = np.empty((num_videos, 2), dtype=np.float32)
    init_velocities_ball2 = np.empty((num_videos, 2), dtype=np.float32)
    manifest_rows = []
    video_idx = 0

    scenarios = []
    for _ in range(num_scenarios):
        phi = float(rng.uniform(0, 2 * np.pi))
        scenarios.append(
            {
                "cos_phi": float(np.cos(phi)),
                "sin_phi": float(np.sin(phi)),
                "v0": float(rng.uniform(80.0, 350.0)),
                "d": float(rng.uniform(2.6 * radius, 5.5 * radius)),
            }
        )

    for scenario_idx, ic in enumerate(scenarios):
        edges = np.exp(np.linspace(np.log(property_range[0]), np.log(property_range[1]), values_per_scenario + 1))
        prop_values = np.exp(rng.uniform(np.log(edges[:-1]), np.log(edges[1:])))
        for val in prop_values:
            frames, p1, v1, p2, v2 = generate_single_video(float(val), ic)
            all_frames[video_idx] = frames
            labels[video_idx] = float(val)
            scenario_ids[video_idx] = scenario_idx
            init_positions_ball1[video_idx] = p1
            init_velocities_ball1[video_idx] = v1
            init_positions_ball2[video_idx] = p2
            init_velocities_ball2[video_idx] = v2
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
    init_positions_ball1 = init_positions_ball1[shuffle_idx]
    init_velocities_ball1 = init_velocities_ball1[shuffle_idx]
    init_positions_ball2 = init_positions_ball2[shuffle_idx]
    init_velocities_ball2 = init_velocities_ball2[shuffle_idx]

    out_path = os.path.join(output_dir, "dataset.npz")
    np.savez_compressed(
        out_path,
        frames=all_frames,
        mass_ratio_m=labels,
        scenario_id=scenario_ids,
        initial_positions_ball1=init_positions_ball1,
        initial_velocities_ball1=init_velocities_ball1,
        initial_positions_ball2=init_positions_ball2,
        initial_velocities_ball2=init_velocities_ball2,
        dataset_seed=dataset_seed,
        num_scenarios=num_scenarios,
        values_per_scenario=values_per_scenario,
        design="matched_initial_conditions_v5",
        radius=radius,
        elasticity=1.0,
        friction=0.0,
        gravity_y=0.0,
        num_frames=64,
        width=224,
        height=224,
    )

    manifest_path = os.path.join(output_dir, "manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "property_value", "scenario_id"])
        writer.writeheader()
        writer.writerows(manifest_rows)


if __name__ == "__main__":
    generate_dataset()
