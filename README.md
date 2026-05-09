# vjepa-probe

Synthetic physics video dataset for probing physical understanding in vision models (V-JEPA and similar).

Videos are generated using 2D rigid-body simulation (pymunk) and rendered to short clips with labeled physical parameters.

## Physics scenarios

- **Friction** — ball rolling on surfaces with varying friction coefficients
- **Elasticity** — bouncing balls with varying restitution
- **Drag** — projectiles under varying air resistance
- **Mass** — objects with varying mass under identical forces

## Usage

Generate and preview a scenario:

```bash
python data/generate_and_preview.py --scenario friction
```

Available scenarios: `friction`, `elasticity`, `drag`, `mass`

View generated clips:

```bash
python data/view_clips.py
```

## Requirements

See `requirements.txt`.
