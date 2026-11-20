# Input sketches

| File | Subject | Why it is a useful test case |
|---|---|---|
| `01_futuristic_tower.png` | Futuristic skyscraper with twin towers and sky bridge | Architecture: long straight lines, perspective-free elevation, repetitive facade grid |
| `02_robot_character.png` | Waving robot character | Character: articulated limbs, small facial details (eyes, visor), pose must survive |
| `03_hover_car.png` | Hover car, side view | Vehicle/product: smooth curves, one closed silhouette, hover pads under the body |
| `04_desk_lamp.png` | Articulated desk lamp | Product design: thin parts, joints, cone shape, light rays (non-object strokes) |
| `05_floating_island.png` | Floating island with cabin, trees, waterfall | Scene: many small objects at different scales, sky elements (clouds, birds) |
| `06_mug_photo.jpg` | Coffee mug, half full | Real pencil drawing on ruled notebook paper, phone photo with hand shadow, hatched shading |
| `07_car_photo.jpg` | Sedan car, side view | Real pencil drawing on ruled notebook paper, small in the frame, strong shadow |

## Provenance

**01–05** are drawn by [`scripts/make_sketches.py`](../scripts/make_sketches.py), a small "pen model"
that draws strokes with hand-like wobble, overshoot, double lines and pressure variation on a textured,
unevenly lit paper background (like a phone photo of a pencil sketch). They can be regenerated exactly with:

```bash
python scripts/make_sketches.py --out sketches
```

**06–07** are my own pencil drawings on a notebook page, photographed with a phone. The only change is
that the mug photo was rotated by 90° so the mug stands upright. The ruled lines of the paper are removed
by the preprocessing (`remove_ruled_lines` in `configs/experiments_handdrawn.yaml`).

No external images were used. To add a new sketch, put the photo in this folder and add a subject line
for it under `sketches:` in a config file.
