import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from langchain_core.tools import tool

OUT = Path("focus_work")
OUT.mkdir(exist_ok=True)

n_dmd, n_cam,noise  = 64, 32, 0.05
rng = np.random.default_rng(0)
T = (rng.normal(size=(n_cam**2, n_dmd**2)) +
     1j * rng.normal(size=(n_cam**2, n_dmd**2))) / np.sqrt(2 * n_dmd**2)

experiments = []


def shoot(pattern):
    I = np.abs(T @ pattern.ravel()) ** 2
    I = I * (1 + noise * rng.normal(size=I.shape))
    return I.reshape(n_cam,n_cam)


def make_regions(focus_radius, background_radius, n_regions):
    y, x = np.indices((n_cam, n_cam))
    dy, dx = y - n_cam // 2, x - n_cam// 2
    r = np.hypot(dx, dy)
    angle = np.arctan2(dy, dx) % (2 * np.pi)
    focus = r <= focus_radius
    ring = (r > focus_radius + 1) & (r <= background_radius)
    sector = (angle / (2 * np.pi) * n_regions).astype(int)
    background = [ring & (sector == k) for k in range(n_regions)]
    return focus, background


def enhancement(img, focus, background):
    return np.mean([img[focus].mean() / img[m].mean() for m in background])


@tool
def run_focusing(macropixel_size: int, focus_radius: int, background_radius: int,
                 n_regions: int, n_iterations: int, n_flips: int = 1) -> str:
    """Run one focusing experiment on the simulated bench (DMD 64x64, camera 32x32, focus at the camera centre).
    macropixel_size: DMD macropixel side in pixels, must divide 64 (1,2,4,8,16,32). Smaller = more macropixels = higher enhancement but slower.
    focus_radius: radius of the focus disk in camera pixels (1 to 5).
    background_radius: outer radius of the background ring in camera pixels (focus_radius+3 up to 15).
    n_regions: number of background sectors (2 to 12).
    n_iterations: number of flip-and-measure steps (10 to 20000).
    n_flips: macropixels flipped per step (1 to 10).
    Returns the final enhancement and the saved image path."""
    mp, n_iter, k, nr = int(macropixel_size), int(n_iterations), int(n_flips), int(n_regions)
    fr, br = int(focus_radius), int(background_radius)
    if n_dmd % mp != 0:
        return "ERROR: macropixel_size must divide 64."
    if not (1 <= n_iter <= 20000):
        return "ERROR: n_iterations must be between 1 and 20000."
    focus, bg = make_regions(fr, br, nr)
    if focus.sum() == 0 or any(m.sum() == 0 for m in bg):
        return "ERROR: a region is empty. Use a larger background_radius or fewer regions."

    g = n_dmd // mp
    grid = rng.integers(0, 2, size=(g, g))

    def pattern():
        return np.kron(grid, np.ones((mp, mp)))

    start_img = shoot(pattern())
    best = enhancement(start_img, focus, bg)
    for _ in range(n_iter):
        old = grid.copy()
        idx = rng.choice(g * g, size=min(k, g * g), replace=False)
        grid.flat[idx] = 1 - grid.flat[idx]
        e = enhancement(shoot(pattern()), focus, bg)
        if e > best:
            best = e
        else:
            grid = old

    imgs = [shoot(pattern()) for _ in range(5)]
    final = float(np.mean([enhancement(im, focus, bg) for im in imgs]))

    run_id = len(experiments) + 1
    path = OUT / f"focus_run_{run_id}.png"
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(start_img, cmap="inferno"); ax[0].set_title("random pattern")
    ax[1].imshow(imgs[0], cmap="inferno"); ax[1].set_title(f"focused, enhancement {final:.1f}")
    for a in ax:
        a.axis("off")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    experiments.append(dict(run=run_id, macropixel=mp, focus_r=fr, bg_r=br, regions=nr,
                            iters=n_iter, flips=k, enhancement=round(final, 1), image=str(path)))
    (OUT / "experiments.json").write_text(json.dumps(experiments, indent=1))
    return f"run {run_id}: enhancement = {final:.1f}, image saved to {path}"