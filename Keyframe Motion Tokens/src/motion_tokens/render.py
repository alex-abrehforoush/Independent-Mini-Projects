"""Stick-figure rendering of global joint positions (matplotlib -> GIF/PNG)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

KEY_COLOR = "#d1495b"
COLORS = ["#2e4057", "#00798c", "#edae49", "#66a182", "#8d96a3"]


def _bones(parents):
    return [(j, p) for j, p in enumerate(parents) if p >= 0]


def animate(clips: list[np.ndarray], titles: list[str], parents, path: str, fps: int = 30,
            key_frames: list[set] | None = None, highlight: tuple[int, int] | None = None):
    """clips: list of (T,J,3) arrays in cm (y up), drawn side by side, shared camera.
    key_frames[i]: frames to flag (e.g. generated keys); highlight: (start, end) of the transition."""
    T = min(len(c) for c in clips)
    allp = np.concatenate([c[:T] for c in clips])
    cx, cz = allp[..., 0].mean(), allp[..., 2].mean()
    span = max(np.ptp(allp[..., 0]), np.ptp(allp[..., 2]), np.ptp(allp[..., 1])) / 2 + 20
    fig = plt.figure(figsize=(3.2 * len(clips), 3.4), dpi=80)
    axes = [fig.add_subplot(1, len(clips), i + 1, projection="3d") for i in range(len(clips))]
    bones = _bones(parents)
    lines = []
    for i, ax in enumerate(axes):
        ax.set_xlim(cx - span, cx + span)
        ax.set_ylim(cz - span, cz + span)
        ax.set_zlim(0, 2 * span)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=15, azim=-60)
        ax.set_axis_off()
        ax.set_title(titles[i], fontsize=9)
        lines.append([ax.plot([], [], [], lw=2, color=COLORS[i % len(COLORS)])[0] for _ in bones])
        # ground trajectory of the root
        c = clips[i][:T]
        ax.plot(c[:, 0, 0], c[:, 0, 2], np.zeros(T), lw=0.8, color="#bbbbbb")

    def update(t):
        for i, c in enumerate(clips):
            in_trans = highlight is not None and highlight[0] <= t < highlight[1]
            is_key = key_frames is not None and t in key_frames[i]
            col = KEY_COLOR if is_key else (COLORS[i % len(COLORS)] if in_trans else "#555555")
            for ln, (j, p) in zip(lines[i], bones):
                ln.set_data_3d([c[t, j, 0], c[t, p, 0]], [c[t, j, 2], c[t, p, 2]], [c[t, j, 1], c[t, p, 1]])
                ln.set_color(col)
        return [l for ls in lines for l in ls]

    anim = FuncAnimation(fig, update, frames=T, interval=1000 / fps, blit=False)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def filmstrip(clips: list[np.ndarray], titles: list[str], parents, path: str, frames: list[int],
              key_frames: list[set] | None = None):
    """Rows = clips, overlaid poses at the given frames (offset along x so they do not overlap)."""
    fig, axes = plt.subplots(len(clips), 1, figsize=(10, 1.9 * len(clips)), dpi=110)
    axes = np.atleast_1d(axes)
    bones = _bones(parents)
    for i, (ax, c) in enumerate(zip(axes, clips)):
        for k, t in enumerate(frames):
            p = c[t]
            dx = k * 70 - p[0, 0]
            col = KEY_COLOR if key_frames is not None and t in key_frames[i] else COLORS[i % len(COLORS)]
            for j, q in bones:
                ax.plot([p[j, 0] + dx, p[q, 0] + dx], [p[j, 1], p[q, 1]], color=col, lw=1.6)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(titles[i], fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
