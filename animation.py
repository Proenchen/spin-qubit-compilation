from matplotlib.animation import FuncAnimation
import networkx as nx
import matplotlib.pyplot as plt

NODE_SIZE = 120

def _build_time_indexed_positions(path):
    """
    Turn a path like [(n0,t0), (n1,t1), ...] into a dict t -> position,
    holding position once agent stops moving.
    """
    if not path:
        return {}
    t_min = path[0][1]
    t_max = path[-1][1]
    per_t = {}
    # For each integer t in [t_min, t_max], assign the node occupied at that t
    i = 0
    for t in range(t_min, t_max + 1):
        # advance along path while next entry's time <= t
        while i + 1 < len(path) and path[i + 1][1] <= t:
            i += 1
        per_t[t] = path[i][0]
    return per_t, t_min, t_max

def _interpolate(a, b, alpha):
    """Linear interpolate between 2D points a, b with alpha in [0,1]."""
    return (a[0] + (b[0] - a[0]) * alpha, a[1] + (b[1] - a[1]) * alpha)

def _make_smooth_positions(path, substeps=5):
    """
    Build a list of positions including substeps between integer times to get smooth motion.
    Returns: positions (list of (x,y)), t_frames (list of float timeline frames), last_index (int)
    """
    if not path:
        return [], [], -1
    # Ensure we have position for each integer t (hold at last)
    per_t, t0, t1 = _build_time_indexed_positions(path)
    # Build fine-grained frames
    positions = []
    frames = []
    for t in range(t0, t1):
        p0 = per_t[t]
        p1 = per_t[t + 1]
        for s in range(substeps):
            alpha = s / float(substeps)
            positions.append(_interpolate(p0, p1, alpha))
            frames.append(t + alpha)
    # include final exact position at t1
    positions.append(per_t[t1])
    frames.append(float(t1))
    return positions, frames, len(positions) - 1

def _make_step_positions(path):
    """
    Stepwise (jump-at-integer-time) positions, one frame per integer t.
    """
    if not path:
        return [], [], -1
    per_t, t0, t1 = _build_time_indexed_positions(path)
    frames = list(range(t0, t1 + 1))
    positions = [per_t[t] for t in frames]
    return positions, frames, len(positions) - 1

def animate_mapf(G, plans, interval_ms=300, smooth=True, substeps=3):
    """
    Animate moving qubits over the grid, ohne Start- und Endmarker.
    """
    # --- prepare background drawing (grid + node types) ---
    pos = {n: n for n in G}
    fig, ax = plt.subplots(figsize=(8, 8))
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3)
    nx.draw_networkx_nodes(
        G, pos, nodelist=[n for n, d in G.nodes(data=True) if d["typ"] == "IN"],
        node_shape="s", node_size=NODE_SIZE, ax=ax
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=[n for n, d in G.nodes(data=True) if d["typ"] == "SN"],
        node_shape="o", node_size=NODE_SIZE, ax=ax
    )
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()

    # --- colors and agent ordering (stable across runs) ---
    colors = ["tab:red","tab:green","tab:orange","tab:purple",
              "tab:brown","tab:pink","tab:gray","tab:olive","tab:cyan"]
    agent_ids = list(plans.keys())

    # --- build per-agent timelines ---
    agent_data = []
    global_frames = set()
    for idx, aid in enumerate(agent_ids):
        path = plans[aid]
        if smooth:
            positions, frames, last_idx = _make_smooth_positions(path, substeps=substeps)
        else:
            positions, frames, last_idx = _make_step_positions(path)

        agent_data.append({
            "aid": aid,
            "color": colors[idx % len(colors)],
            "positions": positions,
            "frames": frames,
            "last_idx": last_idx,
        })
        global_frames.update(frames)

    global_frames = sorted(global_frames)

    for data in agent_data:
        frame_to_pos = dict(zip(data["frames"], data["positions"]))
        data["frame_to_pos"] = frame_to_pos
        data["first_frame"] = data["frames"][0] if data["frames"] else None
        data["last_frame"]  = data["frames"][-1] if data["frames"] else None

    # --- only moving artists (no start/end markers) ---
    moving_artists = []
    for data in agent_data:
        col = data["color"]
        p0 = data["positions"][0] if data["frames"] else (0, 0)
        dot, = ax.plot([p0[0]], [p0[1]], marker="o", markersize=8,
                       markeredgecolor="black", markeredgewidth=1.2,
                       linestyle="None", color=col, alpha=0.95, zorder=4)
        moving_artists.append(dot)

    def get_agent_pos_for_frame(data, gf):
        if gf in data["frame_to_pos"]:
            return data["frame_to_pos"][gf]
        if data["first_frame"] is not None and gf < data["first_frame"]:
            return data["positions"][0]
        if data["last_frame"] is not None and gf > data["last_frame"]:
            return data["positions"][-1]
        prev = max([f for f in data["frames"] if f <= gf], default=data["frames"][0])
        return data["frame_to_pos"][prev]

    def update(frame_idx):
        gf = global_frames[frame_idx]
        artists = []
        for i, data in enumerate(agent_data):
            x, y = get_agent_pos_for_frame(data, gf)
            moving_artists[i].set_data([x], [y])
            artists.append(moving_artists[i])
        return artists

    anim = FuncAnimation(
        fig, update, frames=len(global_frames), interval=interval_ms,
        blit=True, repeat=True
    )

    plt.show()
    return anim
