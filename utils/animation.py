from matplotlib.animation import FuncAnimation
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import math

NODE_SIZE = 120

def _build_time_indexed_positions(path):
    if not path:
        return {}, 0, 0
    t_min = path[0][1]
    t_max = path[-1][1]
    per_t = {}
    i = 0
    for t in range(t_min, t_max + 1):
        while i + 1 < len(path) and path[i + 1][1] <= t:
            i += 1
        per_t[t] = path[i][0]
    return per_t, t_min, t_max

def _interpolate(a, b, alpha):
    return (a[0] + (b[0] - a[0]) * alpha, a[1] + (b[1] - a[1]) * alpha)

def _make_smooth_positions(path, substeps=5):
    if not path:
        return [], [], -1
    per_t, t0, t1 = _build_time_indexed_positions(path)
    positions = []
    frames = []
    for t in range(t0, t1):
        p0 = per_t[t]
        p1 = per_t[t + 1]
        for s in range(substeps):
            alpha = s / float(substeps)
            positions.append(_interpolate(p0, p1, alpha))
            frames.append(t + alpha)
    positions.append(per_t[t1])
    frames.append(float(t1))
    return positions, frames, len(positions) - 1

def _make_step_positions(path):
    if not path:
        return [], [], -1
    per_t, t0, t1 = _build_time_indexed_positions(path)
    frames = list(range(t0, t1 + 1))
    positions = [per_t[t] for t in frames]
    return positions, frames, len(positions) - 1


def animate_mapf(
    G,
    plans,
    interval_ms=0.1,
    smooth=True,
    substeps=200,
    edge_timebands=None,           
    failed_edges_timeline=None,    
):
    pos = {n: n for n in G}

    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)

    all_segments = [(pos[u], pos[v]) for u, v in G.edges()]
    base_edges = LineCollection(all_segments, alpha=0.3, linewidths=1.0, zorder=1)
    ax.add_collection(base_edges)

    failed_lc = LineCollection([], colors="red", linewidths=2.2, alpha=0.9, zorder=3)
    ax.add_collection(failed_lc)

    in_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "IN"]
    sn_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "SN"]
    nx.draw_networkx_nodes(G, pos, nodelist=in_nodes, node_shape="s", node_size=NODE_SIZE, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=sn_nodes, node_shape="o", node_size=NODE_SIZE, ax=ax)

    ax.set_aspect("equal"); ax.axis("off")

    colors = ["tab:red","tab:green","tab:orange","tab:purple",
              "tab:brown","tab:pink","tab:gray","tab:olive","tab:cyan"]

    def _qid_sort_key(aid):
        s = str(aid)
        return (0, int(s)) if s.lstrip("-").isdigit() else (1, s)
    agent_ids = sorted(plans.keys(), key=_qid_sort_key)

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
            "frame_to_pos": dict(zip(frames, positions)),
            "first_frame": frames[0] if frames else None,
            "last_frame": frames[-1] if frames else None,
        })
        global_frames.update(frames)

    global_frames = sorted(global_frames)

    moving_artists = []
    legend_handles = []
    legend_labels = []
    for data in agent_data:
        col = data["color"]
        p0 = data["positions"][0] if data["positions"] else (0, 0)
        dot, = ax.plot([p0[0]], [p0[1]], marker="o", markersize=8,
                       markeredgecolor="black", markeredgewidth=1.2,
                       linestyle="None", color=col, alpha=0.95, zorder=4)
        moving_artists.append(dot)
        legend_handles.append(plt.Line2D([0],[0], marker="o", color="w",
                                         markerfacecolor=col, markeredgecolor="black",
                                         markersize=8))
        legend_labels.append(f"Qubit {data['aid']}")
    ax.legend(legend_handles, legend_labels, loc="center left", bbox_to_anchor=(1.00, 0.5), frameon=True)

    def get_agent_pos_for_frame(data, gf):
        if gf in data["frame_to_pos"]:
            return data["frame_to_pos"][gf]
        if data["first_frame"] is not None and gf < data["first_frame"]:
            return data["positions"][0]
        if data["last_frame"] is not None and gf > data["last_frame"]:
            return data["positions"][-1]
        prev = max([f for f in data["frames"] if f <= gf], default=data["frames"][0])
        return data["frame_to_pos"][prev]

    def defects_for_time(gf):
        if failed_edges_timeline is not None:
            tf = int(math.floor(gf + 1e-9)) 
            failed = failed_edges_timeline.get(tf, set())
        elif edge_timebands is not None:
            tf = int(math.floor(gf + 1e-9))
            failed = set()
            for (t0, t1, edges) in edge_timebands:
                if t0 <= tf < t1:
                    failed |= set(edges)
        else:
            return []

        segs = []
        for e in failed:
            u, v = tuple(e)
            segs.append((pos[u], pos[v]))
        return segs

    def update(frame_idx):
        gf = global_frames[frame_idx]
        failed_segments = defects_for_time(gf)
        failed_lc.set_segments(failed_segments)

        artists = [failed_lc]
        for i, data in enumerate(agent_data):
            x, y = get_agent_pos_for_frame(data, gf)
            moving_artists[i].set_data([x], [y])
            artists.append(moving_artists[i])
        return artists

    anim = FuncAnimation(fig, update, frames=len(global_frames), interval=interval_ms, blit=True, repeat=True)
    plt.show()
    return anim
