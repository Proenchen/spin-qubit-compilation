import networkx as nx

class NetworkBuilder:
    """Builds the 2x2 tiled 8-neighborhood graph with typed nodes (IN/SN)."""

    @staticmethod
    def build_network() -> nx.Graph:
        """
        Create a 2x2 tiling of 8-node patterns (no center),
        connect with 8-neighborhood edges. Node attribute 'typ' ∈ {'IN','SN'}.
        Corners are 'IN'; others are 'SN'.
        """
        template_pos = {
            0: (-1,  1), 1: (0, 1),  2: (1, 1),
            3: (-1,  0),             4: (1, 0),
            5: (-1, -1), 6: (0, -1), 7: (1, -1),
        }
        corners = {0, 2, 5, 7}
        tile_offsets = [(0, 0), (2, 0), (0, -2), (2, -2)]  

        G = nx.Graph()
        for dx, dy in tile_offsets:
            for t_id, (x, y) in template_pos.items():
                coord = (x + dx, y + dy)
                if coord not in G:
                    G.add_node(coord, type=("IN" if t_id in corners else "SN"))

        coords = list(G.nodes())
        coord_set = set(coords)
        for (x, y) in coords:
            for ddx in (-1, 0, 1):
                for ddy in (-1, 0, 1):
                    if ddx == 0 and ddy == 0:
                        continue
                    v = (x + ddx, y + ddy)
                    if v in coord_set:
                        G.add_edge((x, y), v)
        return G