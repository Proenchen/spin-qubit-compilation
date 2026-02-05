import networkx as nx


class NetworkBuilder:

    @staticmethod
    def build_network(width: int, height: int) -> nx.Graph:
        if not (isinstance(width, int) and isinstance(height, int) and width >= 1 and height >= 1):
            raise ValueError("width and height must be integers >= 1")

        # Single-tile template (no center)
        template_pos = {
            0: (-1,  1), 1: (0, 1),  2: (1, 1),
            3: (-1,  0),             4: (1, 0),
            5: (-1, -1), 6: (0, -1), 7: (1, -1),
        }
        corners = {0, 2, 5, 7}

        G = nx.Graph()

        # Place tiles in a width x height grid
        for j in range(height):         
            for i in range(width):      
                dx, dy = 2 * i, -2 * j  
                for t_id, (x, y) in template_pos.items():
                    coord = (x + dx, y + dy)
                    if coord not in G:
                        G.add_node(coord, type=("IN" if t_id in corners else "SN"))

        # Connect 8-neighborhood across the entire tiled grid
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
