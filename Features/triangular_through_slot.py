import random
import numpy as np
import Utils.occ_utils as occ_utils

from Features.machining_features import MachiningFeature


class TriangularThroughSlot(MachiningFeature):
    def __init__(self, shape, label_map, min_len, clearance, feat_names):
        super().__init__(shape, label_map, min_len, clearance, feat_names)
        self.shifter_type = 1
        self.bound_type = 1
        self.depth_type = "through"
        self.feat_type = "triangular_through_slot"

    def _add_sketch(self, bound, hetero):
        # edge_dir = bound[2] - bound[1]

        # normal = np.cross(edge_dir, bound[0] - bound[1])
        # edge_normal = np.cross(normal, edge_dir)
        # edge_normal = edge_normal / np.linalg.norm(edge_normal)  # TODO(ANDREW): Why is the edge_normal not used?
        pnt3 = (bound[0] + bound[1] + bound[2] + bound[3]) / 4

        if hetero:
            scale = random.uniform(0.1, 0.9)
            selected_pt = random.choice([bound[1], bound[2]])
            pnt4 = np.array([scale * x + (1 - scale) * y for x, y in zip(selected_pt, pnt3)])
            if np.array_equal(selected_pt, bound[1]):
                return occ_utils.face_polygon([bound[1], bound[2], pnt3, pnt4])
            else:
                return occ_utils.face_polygon([bound[1], bound[2], pnt4, pnt3])

        else:
            return occ_utils.face_polygon([bound[1], bound[2], pnt3])
