import random
import numpy as np

import Utils.occ_utils as occ_utils
from Features.machining_features import MachiningFeature


class TriangularBlindStep(MachiningFeature):
    def __init__(self, shape, label_map, min_len, clearance, feat_names):
        super().__init__(shape, label_map, min_len, clearance, feat_names)
        self.shifter_type = 2
        self.bound_type = 2
        self.depth_type = "blind"
        self.feat_type = "triangular_blind_step"

    def _add_sketch(self, bound, hetero):
        if hetero:
            scale = random.uniform(0.1, 0.9)
            pt = np.array([scale * x + (1 - scale) * y for x, y in zip(bound[0], bound[2])])

            return occ_utils.face_polygon([bound[0], bound[1], bound[2], pt])

        else:
            return occ_utils.face_polygon([bound[0], bound[1], bound[2]])
