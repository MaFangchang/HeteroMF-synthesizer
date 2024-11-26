import random
import numpy as np
import Utils.occ_utils as occ_utils

from Features.machining_features import MachiningFeature


class RectangularPocket(MachiningFeature):
    def __init__(self, shape, label_map, min_len, clearance, feat_names):
        super().__init__(shape, label_map, min_len, clearance, feat_names)
        self.shifter_type = 4
        self.bound_type = 4
        self.depth_type = "blind"
        self.feat_type = "rectangular_pocket"

    def _add_sketch(self, bound, hetero):
        if hetero:
            scale = random.uniform(0.1, 0.9)
            num = random.randint(0, 3)
            if num == 0:
                pt = np.array([scale * x + (1 - scale) * y for x, y in zip(bound[0], bound[1])])
                return occ_utils.face_polygon([bound[0], pt, bound[1], bound[2], bound[3]])
            elif num == 1:
                pt = np.array([scale * x + (1 - scale) * y for x, y in zip(bound[1], bound[2])])
                return occ_utils.face_polygon([bound[0], bound[1], pt, bound[2], bound[3]])
            elif num == 2:
                pt = np.array([scale * x + (1 - scale) * y for x, y in zip(bound[2], bound[3])])
                return occ_utils.face_polygon([bound[0], bound[1], bound[2], pt, bound[3]])
            else:
                pt = np.array([scale * x + (1 - scale) * y for x, y in zip(bound[3], bound[0])])
                return occ_utils.face_polygon([bound[0], bound[1], bound[2], bound[3], pt])

        else:
            return occ_utils.face_polygon(bound[:4])
