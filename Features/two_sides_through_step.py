import random
import numpy as np
import Utils.occ_utils as occ_utils

from Features.machining_features import MachiningFeature


class TwoSidesThroughStep(MachiningFeature):
    def __init__(self, shape, label_map, min_len, clearance, feat_names):
        super().__init__(shape, label_map, min_len, clearance, feat_names)
        self.shifter_type = 3
        self.bound_type = 3
        self.depth_type = "blind"
        self.feat_type = "2sides_through_step"

    def _add_sketch(self, bound, hetero):
        dir_l = bound[0] - bound[1]
        dir_r = bound[3] - bound[2]

        ratio = random.uniform(0.4, 0.8)
        pt4 = (bound[0] + bound[3]) / 2
        pt0 = bound[1] + dir_l * ratio
        pt1 = bound[1]
        pt2 = bound[2]
        pt3 = bound[2] + dir_r * ratio

        if hetero:
            scale = random.uniform(0.1, 0.9)
            selected_pt = random.choice([pt0, pt3])
            pt5 = np.array([scale * x + (1 - scale) * y for x, y in zip(selected_pt, pt4)])
            if np.array_equal(selected_pt, pt0):
                return occ_utils.face_polygon([pt0, pt1, pt2, pt3, pt4, pt5])
            else:
                return occ_utils.face_polygon([pt0, pt1, pt2, pt3, pt5, pt4])

        else:
            return occ_utils.face_polygon([pt0, pt1, pt2, pt3, pt4])
