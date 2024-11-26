import random
import numpy as np
import Utils.occ_utils as occ_utils

from Features.machining_features import MachiningFeature


class SlantedThroughStep(MachiningFeature):
    def __init__(self, shape, label_map, min_len, clearance, feat_names):
        super().__init__(shape, label_map, min_len, clearance, feat_names)
        self.shifter_type = 3
        self.bound_type = 3
        self.depth_type = "blind"
        self.feat_type = "slanted_through_step"

    def _add_sketch(self, bound, hetero):
        dir_l = bound[0] - bound[1]
        dir_r = bound[3] - bound[2]

        mark = [0, 1]
        random.shuffle(mark)
        ratio = random.uniform(0.3, 0.6)
        pt0 = bound[0] - dir_l * mark[0] * ratio
        pt1 = bound[1]
        pt2 = bound[2]
        pt3 = bound[3] - dir_r * mark[1] * ratio

        if hetero:
            scale = random.uniform(0.1, 0.9)
            pt4 = np.array([scale * x + (1 - scale) * y for x, y in zip(pt0, pt3)])

            return occ_utils.face_polygon([pt0, pt1, pt2, pt3, pt4])

        else:
            return occ_utils.face_polygon([pt0, pt1, pt2, pt3])
