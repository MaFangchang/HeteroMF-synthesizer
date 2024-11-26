import random
import numpy as np
import Utils.occ_utils as occ_utils
from Features.machining_features import MachiningFeature


class RectangularThroughStep(MachiningFeature):
    def __init__(self, shape, label_map, min_len, clearance, feat_names):
        super().__init__(shape, label_map, min_len, clearance, feat_names)
        self.shifter_type = 3
        self.bound_type = 3
        self.depth_type = "blind"
        self.feat_type = "rectangular_through_step"

    def _add_sketch(self, bound, hetero):
        if hetero:
            scale = random.uniform(0.1, 0.9)
            pt = np.array([scale * x + (1 - scale) * y for x, y in zip(bound[3], bound[0])])
            return occ_utils.face_polygon([bound[0], bound[1], bound[2], bound[3], pt])

        else:
            return occ_utils.face_polygon(bound[:4])
