import random
import math
import numpy as np
import Utils.occ_utils as occ_utils

from OCC.Core.gp import gp_Ax2, gp_Pnt, gp_Dir
from OCC.Core.Geom import Geom_Circle
from Features.machining_features import MachiningFeature


class TriangularPassage(MachiningFeature):
    def __init__(self, shape, label_map, min_len, clearance, feat_names):
        super().__init__(shape, label_map, min_len, clearance, feat_names)
        self.shifter_type = 4
        self.bound_type = 4
        self.depth_type = "through"
        self.feat_type = "triangular_passage"

    def _add_sketch(self, bound, hetero):
        dir_w = bound[2] - bound[1]
        dir_h = bound[0] - bound[1]
        width = np.linalg.norm(dir_w)
        height = np.linalg.norm(dir_h)

        dir_w = dir_w / width
        dir_h = dir_h / height
        normal = occ_utils.as_occ(np.cross(dir_w, dir_h), gp_Dir)

        radius = min(width / 2, height / 2)

        center = (bound[0] + bound[1] + bound[2] + bound[3]) / 4

        circ = Geom_Circle(gp_Ax2(gp_Pnt(center[0], center[1], center[2]), normal), radius)

        ang1 = random.uniform(0.0, 2 * math.pi / 3)
        pt1 = occ_utils.as_list(circ.Value(ang1))

        ang2 = ang1 + random.uniform(2 * math.pi / 3 - math.pi / 9, 2 * math.pi / 3 + math.pi / 9)
        if ang2 > 2 * math.pi:
            ang2 = ang2 - 2 * math.pi
        pt2 = occ_utils.as_list(circ.Value(ang2))

        ang3 = ang2 + random.uniform(2 * math.pi / 3 - math.pi / 9, 2 * math.pi / 3 + math.pi / 9)
        if ang3 > 2 * math.pi:
            ang3 = ang3 - 2 * math.pi
        pt3 = occ_utils.as_list(circ.Value(ang3))

        if hetero:
            # Split a face of the triangular passage
            original_list = [pt1, pt2, pt3]
            selected_items = random.sample(original_list, 2)
            unselected_items = [item for item in original_list if item not in selected_items]
            pnt1 = selected_items[0]
            scale = random.uniform(0.1, 0.9)
            pnt2 = [scale * x + (1 - scale) * y for x, y in zip(selected_items[0], selected_items[1])]
            pnt3 = selected_items[1]
            pnt4 = unselected_items[0]

            return occ_utils.face_polygon([pnt1, pnt2, pnt3, pnt4])

        else:
            return occ_utils.face_polygon([pt1, pt2, pt3])
