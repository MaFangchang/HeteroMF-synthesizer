import random
import math
import numpy as np
import Utils.occ_utils as occ_utils

from OCC.Core.gp import gp_Ax2, gp_Pnt, gp_Dir
from OCC.Core.Geom import Geom_Circle
from Features.machining_features import MachiningFeature


class SixSidesPassage(MachiningFeature):
    def __init__(self, shape, label_map, min_len, clearance, feat_names):
        super().__init__(shape, label_map, min_len, clearance, feat_names)
        self.shifter_type = 4
        self.bound_type = 4
        self.depth_type = "through"
        self.feat_type = "6sides_passage"

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

        ang1 = random.uniform(0.0, math.pi / 3)
        pt1 = occ_utils.as_list(circ.Value(ang1))

        ang2 = ang1 + math.pi / 3
        pt2 = occ_utils.as_list(circ.Value(ang2))

        ang3 = ang2 + math.pi / 3
        pt3 = occ_utils.as_list(circ.Value(ang3))

        ang4 = ang3 + math.pi / 3
        pt4 = occ_utils.as_list(circ.Value(ang4))

        ang5 = ang4 + math.pi / 3
        pt5 = occ_utils.as_list(circ.Value(ang5))

        ang6 = ang5 + math.pi / 3
        pt6 = occ_utils.as_list(circ.Value(ang6))

        if hetero:
            scale = random.uniform(0.1, 0.9)
            num = random.randint(0, 5)
            if num == 0:
                pnt = [scale * x + (1 - scale) * y for x, y in zip(pt1, pt2)]
                return occ_utils.face_polygon([pt1, pnt, pt2, pt3, pt4, pt5, pt6])
            elif num == 1:
                pnt = [scale * x + (1 - scale) * y for x, y in zip(pt2, pt3)]
                return occ_utils.face_polygon([pt1, pt2, pnt, pt3, pt4, pt5, pt6])
            elif num == 2:
                pnt = [scale * x + (1 - scale) * y for x, y in zip(pt3, pt4)]
                return occ_utils.face_polygon([pt1, pt2, pt3, pnt, pt4, pt5, pt6])
            elif num == 3:
                pnt = [scale * x + (1 - scale) * y for x, y in zip(pt4, pt5)]
                return occ_utils.face_polygon([pt1, pt2, pt3, pt4, pnt, pt5, pt6])
            elif num == 4:
                pnt = [scale * x + (1 - scale) * y for x, y in zip(pt5, pt6)]
                return occ_utils.face_polygon([pt1, pt2, pt3, pt4, pt5, pnt, pt6])
            else:
                pnt = [scale * x + (1 - scale) * y for x, y in zip(pt6, pt1)]
                return occ_utils.face_polygon([pt1, pt2, pt3, pt4, pt5, pt6, pnt])

        else:
            return occ_utils.face_polygon([pt1, pt2, pt3, pt4, pt5, pt6])
