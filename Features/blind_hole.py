import random
import math
import numpy as np
import Utils.occ_utils as occ_utils

from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace
from OCC.Core.gp import gp_Circ, gp_Ax2, gp_Pnt, gp_Dir
from Features.machining_features import MachiningFeature


class BlindHole(MachiningFeature):
    def __init__(self, shape, label_map, min_len, clearance, feat_names):
        super().__init__(shape, label_map, min_len, clearance, feat_names)
        self.shifter_type = 4
        self.bound_type = 4
        self.depth_type = "blind"
        self.feat_type = "blind_hole"

    def _add_sketch(self, bound, hetero):
        dir_w = bound[2] - bound[1]
        dir_h = bound[0] - bound[1]
        width = np.linalg.norm(dir_w)
        height = np.linalg.norm(dir_h)

        dir_w = dir_w / width
        dir_h = dir_h / height
        normal = np.cross(dir_w, dir_h)

        radius = min(width / 2, height / 2)

        center = (bound[0] + bound[1] + bound[2] + bound[3]) / 4

        if hetero:
            # Generate a hole composed of two semi-cylindrical surfaces
            scale = random.uniform(0.5, 1.5)
            circ = gp_Circ(gp_Ax2(gp_Pnt(center[0], center[1], center[2]), occ_utils.as_occ(normal, gp_Dir)), radius)
            edge1 = BRepBuilderAPI_MakeEdge(circ, 0., scale * math.pi).Edge()
            edge2 = BRepBuilderAPI_MakeEdge(circ, scale * math.pi, 2 * math.pi).Edge()
            outer_wire = BRepBuilderAPI_MakeWire(edge1, edge2).Wire()

        else:
            circ = gp_Circ(gp_Ax2(gp_Pnt(center[0], center[1], center[2]), occ_utils.as_occ(normal, gp_Dir)), radius)
            edge = BRepBuilderAPI_MakeEdge(circ, 0., 2 * math.pi).Edge()
            outer_wire = BRepBuilderAPI_MakeWire(edge).Wire()

        face_maker = BRepBuilderAPI_MakeFace(outer_wire)

        return face_maker.Face()
