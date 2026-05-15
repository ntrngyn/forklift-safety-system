import cv2
import numpy as np
import math

class GeometryUtils:
    def __init__(self, config_manager):
        self.cfg = config_manager
        self.matrix = None
        self.update_matrix()

    def update_matrix(self):
        self.matrix = cv2.getPerspectiveTransform(self.cfg.src_pts, self.cfg.dst_pts)

    def get_bottom_center(self, x1, y1, x2, y2):
        return (int((x1 + x2) / 2), int(y2))

    def get_bev_point(self, point):
        pts = np.array([[[point[0], point[1]]]], dtype=np.float32)
        warped_pt = cv2.perspectiveTransform(pts, self.matrix)
        return (int(warped_pt[0][0][0]), int(warped_pt[0][0][1]))

    def calculate_pixel_distance(self, pt1_bev, pt2_bev):
        return math.hypot(pt1_bev[0] - pt2_bev[0], pt1_bev[1] - pt2_bev[1])