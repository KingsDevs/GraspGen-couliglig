import torch
import numpy as np
import trimesh
import trimesh.transformations as tra
from grasp_gen.robot import load_control_points_core, load_default_gripper_config
from pathlib import Path


class GripperModel(object):
    """Assembles end_effector + two fingers from the Couliglig URDF.

    Mesh is returned in the URDF's native frame. The rotation to GraspGen
    convention is applied afterwards via
    transform_offset_from_asset_to_graspgen_convention in the YAML.
    """

    def __init__(self, data_root_dir=None):
        if data_root_dir is None:
            data_root_dir = (
                f"{Path(__file__).parent.parent.parent}/urdf/gripper/assets"
            )

        ee_mesh = trimesh.load(f"{data_root_dir}/end_effector.stl")
        finger_mesh = trimesh.load(f"{data_root_dir}/finger.stl")

        # Visual transform inside each finger link (from URDF)
        T_vis = tra.compose_matrix(
            translate=[-0.01035, 0.0058, -0.0309],
            angles=[-1.5708, -1.5708, 0],
        )

        # Joint origins from end_effector to each finger link (from URDF)
        T_j_l = tra.compose_matrix(
            translate=[-0.0720576, -0.0248088, 0.0172625],
            angles=[1.5708, 1.5708, 0],
        )
        T_j_r = tra.compose_matrix(
            translate=[-0.0720576, 0.0253912, 0.0104625],
            angles=[-1.5708, -1.5708, 0],
        )

        self.finger_l = finger_mesh.copy()
        self.finger_l.apply_transform(T_j_l @ T_vis)

        self.finger_r = finger_mesh.copy()
        self.finger_r.apply_transform(T_j_r @ T_vis)

        self.base = ee_mesh
        self.mesh = trimesh.util.concatenate(
            [self.base, self.finger_l, self.finger_r]
        )

    def get_gripper_collision_mesh(self):
        return self.mesh.copy()

    def get_gripper_visual_mesh(self):
        return self.mesh.copy()


def get_gripper_offset_bins():
    """Offset bins spanning [0, width]. Used by M2T2 only.

    The width here must stay in sync with the YAML 'width' field.
    """
    width = 0.05
    offset_bins = list(np.linspace(0.0, width, 11))

    offset_bin_weights = [
        0.16652107, 0.21488856, 0.37031708, 0.55618503, 0.75124664,
        0.93943357, 1.07824539, 1.19423112, 1.55731375, 3.17161779,
    ]
    return offset_bins, offset_bin_weights


def load_control_points() -> torch.Tensor:
    """
    Load the control points for the gripper, used for training.
    Returns a tensor of shape (4, N) where N is the number of control points.
    """
    gripper_config = load_default_gripper_config(Path(__file__).stem)
    control_points = load_control_points_core(gripper_config)
    control_points = np.vstack([control_points, np.zeros(3)])
    control_points = np.hstack([control_points, np.ones([len(control_points), 1])])
    control_points = torch.from_numpy(control_points).float()
    return control_points.T


def load_control_points_for_visualization():

    gripper_config = load_default_gripper_config(Path(__file__).stem)

    control_points = load_control_points_core(gripper_config)

    mid_point = (control_points[0] + control_points[1]) / 2

    control_points = [
        control_points[-2], control_points[0], mid_point,
        [0, 0, 0], mid_point, control_points[1], control_points[-1]
    ]
    return [control_points, ]
