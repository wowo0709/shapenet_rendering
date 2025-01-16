import itertools
import json
import os
import zipfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple, BinaryIO, Union, Optional

import numpy as np
from PIL import Image


@dataclass
class Camera(ABC):
    """
    An object describing how a camera corresponds to pixels in an image.
    """

    @abstractmethod
    def image_coords(self) -> np.ndarray:
        """
        :return: ([self.height, self.width, 2]).reshape(self.height * self.width, 2) image coordinates
        """

    @abstractmethod
    def camera_rays(self, coords: np.ndarray) -> np.ndarray:
        """
        For every (x, y) coordinate in a rendered image, compute the ray of the
        corresponding pixel.

        :param coords: an [N x 2] integer array of 2D image coordinates.
        :return: an [N x 2 x 3] array of [2 x 3] (origin, direction) tuples.
                 The direction should always be unit length.
        """

    def depth_directions(self, coords: np.ndarray) -> np.ndarray:
        """
        For every (x, y) coordinate in a rendered image, get the direction that
        corresponds to "depth" in an RGBD rendering.

        This may raise an exception if there is no "D" channel in the
        corresponding ViewData.

        :param coords: an [N x 2] integer array of 2D image coordinates.
        :return: an [N x 3] array of normalized depth directions.
        """
        _ = coords
        raise NotImplementedError

    @abstractmethod
    def center_crop(self) -> "Camera":
        """
        Creates a new camera with the same intrinsics and direction as this one,
        but with a center crop to a square of the smaller dimension.
        """

    @abstractmethod
    def resize_image(self, width: int, height: int) -> "Camera":
        """
        Creates a new camera with the same intrinsics and direction as this one,
        but with resized image dimensions.
        """

    @abstractmethod
    def scale_scene(self, factor: float) -> "Camera":
        """
        Creates a new camera with the same intrinsics and direction as this one,
        but with the scene rescaled by the given factor.
        """


@dataclass
class ProjectiveCamera(Camera):
    """
    A Camera implementation for a standard pinhole camera.

    The camera rays shoot away from the origin in the z direction, with the x
    and y directions corresponding to the positive horizontal and vertical axes
    in image space.
    """

    origin: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    width: int
    height: int
    x_fov: float
    y_fov: float

    def image_coords(self) -> np.ndarray:
        ind = np.arange(self.width * self.height)
        coords = np.stack([ind % self.width, ind // self.width], axis=1).astype(np.float32)
        return coords

    def camera_rays(self, coords: np.ndarray) -> np.ndarray:
        fracs = (coords / (np.array([self.width, self.height], dtype=np.float32) - 1)) * 2 - 1
        fracs = fracs * np.tan(np.array([self.x_fov, self.y_fov]) / 2)
        directions = self.z + self.x * fracs[:, :1] + self.y * fracs[:, 1:]
        directions = directions / np.linalg.norm(directions, axis=-1, keepdims=True)
        return np.stack([np.broadcast_to(self.origin, directions.shape), directions], axis=1)

    def depth_directions(self, coords: np.ndarray) -> np.ndarray:
        return np.tile((self.z / np.linalg.norm(self.z))[None], [len(coords), 1])

    def resize_image(self, width: int, height: int) -> "ProjectiveCamera":
        """
        Creates a new camera for the resized view assuming the aspect ratio does not change.
        """
        assert width * self.height == height * self.width, "The aspect ratio should not change."
        return ProjectiveCamera(
            origin=self.origin,
            x=self.x,
            y=self.y,
            z=self.z,
            width=width,
            height=height,
            x_fov=self.x_fov,
            y_fov=self.y_fov,
        )

    def center_crop(self) -> "ProjectiveCamera":
        """
        Creates a new camera for the center-cropped view
        """
        size = min(self.width, self.height)
        fov = min(self.x_fov, self.y_fov)
        return ProjectiveCamera(
            origin=self.origin,
            x=self.x,
            y=self.y,
            z=self.z,
            width=size,
            height=size,
            x_fov=fov,
            y_fov=fov,
        )

    def scale_scene(self, factor: float) -> "ProjectiveCamera":
        """
        Creates a new camera with the same intrinsics and direction as this one,
        but with the camera frame rescaled by the given factor.
        """
        return ProjectiveCamera(
            origin=self.origin * factor,
            x=self.x,
            y=self.y,
            z=self.z,
            width=self.width,
            height=self.height,
            x_fov=self.x_fov,
            y_fov=self.y_fov,
        )


class ViewData(ABC):
    """
    A collection of rendered camera views of a scene or object.

    This is a generalization of a NeRF dataset, since NeRF datasets only encode
    RGB or RGBA data, whereas this dataset supports arbitrary channels.
    """

    @property
    @abstractmethod
    def num_views(self) -> int:
        """
        The number of rendered views.
        """

    @property
    @abstractmethod
    def channel_names(self) -> List[str]:
        """
        Get all of the supported channels available for the views.

        This can be arbitrary, but there are some standard names:
        "R", "G", "B", "A" (alpha), and "D" (depth).
        """

    @abstractmethod
    def load_view(self, index: int, channels: List[str]) -> Tuple[Camera, np.ndarray]:
        """
        Load the given channels from the view at the given index.

        :return: a tuple (camera_view, data), where data is a float array of
                 shape [height x width x num_channels].
        """


class MemoryViewData(ViewData):
    """
    A ViewData that is implemented in memory.
    """

    def __init__(self, channels: Dict[str, np.ndarray], cameras: List[Camera]):
        assert all(v.shape[0] == len(cameras) for v in channels.values())
        self.channels = channels
        self.cameras = cameras

    @property
    def num_views(self) -> int:
        return len(self.cameras)

    @property
    def channel_names(self) -> List[str]:
        return list(self.channels.keys())

    def load_view(self, index: int, channels: List[str]) -> Tuple[Camera, np.ndarray]:
        outputs = [self.channels[channel][index] for channel in channels]
        return self.cameras[index], np.stack(outputs, axis=-1)


class BlenderViewData(ViewData):
    """
    Interact with a dataset zipfile exported by view_data.py.
    """

    def __init__(self, f_obj: BinaryIO):
        self.zipfile = zipfile.ZipFile(f_obj, mode="r")
        self.infos = []
        with self.zipfile.open("info.json", "r") as f:
            self.info = json.load(f)
        self.channels = list(self.info.get("channels", "RGBAD"))
        assert set("RGBA").issubset(
            set(self.channels)
        ), "The blender output should at least have RGBA images."
        names = set(x.filename for x in self.zipfile.infolist())
        for i in itertools.count():
            name = f"{i:05}.json"
            if name not in names:
                break
            with self.zipfile.open(name, "r") as f:
                self.infos.append(json.load(f))

    @property
    def num_views(self) -> int:
        return len(self.infos)

    @property
    def channel_names(self) -> List[str]:
        return list(self.channels)

    def load_view(self, index: int, channels: List[str]) -> Tuple[Camera, np.ndarray]:
        for ch in channels:
            if ch not in self.channel_names:
                raise ValueError(f"unsupported channel: {ch}")

        # Gather (a superset of) the requested channels.
        channel_map = {}
        if any(x in channels for x in "RGBA"):
            with self.zipfile.open(f"{index:05}.png", "r") as f:
                rgba = np.array(Image.open(f)).astype(np.float32) / 255.0
                channel_map.update(zip("RGBA", rgba.transpose([2, 0, 1])))
        if "D" in channels:
            with self.zipfile.open(f"{index:05}_depth.png", "r") as f:
                # Decode a 16-bit fixed-point number.
                fp = np.array(Image.open(f))
                inf_dist = fp == 0xFFFF
                channel_map["D"] = np.where(
                    inf_dist,
                    np.inf,
                    self.infos[index]["max_depth"] * (fp.astype(np.float32) / 65536),
                )
        if "MatAlpha" in channels:
            with self.zipfile.open(f"{index:05}_MatAlpha.png", "r") as f:
                channel_map["MatAlpha"] = np.array(Image.open(f)).astype(np.float32) / 65536

        # The order of channels is user-specified.
        combined = np.stack([channel_map[k] for k in channels], axis=-1)

        h, w, _ = combined.shape
        return self.camera(index, w, h), combined

    def camera(self, index: int, width: int, height: int) -> ProjectiveCamera:
        info = self.infos[index]
        return ProjectiveCamera(
            origin=np.array(info["origin"], dtype=np.float32),
            x=np.array(info["x"], dtype=np.float32),
            y=np.array(info["y"], dtype=np.float32),
            z=np.array(info["z"], dtype=np.float32),
            width=width,
            height=height,
            x_fov=info["x_fov"],
            y_fov=info["y_fov"],
        )
    

def rotation_from_forward_vec(forward_vec: Union[np.ndarray, list], up_axis: str = 'Y',
                              inplane_rot: Optional[float] = None) -> np.ndarray:
    """ Returns a camera rotation matrix for the given forward vector and up axis using NumPy

    :param forward_vec: The forward vector which specifies the direction the camera should look.
    :param up_axis: The up axis, usually Y.
    :param inplane_rot: The in-plane rotation in radians. If None is given, the in-plane rotation is determined only
                        based on the up vector.
    :return: The corresponding rotation matrix.
    """
    # Normalize the forward vector
    forward_vector = np.array(forward_vec, dtype=np.float64)
    forward_vector_norm = forward_vector / np.linalg.norm(forward_vector, axis=1, keepdims=True)

    # forward_vec = forward_vec / np.linalg.norm(forward_vec)

    # Define the up vector
    if up_axis.upper() == 'Y':
        up_vec = np.array([0.0, 1.0, 0.0])
    elif up_axis.upper() == 'Z':
        up_vec = np.array([0.0, 0.0, 1.0])
    elif up_axis.upper() == 'X':
        up_vec = np.array([1.0, 0.0, 0.0])
    else:
        raise ValueError("Invalid up_axis. Choose from 'X', 'Y', or 'Z'.")

    # Compute the right vector (cross product of forward and up)
    # right_vec = np.cross(up_vec, forward_vector_norm) # left-hand
    right_vec = np.cross(forward_vector_norm, up_vec)   # right-hand
    right_vec /= np.linalg.norm(right_vec, axis=1, keepdims=True)

    # Recompute the true up vector (orthogonal to forward and right)
    # up_vec = np.cross(forward_vector_norm, right_vec) # left-hand
    up_vec = np.cross(right_vec, forward_vector_norm) # right-hand
    up_vec /= np.linalg.norm(up_vec, axis=1, keepdims=True)

    # # Recompute the right vector
    # right_vec = np.cross(forward_vector_norm, up_vec)
    # right_vec /= np.linalg.norm(right_vec, axis=1, keepdims=True)

    # Construct the rotation matrix (columns represent right, up, forward)
    rotation_matrix = np.stack((right_vec, up_vec, -forward_vector_norm), axis=-1)
    # rotation_matrix = np.stack((right_vec, up_vec, -forward_vector_norm), axis=1)

    # Apply in-plane rotation if specified
    if inplane_rot is not None:
        inplane_rotation = np.array([
            [np.cos(inplane_rot), -np.sin(inplane_rot), 0],
            [np.sin(inplane_rot),  np.cos(inplane_rot), 0],
            [0,                   0,                   1]
        ])
        rotation_matrix = rotation_matrix @ inplane_rotation

    return rotation_matrix


class Front3DBlenderViewData(ViewData):
    """
    Interact with a dataset zipfile exported by view_data.py.
    """

    def __init__(self, render_path, camera_path):
        # self.zipfile = zipfile.ZipFile(f_obj, mode="r")
        # self.infos = []
        # with self.zipfile.open("info.json", "r") as f:
        #     self.info = json.load(f)
        # assert all(k in cam_info for k in ["origin", "x", "y", "z", "x_fov", "y_fov"])
        self.render_path = render_path
        self.camera_path = camera_path
        camera = np.load(os.path.join(camera_path, "boxes.npz"), allow_pickle=True)
        self.build_cam_info(camera)
        # self.channels = list(self.info.get("channels", "RGBAD"))
        self.channels = list("RGBAD")
        assert set("RGBA").issubset(
            set(self.channels)
        ), "The blender output should at least have RGBA images."
        # names = set(x.filename for x in self.zipfile.infolist())
        # for i in itertools.count():
        #     name = f"{i:05}.json"
        #     if name not in names:
        #         break
        #     with self.zipfile.open(name, "r") as f:
        #         self.infos.append(json.load(f))

    @property
    def num_views(self) -> int:
        return len(self.infos)

    @property
    def channel_names(self) -> List[str]:
        return list(self.channels)
    
    def build_cam_info(self, camera):
        camera_coords = camera["camera_coords"]
        target_coords = camera["target_coords"]
        floor_plan_centroid = camera["floor_plan_centroid"]

        forward_vec = target_coords - camera_coords
        rotation_matrix = rotation_from_forward_vec(forward_vec)
        right_vec, up_vec, forward_vec = (
            rotation_matrix[..., 0], rotation_matrix[..., 1], rotation_matrix[..., 2]
        )
        self.infos = []
        for i in range(len(rotation_matrix)):
            # print(f"Rotation matrix {i}:")
            # print(rotation_matrix[i])
            # print()
            self.infos.append(
                {
                    "origin": camera_coords[i], # Camera origin
                    "x": -right_vec[i], # right
                    "y": -up_vec[i], # up
                    "z": -forward_vec[i], # forward
                    "x_fov": np.deg2rad(70), # 70
                    "y_fov": np.deg2rad(70), # 70
                }
            )

    def load_view(self, index: int, channels: List[str]) -> Tuple[Camera, np.ndarray]:
        for ch in channels:
            if ch not in self.channel_names:
                raise ValueError(f"unsupported channel: {ch}")

        channel_map = {}
        if any(x in channels for x in "RGBA"):
            rgba = np.array(Image.open(os.path.join(self.render_path, f"{str(index).zfill(4)}_colors.png"))) / 255.0
            channel_map.update(zip("RGBA", rgba.transpose([2, 0, 1])))
        if "D" in channels:
            depth = np.load(os.path.join(self.render_path, f"{str(index).zfill(4)}_depth.npy"))
            inf_dist = depth == 20.0
            channel_map["D"] = np.where(
                inf_dist, 
                np.inf, 
                # (20) * (depth.astype(np.float32) / 65535.0) # max_depth: scaling points
                # 0.1 + (depth.astype(np.float32) / 255.0) * (1000 - 0.1)
                depth
            )

        combined = np.stack([channel_map[k] for k in channels], axis=-1)
        h, w, _ = combined.shape
        return self.camera(index, w, h), combined
            

    def camera(self, index: int, width: int, height: int) -> ProjectiveCamera:
        info = self.infos[index]
        return ProjectiveCamera(
            origin=np.array(info["origin"], dtype=np.float32),
            x=np.array(info["x"], dtype=np.float32), # right
            y=np.array(info["y"], dtype=np.float32), # up
            z=np.array(info["z"], dtype=np.float32), # forward
            width=width,
            height=height,
            x_fov=info["x_fov"],
            y_fov=info["y_fov"],
        )