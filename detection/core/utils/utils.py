import math
from functools import partial

import cv2
import numpy as np
import torch
from pyquaternion import Quaternion
from scipy.linalg import expm, logm
from shapely.geometry import LinearRing, Polygon
from transforms3d.axangles import axangle2mat
from transforms3d.quaternions import axangle2quat, mat2quat, qmult, quat2mat

from detectron2.layers import cat

from .dls_pnp import dls_pnp
from .torch_pnp import UPnP
from .upnp import upnp
from .pose_utils import quat2mat_torch


def set_nan_to_0(a, name=None, verbose=False):
    if torch.isnan(a).any():
        if verbose and name is not None:
            print("nan in {}".format(name))
        a[a != a] = 0
    return a


def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    # return tuple(map(list, zip(*map_results)))
    return list(map_results)


def get_affine_matrix(crop_height, crop_width, zoom_c_x, zoom_c_y, height, width):
    """affine transformation for PyTorch"""
    x1 = (zoom_c_x - crop_width / 2) * 2 / width - 1
    x2 = (zoom_c_x + crop_width / 2) * 2 / width - 1
    y1 = (zoom_c_y - crop_height / 2) * 2 / height - 1
    y2 = (zoom_c_y + crop_height / 2) * 2 / height - 1

    pts_src = np.float32([[-1, -1], [-1, 1], [1, -1]])
    pts_dst = np.float32([[x1, y1], [x1, y2], [x2, y1]])
    affine_matrix = torch.tensor(cv2.getAffineTransform(pts_src, pts_dst), dtype=torch.float32)
    """
    wx = crop_width / width
    wy = crop_height / height
    tx = zoom_c_x / width * 2 - 1
    ty = zoom_c_y / height * 2 - 1
    affine_matrix = torch.tensor([[wx, 0, tx], [0, wy, ty]])
    """
    return affine_matrix


def allocentric_to_egocentric_torch(translation, q_allo, eps=1e-4):
    """ Given an allocentric (object-centric) pose, compute new camera-centric pose
    Since we do detection on the image plane and our kernels are 2D-translationally invariant,
    we need to ensure that rendered objects always look identical, independent of where we render them.
    Since objects further away from the optical center undergo skewing, we try to visually correct by
    rotating back the amount between optical center ray and object centroid ray.
    Another way to solve that might be translational variance (https://arxiv.org/abs/1807.03247)
    Args:
        translation: Nx3
        q_allo: Nx4
    """

    # Compute rotation between ray to object centroid and optical center ray
    cam_ray = torch.tensor([0, 0, 1.0], dtype=translation.dtype, device=translation.device)  # (3,)
    obj_ray = translation / (torch.norm(translation, dim=1, keepdim=True) + eps)

    # cam_ray.dot(obj_ray), assume cam_ray: (0, 0, 1)
    angle = obj_ray[:, 2:3].acos()

    # Compute rotation between ray to object centroid and optical center ray
    axis = torch.cross(cam_ray.expand_as(obj_ray), obj_ray)
    axis = axis / (torch.norm(axis, dim=1, keepdim=True) + eps)

    # Build quaternion representing the rotation around the computed axis
    # angle-axis => quat
    q_allo_to_ego = cat(
        [
            torch.cos(angle / 2.0),
            axis[:, 0:1] * torch.sin(angle / 2.0),
            axis[:, 1:2] * torch.sin(angle / 2.0),
            axis[:, 2:3] * torch.sin(angle / 2.0),
        ],
        dim=1,
    )

    # Apply quaternion for transformation from allocentric to egocentric.
    q_ego = quatmul_torch(q_allo_to_ego, q_allo)[:, 0]  # Remove added Corner dimension here.
    return q_ego


def allo_to_ego_mat_torch(translation, rot_allo, eps=1e-4):
    # translation: Nx3
    # rot_allo: Nx3x3
    # Compute rotation between ray to object centroid and optical center ray
    cam_ray = torch.tensor([0, 0, 1.0], dtype=translation.dtype, device=translation.device)  # (3,)
    obj_ray = translation / (torch.norm(translation, dim=1, keepdim=True) + eps)

    # cam_ray.dot(obj_ray), assume cam_ray: (0, 0, 1)
    angle = obj_ray[:, 2:3].acos()

    # Compute rotation between ray to object centroid and optical center ray
    axis = torch.cross(cam_ray.expand_as(obj_ray), obj_ray)
    axis = axis / (torch.norm(axis, dim=1, keepdim=True) + eps)

    # Build quaternion representing the rotation around the computed axis
    # angle-axis => quat
    q_allo_to_ego = cat(
        [
            torch.cos(angle / 2.0),
            axis[:, 0:1] * torch.sin(angle / 2.0),
            axis[:, 1:2] * torch.sin(angle / 2.0),
            axis[:, 2:3] * torch.sin(angle / 2.0),
        ],
        dim=1,
    )
    rot_allo_to_ego = quat2mat_torch(q_allo_to_ego)
    # Apply quaternion for transformation from allocentric to egocentric.
    rot_ego = torch.matmul(rot_allo_to_ego, rot_allo)
    return rot_ego


def quatmul_torch(q1, q2):
    """Computes the multiplication of two quaternions.
    Note, output dims: NxMx4 with N being the batchsize and N the number of quaternions or 3D points to be
    transformed.
    """
    # RoI dimension. Unsqueeze if not fitting.
    a = q1.unsqueeze(0) if q1.dim() == 1 else q1
    b = q2.unsqueeze(0) if q2.dim() == 1 else q2

    # Corner dimension. Unsequeeze if not fitting.
    a = a.unsqueeze(1) if a.dim() == 2 else a
    b = b.unsqueeze(1) if b.dim() == 2 else b

    # Quaternion product
    x = a[:, :, 1] * b[:, :, 0] + a[:, :, 2] * b[:, :, 3] - a[:, :, 3] * b[:, :, 2] + a[:, :, 0] * b[:, :, 1]
    y = -a[:, :, 1] * b[:, :, 3] + a[:, :, 2] * b[:, :, 0] + a[:, :, 3] * b[:, :, 1] + a[:, :, 0] * b[:, :, 2]
    z = a[:, :, 1] * b[:, :, 2] - a[:, :, 2] * b[:, :, 1] + a[:, :, 3] * b[:, :, 0] + a[:, :, 0] * b[:, :, 3]
    w = -a[:, :, 1] * b[:, :, 1] - a[:, :, 2] * b[:, :, 2] - a[:, :, 3] * b[:, :, 3] + a[:, :, 0] * b[:, :, 0]

    return torch.stack((w, x, y, z), dim=2)


def geodesic_mean(rots, epsilon=1e-2, max_iterations=1000):
    mu = rots[0]
    for i in range(max_iterations):
        avgX = np.zeros((3, 3), dtype=np.float32)
        for rot in rots:
            dXi = np.linalg.solve(mu, rot)
            dxu = logm(dXi)
            avgX = avgX + dxu

        dmu = expm((1.0 / len(rots)) * avgX)
        mu = np.matmul(mu, dmu)
        if np.linalg.norm(logm(dmu)) <= epsilon:
            break

    return mu


def weiszfeld_interpolation(latents, iterations=1000, quats=False):
    z = latents[0]
    for i in range(iterations):
        newz = np.zeros(z.shape, dtype=np.float32)
        dividend = 0
        for j in range(1, len(latents)):
            latent = latents[j]
            if quats:
                distance = np.arccos(np.clip(np.abs(np.sum(np.multiply(latent, z))), 0, 0.999))
            else:
                distance = np.arccos(np.clip(np.sum(np.multiply(latent, z)), -0.999, 0.999))

            if distance < 1e-4:
                continue
            newz = newz + latents[j] / distance
            dividend += 1 / distance
        newz /= dividend
        z = newz
    return z


def matrix2quaternion(matrix):
    """
    This code uses a modification of the algorithm described in:
    https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf
    which is itself based on the method described here:
    http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/

    Altered to work with the column vector convention instead of row vectors
    """
    # This method assumes row-vector and postmultiplication of that vector
    m = matrix.conj().transpose()
    if m[2, 2] < 0:
        if m[0, 0] > m[1, 1]:
            t = 1 + m[0, 0] - m[1, 1] - m[2, 2]
            q = [m[1, 2] - m[2, 1], t, m[0, 1] + m[1, 0], m[2, 0] + m[0, 2]]
        else:
            t = 1 - m[0, 0] + m[1, 1] - m[2, 2]
            q = [m[2, 0] - m[0, 2], m[0, 1] + m[1, 0], t, m[1, 2] + m[2, 1]]
    else:
        if m[0, 0] < -m[1, 1]:
            t = 1 - m[0, 0] - m[1, 1] + m[2, 2]
            q = [m[0, 1] - m[1, 0], m[2, 0] + m[0, 2], m[1, 2] + m[2, 1], t]
        else:
            t = 1 + m[0, 0] + m[1, 1] + m[2, 2]
            q = [t, m[1, 2] - m[2, 1], m[2, 0] - m[0, 2], m[0, 1] - m[1, 0]]

    q = np.array(q)
    q *= 0.5 / np.sqrt(t)

    q = q / np.linalg.norm(q)
    return q


def iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def non_max_suppression(boxes, width, height, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0] * width
    y1 = boxes[:, 1] * height
    x2 = boxes[:, 2] * width
    y2 = boxes[:, 3] * height

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return pick


def bev_supression(poses, extends, scores, iou_threshold=0.05):
    if iou_threshold == 0:
        return range(len(poses))

    polygons = []

    for id, (pose, extend) in enumerate(zip(poses, extends)):

        pose = np.asarray(pose)
        width, height, length = extend[0], extend[1], extend[2]

        # Build box at origin with provided extends
        points = np.asarray([
            [width / 2, 0, length / 2],
            [-width / 2, 0, length / 2],
            [-width / 2, 0, -length / 2],
            [width / 2, 0, -length / 2],
        ])

        # Transform with pose and project into BEV space
        transformed = ((pose[:3, :3] @ points.T).T + pose[:3, 3])[:, [0, 2]]  # take x and z

        polygons.append([Polygon(transformed), scores[id]])

    nms_idx = np.arange(len(polygons)).tolist()

    # as long as we can merge
    while True:
        did_merge = False

        for i in range(len(polygons)):
            poli_i, score_i = polygons[i]
            for j in range(i + 1, len(polygons)):
                poli_j, score_j = polygons[j]
                intersection = poli_i.intersection(poli_j).area
                union = poli_i.union(poli_j).area
                iou = intersection / union

                if iou > iou_threshold:
                    if score_i > score_j:
                        del polygons[j]
                        del nms_idx[j]
                    else:
                        del polygons[i]
                        del nms_idx[i]
                    did_merge = True
                    break
            if did_merge:
                break

        if not did_merge:
            break

    return nms_idx


def simple_3d_nms(poses, extends, shapes, scores, merge=False):
    merged = True
    idx = np.arange(len(poses)).tolist()
    scores_copy = scores.copy()
    while merged:
        merged = False
        for i in range(len(poses)):
            for j in range(i + 1, len(poses)):
                extends_min = np.min(np.asarray(extends[i]) * 0.5 + np.asarray(extends[j]) * 0.5)
                dist = np.linalg.norm(poses[i][:3, 3] - poses[j][:3, 3])

                if dist < extends_min:
                    if merge:
                        avg_pose = np.identity(4)
                        avg_pose[:3, 3] = (poses[i][:3, 3] + poses[j][:3, 3]) / 2.0
                        avg_pose[:3, :3] = geodesic_mean([poses[i][:3, :3], poses[j][:3, :3]], max_iterations=1)

                        # avg_shape = np.mean([shapes[i], shapes[j]], axis=0)
                        avg_shape = weiszfeld_interpolation([shapes[i], shapes[j]], iterations=1)
                        avg_extends = np.mean([extends[i], extends[j]], axis=0)

                        del poses[i], poses[j - 1]
                        poses.append(avg_pose)

                        del shapes[i], shapes[j - 1]
                        shapes.append(avg_shape)

                        del extends[i], extends[j - 1]
                        extends.append(avg_extends)

                    if scores_copy[i] > scores_copy[j]:
                        scores_copy.append(scores_copy[i])
                        idx.append(i)

                        if not merge:
                            shapes.append(shapes[i])
                            extends.append(extends[i])
                            poses.append(poses[i])

                        del scores_copy[i], scores_copy[j - 1]
                        del idx[i], idx[j - 1]
                        if not merge:
                            del poses[i], poses[j - 1]
                            del shapes[i], shapes[j - 1]
                            del extends[i], extends[j - 1]

                    else:
                        scores_copy.append(scores_copy[j])
                        idx.append(j)

                        if not merge:
                            shapes.append(shapes[j])
                            extends.append(extends[j])
                            poses.append(poses[j])

                        del scores_copy[i], scores_copy[j - 1]
                        del idx[i], idx[j - 1]
                        if not merge:
                            del poses[i], poses[j - 1]
                            del shapes[i], shapes[j - 1]
                            del extends[i], extends[j - 1]
                    merged = True
                    break
            if merged:
                break

    return poses, extends, shapes, idx


def PnP(bbox_projection, cam, extends, method="epnp"):
    width, height, length = extends[0], extends[1], extends[2]

    # Build 3D object points (front face, back face, centroid)
    obj_points = [
        [-width / 2, -height / 2, length / 2],
        [width / 2, -height / 2, length / 2],
        [width / 2, height / 2, length / 2],
        [-width / 2, height / 2, length / 2],
        [-width / 2, -height / 2, -length / 2],
        [width / 2, -height / 2, -length / 2],
        [width / 2, height / 2, -length / 2],
        [-width / 2, height / 2, -length / 2],
        [0, 0, 0],
    ]

    if isinstance(bbox_projection, torch.Tensor):
        pose = torch.eye(4, dtype=cam.dtype, device=cam.device)
        obj_points = torch.tensor(obj_points)
        results = UPnP()(obj_points, bbox_projection, cam)
        pose[:3, :3] = results[0][1]
        pose[:3, 3] = results[0][2]
        return pose

    # Run PnP to retrieve full 6D pose
    pose = np.eye(4)
    obj_points = np.asarray(obj_points)
    if method == "epnp":
        _, rvecs, tvecs = cv2.solvePnP(
            obj_points, bbox_projection.reshape((9, 1, 2)), cam, None, flags=cv2.SOLVEPNP_EPNP)
        pose[:3, :3] = cv2.Rodrigues(rvecs)[0]
        pose[:3, 3] = tvecs.flatten()
    elif method == "dls":
        results = dls_pnp(obj_points, bbox_projection, cam)
        pose[:3, :3] = results[0][0]
        pose[:3, 3] = results[0][1]
    elif method == "upnp":
        results = upnp(obj_points, bbox_projection, cam)
        pose[:3, :3] = results[0][1]
        pose[:3, 3] = results[0][2]
    return pose


def allocentric_to_egocentric(allo_pose, src_type="mat", dst_type="mat"):
    """ Given an allocentric (object-centric) pose, compute new camera-centric pose
    Since we do detection on the image plane and our kernels are 2D-translationally invariant,
    we need to ensure that rendered objects always look identical, independent of where we render them.
    Since objects further away from the optical center undergo skewing, we try to visually correct by
    rotating back the amount between optical center ray and object centroid ray.
    Another way to solve that might be translational variance (https://arxiv.org/abs/1807.03247)
    """
    # Compute rotation between ray to object centroid and optical center ray
    cam_ray = np.asarray([0, 0, 1.0])
    if src_type == "mat":
        trans = allo_pose[:3, 3]
    elif src_type == "quat":
        trans = allo_pose[4:7]
    else:
        raise ValueError("src_type should be mat or quat, got: {}".format(src_type))
    obj_ray = trans.copy() / np.linalg.norm(trans)
    angle = math.acos(cam_ray.dot(obj_ray))

    # Rotate back by that amount
    if angle > 0:
        if dst_type == "mat":
            ego_pose = np.zeros((3, 4), dtype=allo_pose.dtype)
            ego_pose[:3, 3] = trans
            rot_mat = axangle2mat(axis=np.cross(cam_ray, obj_ray), angle=angle)
            if src_type == "mat":
                ego_pose[:3, :3] = np.dot(rot_mat, allo_pose[:3, :3])
            elif src_type == "quat":
                ego_pose[:3, :3] = np.dot(rot_mat, quat2mat(allo_pose[:4]))
        elif dst_type == "quat":
            ego_pose = np.zeros((7,), dtype=allo_pose.dtype)
            ego_pose[4:7] = trans
            rot_q = axangle2quat(np.cross(cam_ray, obj_ray), angle)
            if src_type == "quat":
                ego_pose[:4] = qmult(rot_q, allo_pose[:4])
            elif src_type == "mat":
                ego_pose[:4] = qmult(rot_q, mat2quat(allo_pose[:3, :3]))
        else:
            raise ValueError("dst_type should be mat or quat, got: {}".format(dst_type))
    else:  # allo to ego
        if src_type == "mat" and dst_type == "quat":
            ego_pose = np.zeros((7,), dtype=allo_pose.dtype)
            ego_pose[:4] = mat2quat(allo_pose[:3, :3])
            ego_pose[4:7] = allo_pose[:3, 3]
        elif src_type == "quat" and dst_type == "mat":
            ego_pose = np.zeros((3, 4), dtype=allo_pose.dtype)
            ego_pose[:3, :3] = quat2mat(allo_pose[:4])
            ego_pose[:3, 3] = allo_pose[4:7]
        else:
            ego_pose = allo_pose.copy()
    return ego_pose


def egocentric_to_allocentric(ego_pose, src_type="mat", dst_type="mat", cam_ray=(0, 0, 1.0)):
    # Compute rotation between ray to object centroid and optical center ray
    cam_ray = np.asarray(cam_ray)
    if src_type == "mat":
        trans = ego_pose[:3, 3]
    elif src_type == "quat":
        trans = ego_pose[4:7]
    else:
        raise ValueError("src_type should be mat or quat, got: {}".format(src_type))
    obj_ray = trans.copy() / np.linalg.norm(trans)
    angle = math.acos(cam_ray.dot(obj_ray))

    # Rotate back by that amount
    if angle > 0:
        if dst_type == "mat":
            allo_pose = np.zeros((3, 4), dtype=ego_pose.dtype)
            allo_pose[:3, 3] = trans
            rot_mat = axangle2mat(axis=np.cross(cam_ray, obj_ray), angle=-angle)
            if src_type == "mat":
                allo_pose[:3, :3] = np.dot(rot_mat, ego_pose[:3, :3])
            elif src_type == "quat":
                allo_pose[:3, :3] = np.dot(rot_mat, quat2mat(ego_pose[:4]))
        elif dst_type == "quat":
            allo_pose = np.zeros((7,), dtype=ego_pose.dtype)
            allo_pose[4:7] = trans
            rot_q = axangle2quat(np.cross(cam_ray, obj_ray), -angle)
            if src_type == "quat":
                allo_pose[:4] = qmult(rot_q, ego_pose[:4])
            elif src_type == "mat":
                allo_pose[:4] = qmult(rot_q, mat2quat(ego_pose[:3, :3]))
        else:
            raise ValueError("dst_type should be mat or quat, got: {}".format(dst_type))
    else:
        if src_type == "mat" and dst_type == "quat":
            allo_pose = np.zeros((7,), dtype=ego_pose.dtype)
            allo_pose[:4] = mat2quat(ego_pose[:3, :3])
            allo_pose[4:7] = ego_pose[:3, 3]
        elif src_type == "quat" and dst_type == "mat":
            allo_pose = np.zeros((3, 4), dtype=ego_pose.dtype)
            allo_pose[:3, :3] = quat2mat(ego_pose[:4])
            allo_pose[:3, 3] = ego_pose[4:7]
        else:
            allo_pose = ego_pose.copy()
    return allo_pose


def test_ego_allo():
    ego_pose = np.zeros((3, 4), dtype=np.float32)
    ego_pose[:3, :3] = axangle2mat((1, 2, 3), 1)
    ego_pose[:3, 3] = np.array([0.4, 0.5, 0.6])
    ego_pose_q = np.zeros((7,), dtype=np.float32)
    ego_pose_q[:4] = mat2quat(ego_pose[:3, :3])
    ego_pose_q[4:7] = ego_pose[:3, 3]
    ego_poses = {"mat": ego_pose, "quat": ego_pose_q}
    rot_types = ["mat", "quat"]
    for src_type in rot_types:
        for dst_type in rot_types:
            allo_pose = egocentric_to_allocentric(ego_poses[src_type], src_type, dst_type)
            ego_pose_1 = allocentric_to_egocentric(allo_pose, dst_type, src_type)
            print(src_type, dst_type)
            print("ego_pose: ", ego_poses[src_type])
            print("allo_pose from ego_pose: ", allo_pose)
            print("ego_pose from allo_pose: ", ego_pose_1)
            print(np.allclose(ego_poses[src_type], ego_pose_1))
            print("************************")


# test_ego_allo()
# exit(0)


def project_flipped(points, cam, height=1):
    points = (cam @ points.T).T
    points = points[:, :2] / points[:, 2:3]
    points[:, 1] = height - points[:, 1]
    return points


def backproject_flipped(points, z, cam, height=1):
    points[:, 1] = height - points[:, 1]
    return np.concatenate([(points[:, 0:1] - cam[0:1, 2]) / cam[0:1, 0] * z,
                           (points[:, 1:2] - cam[1:2, 2]) / cam[1:2, 1] * z, z],
                          axis=1)


def backproject(depth, cam):
    """ Backproject a depth map to a cloud map
    :param depth: Input depth map
    :param cam: Intrinsics of the camera
    :return: An organized cloud map
    """
    X = np.asarray(range(depth.shape[1])) - cam[0, 2]
    X = np.tile(X, (depth.shape[0], 1))
    Y = np.asarray(range(depth.shape[0])) - cam[1, 2]
    Y = np.tile(Y, (depth.shape[1], 1)).transpose()
    return np.stack((X * depth / cam[0, 0], Y * depth / cam[1, 1], depth), axis=2)


# Pose of can not be rotated around Z
def car_pose_sanity_check(pose, sanity_threshol=10):
    yaw, pitch, roll = Quaternion(matrix=pose[:3, :3]).yaw_pitch_roll
    roll = np.abs(roll) / np.pi * 180

    return roll < sanity_threshol or roll > 180 - sanity_threshol


def draw_birdview(hy_poses,
                  hy_extends,
                  gt_poses=None,
                  gt_extends=None,
                  gt_labels=None,
                  target_size=None,
                  bev_idx=None,
                  ply=False):

    bev = np.zeros((1400, 1200, 3), dtype=np.float32)

    for i in range(10):
        circle_center = (int(bev.shape[1] / 2), int(bev.shape[0]))
        bev = cv2.circle(bev, circle_center, i * 200, (1, 1, 1), thickness=2)

    if bev_idx is None:
        bev_idx = range(len(hy_poses))

    for idx in bev_idx:
        hy_pose, hy_extend = hy_poses[idx], hy_extends[idx]
        hy_pose = np.asarray(hy_pose)
        width, height, length = hy_extend[0], hy_extend[1], hy_extend[2]

        # Build box at origin with provided extends
        if ply:
            hy_points = np.asarray([
                [length / 2, width / 2, 0],
                [length / 2, -width / 2, 0],
                [-length / 2, width / 2, 0],
                [-length / 2, -width / 2, 0],
            ])
        else:
            hy_points = np.asarray([
                [width / 2, 0, length / 2],
                [-width / 2, 0, length / 2],
                [width / 2, 0, -length / 2],
                [-width / 2, 0, -length / 2],
            ])

        # Transform with pose and project into BEV space
        hy_points = np.asarray(hy_points)
        hy_transformed = ((hy_pose[:3, :3] @ hy_points.T).T + hy_pose[:3, 3]) * 20.0

        hy_bev_x = np.clip(hy_transformed[:, 0] + bev.shape[1] / 2.0, 0, bev.shape[1])
        if ply:
            hy_bev_y = np.clip(bev.shape[0] - hy_transformed[:, 1], 0, bev.shape[0])
        else:
            hy_bev_y = np.clip(bev.shape[0] - hy_transformed[:, 2], 0, bev.shape[0])
        hy_bev = np.vstack((hy_bev_x, hy_bev_y)).astype(np.int32).T

        # Draw the bounding rectangle in BEV
        bev = cv2.line(bev, tuple(hy_bev[0]), tuple(hy_bev[1]), color=(0, 1, 0), thickness=2)
        bev = cv2.line(bev, tuple(hy_bev[0]), tuple(hy_bev[2]), color=(0, 1, 0), thickness=2)
        bev = cv2.line(bev, tuple(hy_bev[1]), tuple(hy_bev[3]), color=(0, 1, 0), thickness=2)
        bev = cv2.line(bev, tuple(hy_bev[2]), tuple(hy_bev[3]), color=(0, 1, 0), thickness=2)

        # Draw direction vector
        bev_center = np.mean(hy_bev, axis=0)
        bev_dir = np.mean(hy_bev[:2] - hy_bev[2:], axis=0)
        bev_dir = 50 * bev_dir / np.linalg.norm(bev_dir)

        bev = cv2.line(
            bev,
            tuple(bev_center.astype(np.int32)),
            tuple((bev_center + bev_dir).astype(np.int32)),
            color=(0, 1, 0),
            thickness=2,
        )

        car_dir = np.asarray([[0, 0, length], [0, 0, 0]])
        car_dir = (np.matmul(hy_pose[:3, :3], car_dir.T) + hy_pose[:3, 3:4]).T[:, 0::2] * 20.0

        car_dir[:, 0] = np.clip(car_dir[:, 0] + bev.shape[1] / 2.0, 0, bev.shape[1])
        car_dir[:, 1] = np.clip(bev.shape[0] - car_dir[:, 1], 0, bev.shape[0])

        car_dir = np.asarray(car_dir, dtype=np.int32)
        bev = cv2.line(bev, tuple(car_dir[0]), tuple(car_dir[1]), color=(0, 1, 0), thickness=1)

    if gt_poses is not None and gt_extends is not None:

        for gt_pose, gt_extend, gt_label in zip(gt_poses, gt_extends, gt_labels):

            gt_pose = np.asarray(gt_pose)

            width, height, length = gt_extend[0], gt_extend[1], gt_extend[2]
            if ply:
                gt_points = np.asarray([
                    [length / 2, width / 2, 0],
                    [length / 2, -width / 2, 0],
                    [-length / 2, width / 2, 0],
                    [-length / 2, -width / 2, 0],
                ])
            else:
                gt_points = np.asarray([
                    [width / 2, 0, length / 2],
                    [-width / 2, 0, length / 2],
                    [width / 2, 0, -length / 2],
                    [-width / 2, 0, -length / 2],
                ])

            gt_points_plane = (np.matmul(gt_pose[:3, :3], gt_points.T) + gt_pose[:3, 3:4]).T[:, 0::2] * 20.0

            gt_points_plane[:, 0] = np.clip(gt_points_plane[:, 0] + bev.shape[1] / 2.0, 0, bev.shape[1])
            gt_points_plane[:, 1] = np.clip(bev.shape[0] - gt_points_plane[:, 1], 0, bev.shape[0])

            gt_points_plane = np.asarray(gt_points_plane, dtype=np.int32)

            draw_col = (0, 0, 1)
            if gt_label == 1:
                draw_col = (0, 1, 1)
            bev = cv2.line(bev, tuple(gt_points_plane[0]), tuple(gt_points_plane[1]), color=draw_col, thickness=2)
            bev = cv2.line(bev, tuple(gt_points_plane[0]), tuple(gt_points_plane[2]), color=draw_col, thickness=2)
            bev = cv2.line(bev, tuple(gt_points_plane[1]), tuple(gt_points_plane[3]), color=draw_col, thickness=2)
            bev = cv2.line(bev, tuple(gt_points_plane[2]), tuple(gt_points_plane[3]), color=draw_col, thickness=2)

            car_dir = np.asarray([[0, 0, length], [0, 0, 0]])
            car_dir = (np.matmul(gt_pose[:3, :3], car_dir.T) + gt_pose[:3, 3:4]).T[:, 0::2] * 20.0

            car_dir[:, 0] = np.clip(car_dir[:, 0] + bev.shape[1] / 2.0, 0, bev.shape[1])
            car_dir[:, 1] = np.clip(bev.shape[0] - car_dir[:, 1], 0, bev.shape[0])

            car_dir = np.asarray(car_dir, dtype=np.int32)
            bev = cv2.line(bev, tuple(car_dir[0]), tuple(car_dir[1]), color=draw_col, thickness=2)

    # if scaling is desired
    if target_size is not None:

        # if no width given, scale width according to height ratio
        if target_size[0] is None:
            target_size[0] = (target_size[1] / bev.shape[0]) * bev.shape[1]

        # if no height given, scale height according to width ratio
        elif target_size[1] is None:
            target_size[1] = (target_size[0] / bev.shape[1]) * bev.shape[0]

        bev = cv2.resize(bev, (int(target_size[0]), int(target_size[1])))

    return bev


def angle_bev(pose):
    car = np.asarray([[0, 0, 1], [0, 0, 0]])
    car = (np.matmul(pose[:3, :3], car.T) + pose[:3, 3:4]).T[:, ::2]
    car_dir = car[0] - car[1]

    angle = np.arccos(np.dot(car_dir, (1, 0)))

    if car_dir[1] > 0:
        angle *= -1

    alpha = angle - np.arctan2(car[0, 0], car[0, 1])
    return angle, alpha


def dump_points(filename, points, rgb=True):
    rofl = open(filename, "w")
    rofl.write("ply\n")
    rofl.write("format ascii 1.0\n")
    rofl.write("element vertex {}\n".format(len(points)))
    rofl.write("property float x\n")
    rofl.write("property float y\n")
    rofl.write("property float z\n")
    if rgb:
        rofl.write("property uchar red\n")
        rofl.write("property uchar green\n")
        rofl.write("property uchar blue\n")
    rofl.write("end_header\n")
    [rofl.write(" ".join(list(map(str, p))) + "\n") for p in points]


def iou(a, b):
    # Compute Intersection
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])

    # compute the area of intersection rectangle
    inter = max(0, xB - xA) * max(0, yB - yA)

    # Compute Union
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 10e-5)


def compute_contour(mask):
    interior = cv2.erode(mask, np.ones((3, 3), np.uint8))
    return mask * (1 - interior).astype(np.uint8), interior


def compute_sdf(mask, signed=True, with_indices=True):
    """ Returns a signed distance transform for a mask, and label indices.
    :param mask: Zero values are exterior, non-zero values are interior area
    """

    # Get interior mask by erosion, compute contour from that and invert for 0-level set
    contour, interior = compute_contour(mask)
    zero_level = (1 - contour).astype(np.uint8)

    # Build mapping between 1D contour and label index
    mask2idx = [-1]
    mask2idx.extend(np.flatnonzero(contour))

    if with_indices:
        dt, idx = cv2.distanceTransformWithLabels(
            zero_level, cv2.DIST_L2, cv2.DIST_MASK_PRECISE, labelType=cv2.DIST_LABEL_PIXEL)
    else:
        dt = cv2.distanceTransform(zero_level, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    if signed:
        dt[interior.astype(bool)] *= -1  # Make interior negative

    if not with_indices:
        return dt

    # Map back from pixel label to 1D image position
    for r in range(idx.shape[0]):
        for c in range(idx.shape[1]):
            idx[r, c] = mask2idx[idx[r, c]]

    return dt, idx, contour


def heatmap(input, min=None, max=None, to_255=False, to_rgb=False):
    """ Returns a BGR heatmap representation """
    if min is None:
        min = np.amin(input)
    if max is None:
        max = np.amax(input)
    rescaled = 255 * ((input - min) / (max - min + 0.001))

    final = cv2.applyColorMap(rescaled.astype(np.uint8), cv2.COLORMAP_JET)
    if to_rgb:
        final = final[:, :, [2, 1, 0]]
    if to_255:
        return final.astype(np.uint8)
    else:
        return final.astype(np.float32) / 255.0
