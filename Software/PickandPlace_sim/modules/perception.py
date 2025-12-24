import cv2
import numpy as np
from ultralytics import YOLO
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from modules.utils import YOLO_MODEL_PATH

# CAM_TO_WORLD 및 load_yolo 함수는 변경 없음
CAM_TO_WORLD = np.array([
    [-0., 1., 0., 0.],
    [1., -0., -0., 0.],
    [-0., -0., -1., 8.],
    [0., 0., 0., 1.]
])


def load_yolo():
    model = YOLO(YOLO_MODEL_PATH)
    print(f" YOLOv8 모델 로드 완료: {YOLO_MODEL_PATH}")
    return model


def detect_pallet(rgb_data, depth_data, cam_params, model, robot_position):
    h, w, c = rgb_data.shape

    if c == 4:
        img_input = cv2.cvtColor(rgb_data, cv2.COLOR_RGBA2BGR)
    else:
        img_input = cv2.cvtColor(rgb_data, cv2.COLOR_RGB2BGR)

    # 2. YOLO 탐지 (회전 재시도 로직 포함)
    results = model(img_input, verbose=False)
    boxes = [b for b in results[0].boxes if model.names[int(b.cls[0])] == "pallet" and b.conf[0] > 0.5]

    detected_mask = None

    if not boxes:
        print("  [detect_pallet] 정방향 감지 실패 -> 90도 회전 시도...")
        img_rotated = cv2.rotate(img_input, cv2.ROTATE_90_CLOCKWISE)
        results_rot = model(img_rotated, verbose=False)
        boxes_rot = [b for b in results_rot[0].boxes if model.names[int(b.cls[0])] == "pallet" and b.conf[0] > 0.5]

        if boxes_rot:
            print("  [detect_pallet] 회전된 이미지에서 감지 성공!")
            # 회전된 상태에서의 마스크 생성
            box = boxes_rot[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            mask_rotated = np.zeros((w, h), dtype="uint8")
            mask_rotated[y1:y2, x1:x2] = 1

            detected_mask = cv2.rotate(mask_rotated, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # 정방향 성공 시 마스크 생성
    elif boxes:
        box = boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        detected_mask = np.zeros((h, w), dtype="uint8")
        detected_mask[y1:y2, x1:x2] = 1

    # 최종적으로 감지 실패 시
    if detected_mask is None:
        return None, None

    # 3. 마스킹 및 PCD 생성
    # RGB 마스킹
    rgb_masked = np.zeros_like(rgb_data[:, :, :3])
    src_rgb = rgb_data[:, :, :3]
    rgb_masked[detected_mask == 1] = src_rgb[detected_mask == 1]

    # Depth 마스킹
    depth_masked = depth_data.copy()
    depth_masked[detected_mask == 0] = 0

    color_img = o3d.geometry.Image(rgb_masked.copy())
    depth_img = o3d.geometry.Image(depth_masked)
    width, height = cam_params["renderProductResolution"]
    fx = (cam_params["cameraFocalLength"] / cam_params["cameraAperture"][0]) * width
    fy = (cam_params["cameraFocalLength"] / cam_params["cameraAperture"][1]) * height

    intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, width / 2, height / 2)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color_img, depth_img, depth_scale=1.0, depth_trunc=4.9,
                                                              convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)

    if not pcd.has_points() or len(pcd.points) < 100:
        return None, None

    # 4. DBSCAN 클러스터링
    eps = 0.05
    min_points = 10
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    unique_labels = np.unique(labels[labels != -1])
    if len(unique_labels) == 0:
        return None, None
    max_count = 0
    largest_cluster_label = -1
    for label in unique_labels:
        count = np.sum(labels == label)
        if count > max_count:
            max_count = count
            largest_cluster_label = label
    if largest_cluster_label != -1 and max_count >= 100:
        largest_cluster_indices = np.where(labels == largest_cluster_label)[0]
        pcd_cluster = pcd.select_by_index(largest_cluster_indices)
    else:
        return None, None

    # 5. RANSAC 및 나머지 로직
    ransac_n = 3
    if len(pcd_cluster.points) < ransac_n:
        return None, None

    plane_model, inliers = pcd_cluster.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
    if len(inliers) < 100:
        return None, None
    plane_pcd_cam = pcd_cluster.select_by_index(inliers)

    [A, B, C, D] = plane_model
    ransac_normal_cam = np.array([A, B, C])

    points_cam = np.asarray(plane_pcd_cam.points)
    center_cam = np.mean(points_cam, axis=0)
    centered_points = points_cam - center_cam
    cov_matrix = np.cov(centered_points, rowvar=False)
    eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix)
    sort_indices = np.argsort(eigen_values)
    pca_v3_normal = eigen_vectors[:, sort_indices[0]]

    if np.dot(ransac_normal_cam, pca_v3_normal) < 0:
        ransac_normal_cam = -ransac_normal_cam

    normal_cam = ransac_normal_cam
    R_cam_world = CAM_TO_WORLD[0:3, 0:3]

    plane_points_h = np.hstack([points_cam, np.ones((len(points_cam), 1))])
    plane_points_world = (CAM_TO_WORLD @ plane_points_h.T).T[:, :3]
    center_world = np.mean(plane_points_world, axis=0)
    normal_world = R_cam_world @ normal_cam

    # 5.2. 2D Hull + 최소 면적 직사각형 피팅
    z_axis = normal_world / np.linalg.norm(normal_world)
    world_z = np.array([0, 0, 1])

    axis = np.cross(world_z, z_axis)
    axis_norm = np.linalg.norm(axis)
    if axis_norm > 1e-6:
        axis = axis / axis_norm
        angle = np.arccos(np.dot(world_z, z_axis))
        rot_inverse = R.from_rotvec(angle * axis)
    else:
        rot_inverse = R.from_matrix(np.eye(3))
        if np.dot(world_z, z_axis) < 0:
            rot_inverse = R.from_euler('x', 180, degrees=True)

    plane_points_centered = plane_points_world - center_world
    points_2d_temp = rot_inverse.apply(plane_points_centered)
    points_2d = points_2d_temp[:, :2].astype(np.float32)

    if len(points_2d) < 5:
        print("Fallback to PCA for direction vectors.")
        centered_points_world = plane_points_world - center_world
        cov_matrix_world = np.cov(centered_points_world, rowvar=False)
        eigen_values_world, eigen_vectors_world = np.linalg.eigh(cov_matrix_world)
        sort_indices_world = np.argsort(eigen_values_world)
        vec_L_world = eigen_vectors_world[:, sort_indices_world[2]]
        vec_W_world = eigen_vectors_world[:, sort_indices_world[1]]
        length = np.max(np.dot(centered_points_world, vec_L_world)) - np.min(np.dot(centered_points_world, vec_L_world))
        width = np.max(np.dot(centered_points_world, vec_W_world)) - np.min(np.dot(centered_points_world, vec_W_world))

    else:
        rect = cv2.minAreaRect(points_2d)
        angle_rad = np.deg2rad(rect[2])
        v1_2d = np.array([np.cos(angle_rad), np.sin(angle_rad)])
        v2_2d = np.array([-np.sin(angle_rad), np.cos(angle_rad)])

        rot_forward = rot_inverse.inv()
        v1_3d = np.array([v1_2d[0], v1_2d[1], 0.0])
        v2_3d = np.array([v2_2d[0], v2_2d[1], 0.0])

        v1_world_rect = rot_forward.apply(v1_3d)
        v2_world_rect = rot_forward.apply(v2_3d)

        size_x, size_y = rect[1]
        if size_x > size_y:
            length, width = size_x, size_y
            vec_L_world, vec_W_world = v1_world_rect, v2_world_rect
        else:
            length, width = size_y, size_x
            vec_L_world, vec_W_world = v2_world_rect, v1_world_rect

    vec_L_world /= np.linalg.norm(vec_L_world)
    vec_W_world /= np.linalg.norm(vec_W_world)

    side_info = {
        "long_side_A": {"center": center_world + vec_W_world * (width / 2), "normal": vec_W_world},
        "long_side_B": {"center": center_world - vec_W_world * (width / 2), "normal": -vec_W_world},
        "short_side_A": {"center": center_world + vec_L_world * (length / 2), "normal": vec_L_world},
        "short_side_B": {"center": center_world - vec_L_world * (length / 2), "normal": -vec_L_world}
    }

    pallet_top_z = center_world[2]
    for name in side_info:
        side_info[name]["center"][2] = pallet_top_z

    min_dist = float('inf')
    closest_side_name = None
    for name, info in side_info.items():
        dist = np.linalg.norm((info["center"] - robot_position)[:2])
        if dist < min_dist:
            min_dist = dist
            closest_side_name = name

    closest_side = side_info[closest_side_name]
    target_position = closest_side["center"] + np.array([0, 0, 0.015]) + closest_side["normal"] * 0.19

    z_axis = -closest_side["normal"]
    y_axis = np.cross(z_axis, np.array([1, 0, 0]))
    if np.linalg.norm(y_axis) < 1e-6:
        y_axis = np.cross(z_axis, np.array([0, 1, 0]))
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)

    rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
    rot = R.from_matrix(rotation_matrix)
    quat = rot.as_quat()
    target_orientation = np.array([quat[3], quat[0], quat[1], quat[2]])

    pallet_info = {
        "position": target_position,
        "orientation": target_orientation,
        "target_side_name": closest_side_name,
        "side_info_world": side_info,
        "center_world": center_world,
        "normal_world": normal_world,
        "pca_v1_world": vec_L_world
    }

    plane_pcd_world = o3d.geometry.PointCloud()
    plane_pcd_world.points = o3d.utility.Vector3dVector(plane_points_world)
    return pallet_info, plane_pcd_world