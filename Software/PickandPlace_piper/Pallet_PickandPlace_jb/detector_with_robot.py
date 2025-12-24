#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import sys
import time
import threading
import numpy as np

import argparse
import torch
import cv2
import pyzed.sl as sl
from ultralytics import YOLO

from threading import Lock, Thread
from time import sleep

import ogl_viewer.viewer as gl
import cv_viewer.tracking_viewer as cv_viewer
import datetime
import open3d as o3d

# 로봇 관련
from scipy.spatial.transform import Rotation as R
from piper_sdk import *

# PiPER 로봇 클래스
def snap_height_to_nearest_level(target_z):
    # 타겟 층 높이 값 (mm)
    pallet_heights = [195.5, 176.7, 156.6, 136.2]

    # 가장 가까운 값 선택
    nearest = min(pallet_heights, key=lambda h: abs(h - target_z))
    return nearest


class PiPER_Palletizing:
    def __init__(self, piper: C_PiperInterface):
        self.piper = piper
        self.enable()

    # 그리퍼 열기 / 닫기 헬퍼
    def grip_open(self, force: int = 80000, speed: int = 1000):
        # 그리퍼 열기
        self.piper.GripperCtrl(force, speed, 0x01, 0)
        time.sleep(1.5)

    def grip_close(self, force: int = 50000, speed: int = 1000):
        # 그리퍼 닫기
        self.piper.GripperCtrl(force, speed, 0x01, 0)
        time.sleep(1.5)

    def grip_grab(self, force: int = 15000, speed: int = 1000):
        # 그리퍼 잡기
        self.piper.GripperCtrl(force, speed, 0x01, 0)
        time.sleep(1.5)

    def enable(self):
        
        # 로봇팔을 활성화하고, 모터가 준비될 때까지 대기
        
        enable_flag = False
        elapsed_time_flag = False
        timeout = 5
        start_time = time.time()

        while not enable_flag:
            elapsed_time = time.time() - start_time
            enable_flag = all(
                getattr(self.piper.GetArmLowSpdInfoMsgs(), f"motor_{i}").foc_status.driver_enable_status
                for i in range(1, 7)
            )
            self.piper.EnableArm(7)
            if elapsed_time > timeout:
                print("시간초과")
                elapsed_time_flag = True
                enable_flag = True
                break
            time.sleep(1)

        if elapsed_time_flag:
            print("프로그램을 종료합니다")
            exit(0)

        # 초기 상태: 그리퍼 열기
        self.grip_open()
        self.grip_close()
        print("✅ Piper arm enabled and gripper opened")

    def wait_for_motion_complete(self, target):
        # 목표 위치 도달 확인
        while True:
            pose = self.piper.GetArmEndPoseMsgs().end_pose
            cur = [pose.X_axis, pose.Y_axis, pose.Z_axis]
            if all(abs(cur[i] - target[i]) < 1000 for i in range(3)):
                break
            time.sleep(0.1)

    def position_to_int(self, pos):
        # 원본 코드와 동일: 모든 축에 1000 곱함
        return [int(c * 1000) for c in pos]

    def move_l(self, x, y, z, rx, ry, rz, speed=20):
        x, y, z, rx, ry, rz = self.position_to_int([x, y, z, rx, ry, rz])
        self.piper.MotionCtrl_2(0x01, 0x02, speed)
        self.piper.EndPoseCtrl(x, y, z, rx, ry, rz)

    def move_p(self, x, y, z, rx, ry, rz, speed=50):
        x, y, z, rx, ry, rz = self.position_to_int([x, y, z, rx, ry, rz])
        self.piper.MotionCtrl_2(0x01, 0x00, speed)
        self.piper.EndPoseCtrl(x, y, z, rx, ry, rz)


# PCA / 평면 처리
def compute_rotation_angle_from_pca(pca_v1_cam, R_base_to_cam):
     
    # 1. PCA 벡터를 로봇 베이스 좌표계로 회전 변환
    # V_base = R_base_cam * V_cam
    pca_v1_base = R_base_to_cam @ pca_v1_cam

    # 2. XY 평면에서의 각도 계산 (arctan2)
    # 주축 벡터 (vx, vy)
    vx, vy = pca_v1_base[0], pca_v1_base[1]

    # 각도 계산 (Rad -> Deg)
    angle_deg = np.degrees(np.arctan2(vy, vx))

    # 3. 각도 정규화 (-180 ~ 180)
    if angle_deg > 180:
        angle_deg -= 360
    elif angle_deg < -180:
        angle_deg += 360

    return angle_deg

def process_class1_pointcloud(pcd, max_angular_distance=5):
    
    print("\nStarting Class 1 PCA-informed RANSAC...")

    # 1. 클러스터링 및 가장 큰 그룹 선택 (DBSCAN 사용)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error) as vcm:
        labels = np.array(pcd.cluster_dbscan(eps=0.005, min_points=200, print_progress=False))

    if len(labels) == 0:
        print("Clustering failed or resulted in zero clusters.")
        return None, None, None, None

    # 가장 큰 클러스터 ID 찾기
    unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
    if len(unique_labels) == 0:
        print("No substantial cluster found (only noise).")
        return None, None, None, None

    largest_cluster_id = unique_labels[np.argmax(counts)]
    indices = np.where(labels == largest_cluster_id)[0]
    pcd_filtered = pcd.select_by_index(indices)

    print(f"Filtered to largest cluster ({len(indices)} points).")

    # 2. PCA 계산 (법선 추정)
    points = np.asarray(pcd_filtered.points)
    center = np.mean(points, axis=0)

    centered_points = points - center
    cov_matrix = np.cov(centered_points, rowvar=False)

    eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix)

    sort_indices = np.argsort(eigen_values)
    pca_v3_normal = eigen_vectors[:, sort_indices[0]]  # 법선
    pca_v1 = eigen_vectors[:, sort_indices[2]]         # 평면 내 주축

    print(f"PCA Normal (V3) calculated: {pca_v3_normal}")

    # 3. RANSAC 평면 피팅
    distance_threshold = 0.005
    max_iterations = 1000

    plane_model, inliers = pcd_filtered.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=3,
        num_iterations=max_iterations
    )

    [A, B, C, D] = plane_model
    ransac_normal = np.array([A, B, C])

    # PCA 법선과 방향 맞추기
    if np.dot(ransac_normal, pca_v3_normal) < 0:
        ransac_normal = -ransac_normal
        A, B, C = ransac_normal

    print(f"RANSAC Final Normal: {[A, B, C]}")

    return center, ransac_normal, pca_v1, pcd_filtered


def save_pointcloud_from_all(point_cloud, filename):
    if point_cloud.get_memory_type() == sl.MEM.GPU:
        import cupy as cp
        data_np = cp.asnumpy(cp.asarray(point_cloud.get_data(memory_type=sl.MEM.GPU, deep_copy=False)))
    else:
        data_np = point_cloud.get_data(memory_type=sl.MEM.CPU, deep_copy=False)

    all_points = data_np.reshape(-1, 4)
    valid_mask = ~np.isnan(all_points[:, 0]) & ~np.isnan(all_points[:, 1]) & ~np.isnan(all_points[:, 2])
    valid_points = all_points[valid_mask, :3]

    if len(valid_points) == 0:
        print("No valid points found in the entire point cloud.")
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(valid_points)

    o3d.io.write_point_cloud(filename, pcd)
    print(f"Saved full point cloud ({len(valid_points)} points) → {filename}")


if gl.GPU_ACCELERATION_AVAILABLE:
    import cupy as cp

lock = Lock()
run_signal = False
exit_signal = False
image_net = None
detections = []
robot_busy = False  # 로봇 동작 중 여부 플래그
pallet_count = 0

def save_pointcloud_from_bbox(point_cloud, bbox, filename, image_resolution, point_cloud_resolution):
    # 1. 스케일링 팩터 계산
    scale_x = point_cloud_resolution.width / image_resolution.width
    scale_y = point_cloud_resolution.height / image_resolution.height

    # 2. bbox를 포인트클라우드 해상도로 스케일
    bbox_scaled = bbox.copy()
    bbox_scaled[:, 0] = bbox[:, 0] * scale_x
    bbox_scaled[:, 1] = bbox[:, 1] * scale_y

    x_min_scaled = int(max(0, min(bbox_scaled[:, 0])))
    x_max_scaled = int(min(point_cloud_resolution.width, max(bbox_scaled[:, 0])))
    y_min_scaled = int(max(0, min(bbox_scaled[:, 1])))
    y_max_scaled = int(min(point_cloud_resolution.height, max(bbox_scaled[:, 1])))

    points = []

    for y in range(y_min_scaled, y_max_scaled):
        for x in range(x_min_scaled, x_max_scaled):
            success, value = point_cloud.get_value(x, y)
            if success == sl.ERROR_CODE.SUCCESS:
                X, Y, Z, _ = value
                if not np.isnan(X) and not np.isnan(Y) and not np.isnan(Z):
                    points.append([X, Y, Z])

    if len(points) == 0:
        print("No valid points found in bbox after scaling.")
        return None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))

    o3d.io.write_point_cloud(filename, pcd)
    print(f"Saved point cloud → {filename}")
    return pcd


def depth_to_color(z, z_min=0.2, z_max=5.0):
    z = np.clip((z - z_min) / (z_max - z_min), 0.0, 1.0)
    r = (z * 255).astype(np.uint8)
    g = ((1 - z) * 255).astype(np.uint8)
    b = np.full_like(r, 0)
    return (b, g, r)


def xywh2abcd(xywh):
    output = np.zeros((4, 2))

    x_min = max(0, xywh[0] - 0.5 * xywh[2])
    x_max = (xywh[0] + 0.5 * xywh[2])
    y_min = max(0, xywh[1] - 0.5 * xywh[3])
    y_max = (xywh[1] + 0.5 * xywh[3])

    output[0][0] = x_min
    output[0][1] = y_min

    output[1][0] = x_max
    output[1][1] = y_min

    output[2][0] = x_max
    output[2][1] = y_max

    output[3][0] = x_min
    output[3][1] = y_max
    return output


def detections_to_custom_box(detections):
    output = []
    for det in detections:
        xywh = det.xywh[0]
        obj = sl.CustomBoxObjectData()
        obj.bounding_box_2d = xywh2abcd(xywh)
        obj.label = det.cls
        obj.probability = det.conf
        obj.is_grounded = False
        output.append(obj)
    return output


def torch_thread(weights, img_size, conf_thres=0.2, iou_thres=0.45):
    global image_net, exit_signal, run_signal, detections

    print("Intializing Network...")
    model = YOLO(weights)

    while not exit_signal:
        if run_signal:
            lock.acquire()

            if gl.GPU_ACCELERATION_AVAILABLE:
                img_cupy = cp.asarray(image_net)[:, :, :3]
                img = cp.asnumpy(img_cupy)
            else:
                img = cv2.cvtColor(image_net, cv2.COLOR_RGBA2RGB)

            det = model.predict(
                img,
                save=False,
                imgsz=img_size,
                conf=conf_thres,
                iou=iou_thres,
                verbose=False
            )[0].cpu().numpy().boxes

            detections = detections_to_custom_box(det)
            lock.release()
            run_signal = False
        sleep(0.005)


def start_detection_thread(opt):
    capture_thread = Thread(
        target=torch_thread,
        kwargs={
            'weights': opt.weights,
            'img_size': opt.img_size,
            'conf_thres': opt.conf_thres
        }
    )
    capture_thread.start()
    return capture_thread


def get_memory_type(opt):
    use_gpu = gl.GPU_ACCELERATION_AVAILABLE and not opt.disable_gpu_data_transfer
    mem_type = sl.MEM.GPU if use_gpu else sl.MEM.CPU

    if use_gpu:
        print("Using GPU data transfer with CuPy")
    else:
        print("Using CPU data transfer")
    return mem_type


def initialize_camera_and_viewers(opt, mem_type):
    print("Initializing Camera...")

    zed = sl.Camera()

    input_type = sl.InputType()
    if opt.svo is not None:
        input_type.set_from_svo_file(opt.svo)

    init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=True)
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE
    init_params.depth_maximum_distance = 50
    init_params.depth_stabilization = 30

    runtime_params = sl.RuntimeParameters()
    status = zed.open(init_params)

    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        sys.exit(1)

    image_left_tmp = sl.Mat(0, 0, sl.MAT_TYPE.U8_C4, mem_type)

    print("Initialized Camera")

    positional_tracking_parameters = sl.PositionalTrackingParameters()
    zed.enable_positional_tracking(positional_tracking_parameters)

    obj_param = sl.ObjectDetectionParameters()
    obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
    obj_param.enable_tracking = True
    obj_param.enable_segmentation = False
    zed.enable_object_detection(obj_param)

    objects = sl.Objects()
    obj_runtime_param = sl.CustomObjectDetectionRuntimeParameters()

    camera_infos = zed.get_camera_information()
    camera_res = camera_infos.camera_configuration.resolution

    viewer = gl.GLViewer()
    point_cloud_res = sl.Resolution(min(camera_res.width, 720), min(camera_res.height, 404))
    viewer.init(camera_infos.camera_model, point_cloud_res, obj_param.enable_tracking)
    point_cloud = sl.Mat(point_cloud_res.width, point_cloud_res.height, sl.MAT_TYPE.F32_C4, mem_type)
    image_left = sl.Mat(0, 0, sl.MAT_TYPE.U8_C4, mem_type)

    display_resolution = sl.Resolution(min(camera_res.width, 1280), min(camera_res.height, 720))
    image_scale = [display_resolution.width / camera_res.width, display_resolution.height / camera_res.height]
    image_left_ocv = np.full((display_resolution.height, display_resolution.width, 4),
                             [245, 239, 239, 255], np.uint8)

    camera_config = camera_infos.camera_configuration
    tracks_resolution = sl.Resolution(400, display_resolution.height)
    track_view_generator = cv_viewer.TrackingViewer(tracks_resolution, camera_config.fps,
                                                    init_params.depth_maximum_distance)
    track_view_generator.set_camera_calibration(camera_config.calibration_parameters)
    image_track_ocv = np.zeros((tracks_resolution.height, tracks_resolution.width, 4), np.uint8)

    cam_w_pose = sl.Pose()

    ctx = {
        "zed": zed,
        "runtime_params": runtime_params,
        "image_left_tmp": image_left_tmp,
        "obj_param": obj_param,
        "objects": objects,
        "obj_runtime_param": obj_runtime_param,
        "camera_infos": camera_infos,
        "camera_res": camera_res,
        "viewer": viewer,
        "point_cloud_res": point_cloud_res,
        "point_cloud": point_cloud,
        "image_left": image_left,
        "display_resolution": display_resolution,
        "image_scale": image_scale,
        "image_left_ocv": image_left_ocv,
        "tracks_resolution": tracks_resolution,
        "track_view_generator": track_view_generator,
        "image_track_ocv": image_track_ocv,
        "cam_w_pose": cam_w_pose,
    }

    return ctx


def colorize_bbox_depth(objects, point_cloud, image_left_ocv):
    for obj in objects.object_list:
        if obj.label == 1 or obj.label == 0:
            bbox = np.array(obj.bounding_box_2d).astype(int)

            x_min = max(0, np.min(bbox[:, 0]))
            x_max = min(image_left_ocv.shape[1] - 1, np.max(bbox[:, 0]))
            y_min = max(0, np.min(bbox[:, 1]))
            y_max = min(image_left_ocv.shape[0] - 1, np.max(bbox[:, 1]))

            for y in range(y_min, y_max):
                for x in range(x_min, x_max):
                    success, value = point_cloud.get_value(x, y)
                    if success == sl.ERROR_CODE.SUCCESS:
                        X, Y, Z, _ = value
                        if not np.isnan(Z) and Z > 0:
                            b, g, r = depth_to_color(Z)
                            cv2.circle(image_left_ocv, (x, y), 1, (int(b), int(g), int(r)), -1)


# 카메라 → 로봇 좌표 변환 + 자세 계산

def compute_robot_pose_from_pallet(center_cam, normal_cam, pca_v1_cam,
                                   R_cam_to_robot, t_cam_to_robot):
    print("\n[DEBUG] center_cam (m):", center_cam)
    print("[DEBUG] normal_cam:", normal_cam)
    print("[DEBUG] pca_v1_cam:", pca_v1_cam)

    # z축: 팔렛 노멀
    z_axis_c = normal_cam / np.linalg.norm(normal_cam)

    # x축: PCA V1 (평면 내 방향)
    x_axis_c = pca_v1_cam / np.linalg.norm(pca_v1_cam)

    # x, z 축이 수직이 아니면, z에 대해 정직교화
    x_axis_c = x_axis_c - np.dot(x_axis_c, z_axis_c) * z_axis_c
    x_axis_c = x_axis_c / np.linalg.norm(x_axis_c)

    # y축 = z × x  (오른손 좌표계)
    y_axis_c = np.cross(z_axis_c, x_axis_c)
    y_axis_c = y_axis_c / np.linalg.norm(y_axis_c)

    # 카메라 좌표계에서의 회전 행렬
    R_cam_pallet = np.column_stack((x_axis_c, y_axis_c, z_axis_c))  # 3x3

    # 로봇 좌표계로 변환
    center_r = R_cam_to_robot @ center_cam + t_cam_to_robot
    R_robot_pallet = R_cam_to_robot @ R_cam_pallet

    print("[DEBUG] R_cam_to_robot:\n", R_cam_to_robot)
    print("[DEBUG] t_cam_to_robot (m):", t_cam_to_robot)
    print("[DEBUG] center_robot (m):", center_r)
    print("[DEBUG] R_robot_pallet:\n", R_robot_pallet)

    r = R.from_matrix(R_robot_pallet)
    rx, ry, rz = r.as_euler("zyx", degrees=True)[::-1]

    # 위치는 m → mm (로봇 API 단위에 맞게)
    x, y, z = center_r * 1000.0

    pose = [x, y, z, rx, ry, rz]
    print(f"Robot pose from pallet (x,y,z,rx,ry,rz): {pose}")
    return pose


def process_and_save_topmost_class1(objects, point_cloud, camera_infos, point_cloud_res, image_left_ocv):
    
    # s 키 눌렀을 때:
    # - 가장 위 팔렛(Class 1) 선택
    # - bbox 포인트클라우드 저장
    # - 전체 포인트클라우드 저장
    # - PCA/RANSAC로 center, normal, pca_v1 계산
    # - bbox 이미지 crop 저장
    # - (center, normal, pca_v1) 반환
    
    print("Saving point cloud and processing for the TOPMOST CLASS 1 object...")

    original_res = camera_infos.camera_configuration.resolution

    # 최상단 객체 찾기
    topmost_obj = None
    min_y_min = float('inf')

    for obj in objects.object_list:
        if obj.raw_label == 1 or obj.label == 1:
            bbox_2d = np.array(obj.bounding_box_2d)
            y_min = np.min(bbox_2d[:, 1])
            if y_min < min_y_min:
                min_y_min = y_min
                topmost_obj = obj

    if topmost_obj is None:
        print("No object with class ID 1 detected.")
        return None

    bbox_2d = np.array(topmost_obj.bounding_box_2d)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. 포인트클라우드 저장 + O3D PCD 얻기
    filename_pcd_bbox = f"pointcloud_class1_BBOX_{timestamp}_pallet{pallet_count}.pcd"
    pcd_object = save_pointcloud_from_bbox(
        point_cloud, bbox_2d, filename_pcd_bbox,
        original_res, point_cloud_res
    )
    save_pointcloud_from_all(point_cloud, f"pointcloud_class1_{timestamp}_pallet{pallet_count}.pcd")

    center = normal = pca_v1 = None
    if pcd_object is not None:
        center, normal, pca_v1, pcd_filtered = process_class1_pointcloud(pcd_object)

        if center is not None:
            print("FINAL CLASS 1 PROCESSING RESULTS 🌟")
            print(f"Center Point (cam frame, m): {center}")
            print(f"Normal Vector (RANSAC/PCA): {normal}")
            print(f"Principal Axis (PCA V1): {pca_v1}")

    # 3. bbox 이미지 저장
    bbox_int = bbox_2d.astype(int)
    x_min = max(0, np.min(bbox_int[:, 0]))
    x_max = min(image_left_ocv.shape[1], np.max(bbox_int[:, 0]))
    y_min = max(0, np.min(bbox_int[:, 1]))
    y_max = min(image_left_ocv.shape[0], np.max(bbox_int[:, 1]))

    cropped_image = image_left_ocv[y_min:y_max, x_min:x_max]

    if cropped_image.size > 0:
        filename_img = f"image_class1_{timestamp}_pallet{pallet_count}.png"
        cv2.imwrite(filename_img, cropped_image)
        print(f"Saved image crop → {filename_img}")
    else:
        print("Cropped image is empty.")

    if center is None:
        return None
    return center, normal, pca_v1


# 프레임 단위 처리 + 로봇 연동

def process_single_frame(ctx, mem_type,
                         robot=None,
                         T_cam_to_gripper=None,
                         home_pose=None,
                         release_pose=None):

    global image_net, exit_signal, run_signal, detections, robot_busy

    zed = ctx["zed"]
    runtime_params = ctx["runtime_params"]
    image_left_tmp = ctx["image_left_tmp"]
    obj_param = ctx["obj_param"]
    objects = ctx["objects"]
    obj_runtime_param = ctx["obj_runtime_param"]
    camera_infos = ctx["camera_infos"]
    point_cloud_res = ctx["point_cloud_res"]
    point_cloud = ctx["point_cloud"]
    image_left = ctx["image_left"]
    display_resolution = ctx["display_resolution"]
    image_scale = ctx["image_scale"]
    image_left_ocv = ctx["image_left_ocv"]
    track_view_generator = ctx["track_view_generator"]
    image_track_ocv = ctx["image_track_ocv"]
    cam_w_pose = ctx["cam_w_pose"]

    # Grab
    if zed.grab(runtime_params) > sl.ERROR_CODE.SUCCESS:
        exit_signal = True
        return False

    # YOLO 입력용 이미지
    lock.acquire()
    zed.retrieve_image(image_left_tmp, sl.VIEW.LEFT, mem_type)
    image_net = image_left_tmp.get_data(memory_type=mem_type, deep_copy=False)
    lock.release()
    run_signal = True

    while run_signal:
        sleep(0.001)

    lock.acquire()
    zed.ingest_custom_box_objects(detections)
    lock.release()
    zed.retrieve_custom_objects(objects, obj_runtime_param)

    zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, mem_type, point_cloud_res)
    zed.retrieve_image(image_left, sl.VIEW.LEFT, mem_type, display_resolution)
    zed.get_position(cam_w_pose, sl.REFERENCE_FRAME.WORLD)

    ctx["viewer"].updateData(point_cloud, objects)

    if mem_type == sl.MEM.GPU:
        ctx["image_left_ocv"][:] = cp.asnumpy(cp.asarray(image_left.get_data(memory_type=mem_type, deep_copy=False)))
    else:
        np.copyto(image_left_ocv, image_left.get_data(memory_type=mem_type, deep_copy=False))

    cv_viewer.render_2D(image_left_ocv, image_scale, objects, obj_param.enable_tracking)

    colorize_bbox_depth(objects, point_cloud, image_left_ocv)

    track_view_generator.generate_view(objects, cam_w_pose, image_track_ocv, objects.is_tracked)

    global_image = cv2.hconcat([image_left_ocv, image_track_ocv])

    # 로봇 상태 텍스트
    status_text = "Robot: BUSY" if robot_busy else "Robot: READY"
    cv2.putText(global_image, status_text, (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0, 0, 255) if robot_busy else (0, 255, 0), 2)

    cv2.imshow("ZED | 2D View and Birds View", global_image)
    key = cv2.waitKey(1)

    if key == ord('s') or key == ord('S'):
        # 팔렛 자세 계산
        result = process_and_save_topmost_class1(
            objects, point_cloud, camera_infos, point_cloud_res, image_left_ocv
        )

        if result is not None and robot is not None and not robot_busy:
            center_cam, normal_cam, pca_v1_cam = result

            # [Step 1] Camera -> Gripper
            R_cg = T_cam_to_gripper[:3, :3]
            t_cg = T_cam_to_gripper[:3, 3]

            center_gripper = (R_cg @ center_cam) + t_cg
            pca1_gripper = R_cg @ pca_v1_cam

            # [Step 2] Gripper -> Base
            pose_msg = robot.piper.GetArmEndPoseMsgs().end_pose

            x_base = pose_msg.X_axis / 1000000.0
            y_base = pose_msg.Y_axis / 1000000.0
            z_base = pose_msg.Z_axis / 1000000.0
            rx_base = pose_msg.RX_axis / 1000.0
            ry_base = pose_msg.RY_axis / 1000.0
            rz_base = pose_msg.RZ_axis / 1000.0

            R_bg = R.from_euler("xyz", [rx_base, ry_base, rz_base], degrees=True).as_matrix()
            t_bg = np.array([x_base, y_base, z_base])

            # [Step 3] Final Base Coordinates & Vector
            center_base = (R_bg @ center_gripper) + t_bg
            pca1_base = R_bg @ pca1_gripper

            # [Step 4] PCA Yaw Calculation
            yaw_deg = np.degrees(np.arctan2(pca1_base[1], pca1_base[0]))

            # Offset: 90(직각 잡기) or 0(나란히 잡기)
            final_rz = abs(yaw_deg) - 90
            print("yaw_deg: ", yaw_deg)

            # Normalize -180 ~ 180
            if final_rz > 270:
                final_rz -= 180
            elif final_rz < 90:
                final_rz += 180
            print("final_rz: ", final_rz)

            # [Step 5] Target Generation (mm)
            target_x = center_base[0] * 1000.0
            target_y = center_base[1] * 1000.0
            target_z = center_base[2] * 1000.0

            print(f"Target(mm): {target_x:.1f}, {target_y:.1f}, {target_z:.1f} | Yaw: {final_rz:.1f}")

            try:
                # 1. 카메라 내부 파라미터(Intrinsics) 가져오기
                calib = ctx["camera_infos"].camera_configuration.calibration_parameters.left_cam
                fx, fy = calib.fx, calib.fy
                cx, cy = calib.cx, calib.cy

                # 3D 좌표(m) -> 2D 픽셀(u, v) 변환 함수
                def project_to_pixel(pt3d):
                    x, y, z = pt3d
                    if z <= 0: return (0, 0)  # 뒤에 있는 점 예외처리
                    u = int((x * fx / z) + cx)
                    v = int((y * fy / z) + cy)
                    return (u, v)

                # 2. 그리기 시작점(물체 중심)과 끝점(PCA 방향) 계산
                # 끝점은 중심에서 PCA 벡터 방향으로 10cm(0.1m) 떨어진 곳
                pt_start = project_to_pixel(center_cam)
                pt_end = project_to_pixel(center_cam + pca_v1_cam * 0.1)

                # 3. 이미지에 그리기 (image_left_ocv 원본에 덧칠)
                # 빨간색 화살표 (PCA 주축)
                cv2.arrowedLine(image_left_ocv, pt_start, pt_end, (0, 0, 255), 3, tipLength=0.3)
                # 1) final_rz(deg) → 로봇 Base 벡터
                yaw_rad = np.radians(final_rz)
                v_base = np.array([np.cos(yaw_rad), np.sin(yaw_rad), 0.0])  # XY 평면

                # 2) Base → Gripper → Camera 변환
                v_gripper = R_bg.T @ v_base  # Base → Gripper
                v_cam = R_cg.T @ v_gripper  # Gripper → Camera

                # 3) 카메라 벡터를 픽셀 좌표로 변환
                arrow_len = 0.1  # meter

                pt_start = project_to_pixel(center_cam)
                pt_finalrz = project_to_pixel(center_cam + v_cam * arrow_len)

                # 4) 이미지에 final_rz 방향 그리기
                cv2.arrowedLine(image_left_ocv, pt_start, pt_finalrz, (255, 0, 0), 3, tipLength=0.3)
                cv2.putText(image_left_ocv, "RobotYaw", (pt_finalrz[0], pt_finalrz[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                # 노란색 텍스트 (계산된 로봇 각도)
                text = f"Angle: {final_rz:.1f} deg"
                cv2.putText(image_left_ocv, text, (pt_start[0] + 10, pt_start[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                # 4. 확인용 팝업창 띄우기 (로봇 움직이는 동안 확인 가능)
                cv2.imshow("Detected Angle Check", image_left_ocv)
                filename_img = f"vector_image_class1__pallet{pallet_count}.png"
                cv2.imwrite(filename_img, image_left_ocv)
                cv2.waitKey(1)
            except Exception as e:
                print(f"Visualization Error: {e}")

            gripper_offset = 130
            raw_z = target_z + gripper_offset
            corrected_z = snap_height_to_nearest_level(raw_z)

            print("원본 Z =", raw_z)
            print("보정된 Z =", corrected_z)

            pallet_pose = [target_x + 20, target_y + 20, corrected_z, -180, 0, final_rz]

            def robot_job():
                global robot_busy, pallet_count
                robot_busy = True
                try:
                    base_z = 139.9  # 1층 높이
                    thickness = 20
                    current_stack_z = base_z + (pallet_count * thickness)

                    current_release_pose = list(release_pose)
                    current_release_pose[2] = current_stack_z

                    print(f"Stacking Pallet #{pallet_count + 1} at Z={current_stack_z}mm")
                    print("Moving to pallet...")
                    print("[DEBUG] pallet_pose:", pallet_pose)

                    # Side 접근 방향 벡터 계산
                    side_offset = 150  # mm (조절 가능)
                    yaw_rad = np.radians(final_rz)
                    dx = np.cos(yaw_rad)
                    dy = np.sin(yaw_rad)

                    # pallet_pose 기준으로 side offset 적용
                    side = pallet_pose.copy()
                    side[0] += dx * side_offset
                    side[1] += dy * side_offset

                    print(f"[DEBUG] Side approach pose = {side}")

                    # Side 접근
                    robot.move_l(*side)
                    time.sleep(3)

                    # 팔렛 방향으로 직선 접근
                    pallet_pose[0] += dx * 30
                    pallet_pose[1] += dy * 30
                    robot.move_l(*pallet_pose)
                    time.sleep(2.5)

                    # 그리퍼로 잡기
                    robot.grip_grab()
                    time.sleep(1.5)

                    # 위로 이동
                    above = pallet_pose.copy()
                    above[2] += 50
                    robot.move_l(*above)
                    time.sleep(1.5)

                    # Release stack으로 이동
                    # 1) Release location 바로 위로 이동
                    above_current_release_pose = current_release_pose.copy()
                    above_current_release_pose[2] += 50  # +50mm 위
                    robot.move_p(*above_current_release_pose)
                    time.sleep(2)

                    # 2) 층의 정확한 release 위치까지 내려가기
                    robot.move_l(*current_release_pose)
                    time.sleep(1.5)

                    # 3) 안정적 놓기 위해 추가로 4mm 더 내리기
                    lower_pose = current_release_pose.copy()
                    lower_pose[2] -= 4  # 아래로 4mm
                    robot.move_l(*lower_pose)
                    time.sleep(1.5)

                    robot.grip_close()
                    time.sleep(1.5)

                    # 4) 다시 정확한 층 높이로 올라오기
                    robot.move_l(*current_release_pose)
                    time.sleep(1.5)

                    # 5) 뒤로 빠지기 (x축 기준으로 -70mm)
                    back_pose = current_release_pose.copy()
                    back_pose[0] -= 70
                    robot.move_l(*back_pose)
                    time.sleep(1.5)

                    # 6) 뒤로 뺀 위치에서 위로 +50 올라가기
                    above_back_pose = back_pose.copy()
                    above_back_pose[2] += 50
                    robot.move_l(*above_back_pose)
                    time.sleep(2.0)

                    robot.move_p(*home_pose)

                    pallet_count += 1
                    print(f" pallet_count: {pallet_count}")
                    print("Pallet picked & placed.")
                except Exception as e:
                    print(f"Robot error: {e}")
                robot_busy = False

            threading.Thread(target=robot_job, daemon=True).start()
        else:
            if robot is None:
                print("Robot is not initialized. Only pose is computed.")

    if key == 27 or key == ord('q') or key == ord('Q'):
        end_pose = [60, 0, 200, 0, 85, 0]
        robot.move_p(*end_pose)
        exit_signal = True

    return True


# 전체 실행 루프

def run(opt):
    global exit_signal

    mem_type = get_memory_type(opt)
    capture_thread = start_detection_thread(opt)
    ctx = initialize_camera_and_viewers(opt, mem_type)

    viewer = ctx["viewer"]
    zed = ctx["zed"]

    # 로봇 초기화
    print("Connecting PiPER robot...")
    piper = C_PiperInterface("can0")
    piper.ConnectPort()
    robot = PiPER_Palletizing(piper)

    pose = piper.GetArmEndPoseMsgs().end_pose
    print("=========== RAW END POSE ===========")
    print(pose)
    print("X_axis:", pose.X_axis)
    print("Y_axis:", pose.Y_axis)
    print("Z_axis:", pose.Z_axis)

    # 홈 / 릴리즈 포즈
    # home_pose = [221.5, 0, 299.3, 180 ,0, 180]
    home_pose = [221.5, 0, 309.3, 174.8, 14, 180]
    release_pose = [182, -128, 139.9, -180, 0, -180]

    # Hand–Eye Calibration 결과: camera → gripper
    R_cam_to_gripper = np.array([
                                    [0.06530568,  0.99408768,  0.08674586],
                                    [-0.99784175,  0.06565429, -0.00116878],
                                    [-0.00685711, -0.08648231,    0.99622979]
    ])

    t_cam_to_gripper = np.array([-0.04087455,
                                 0.04529892,
                                 0.04977192])  # [m]

    T_cam_to_gripper = np.eye(4)
    T_cam_to_gripper[:3, :3] = R_cam_to_gripper
    T_cam_to_gripper[:3, 3] = t_cam_to_gripper

    print("T_cam_to_gripper (Hand–Eye result):\n", T_cam_to_gripper)

    # 시작 시 홈 포즈로 이동
    robot.move_p(*home_pose)

    while viewer.is_available() and not exit_signal:
        if not process_single_frame(
            ctx, mem_type,
            robot=robot,
            T_cam_to_gripper=T_cam_to_gripper,
            home_pose=home_pose,
            release_pose=release_pose
        ):
            break

    viewer.exit()
    exit_signal = True
    zed.close()
    capture_thread.join()


# 진입점

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str,
                        default=r'/home/hada/piper_ws/src/piper_sdk/pytorch_yolo/best.pt')
    parser.add_argument('--svo', type=str, default=None,
                        help='optional svo file, if not passed, use the plugged camera instead')
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--disable-gpu-data-transfer', action='store_true',
                        help='Disable GPU data transfer acceleration with CuPy even if CuPy is available')
    opt = parser.parse_args()

    with torch.no_grad():
        run(opt)


if __name__ == '__main__':
    main()
