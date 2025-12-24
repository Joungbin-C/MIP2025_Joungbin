#!/usr/bin/env python3

import sys
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
### ADDED ###
import datetime
import open3d as o3d


# =========================================================================
# === Class 1 (Pallet Top) Processing Logic (using Open3D) ===
# =========================================================================

def process_class1_pointcloud(pcd, max_angular_distance=5):
    """
    Class 1 (팔레트 윗면) 포인트 클라우드에 대해 클러스터링, PCA 기반 RANSAC을 수행합니다.
    """
    print("\n⚙️ Starting Class 1 PCA-informed RANSAC...")

    # 1. 클러스터링 및 가장 큰 그룹 선택 (DBSCAN 사용)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error) as vcm:
        labels = np.array(pcd.cluster_dbscan(eps=0.005, min_points=100, print_progress=False))

    if len(labels) == 0:
        print("⚠ Clustering failed or resulted in zero clusters.")
        return None, None, None, None

    # 가장 큰 클러스터 ID 찾기
    unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
    if len(unique_labels) == 0:
        print("⚠ No substantial cluster found (only noise).")
        return None, None, None, None

    largest_cluster_id = unique_labels[np.argmax(counts)]
    indices = np.where(labels == largest_cluster_id)[0]
    pcd_filtered = pcd.select_by_index(indices)

    print(f"✅ Filtered to largest cluster ({len(indices)} points).")

    # 2. PCA 계산 (법선 추정)
    points = np.asarray(pcd_filtered.points)
    center = np.mean(points, axis=0)

    # 공분산 행렬 계산
    centered_points = points - center
    cov_matrix = np.cov(centered_points, rowvar=False)

    # 고유값 및 고유 벡터 계산
    eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix)

    # 고유값 오름차순 정렬 (가장 작은 고유값 -> V3, 법선 벡터)
    sort_indices = np.argsort(eigen_values)
    pca_v3_normal = eigen_vectors[:, sort_indices[0]]  # V3 (PCA Normal)
    pca_v1 = eigen_vectors[:, sort_indices[2]]         # V1 (Principal Axis)

    print(f"✅ PCA Normal (V3) calculated: {pca_v3_normal}")

    # 3. PCA 기반 RANSAC 피팅
    distance_threshold = 0.005  # MAX_RANSAC_DISTANCE
    max_iterations = 1000

    plane_model, inliers = pcd_filtered.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=3,
        num_iterations=max_iterations
    )

    [A, B, C, D] = plane_model
    ransac_normal = np.array([A, B, C])

    # PCA 법선과 RANSAC 법선의 방향을 일치시킵니다.
    if np.dot(ransac_normal, pca_v3_normal) < 0:
        ransac_normal = -ransac_normal
        A, B, C = ransac_normal

    print(f"✅ RANSAC Final Normal: {[A, B, C]}")

    return center, ransac_normal, pca_v1, pcd_filtered


# =========================================================================
# === Point Cloud I/O and Utility Functions ===
# =========================================================================

def save_pointcloud_from_all(point_cloud, filename):
    # 기존 코드 유지
    if point_cloud.get_memory_type() == sl.MEM.GPU:
        import cupy as cp
        data_np = cp.asnumpy(cp.asarray(point_cloud.get_data(memory_type=sl.MEM.GPU, deep_copy=False)))
    else:
        data_np = point_cloud.get_data(memory_type=sl.MEM.CPU, deep_copy=False)

    all_points = data_np.reshape(-1, 4)
    valid_mask = ~np.isnan(all_points[:, 0]) & ~np.isnan(all_points[:, 1]) & ~np.isnan(all_points[:, 2])
    valid_points = all_points[valid_mask, :3]

    if len(valid_points) == 0:
        print("⚠ No valid points found in the entire point cloud.")
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(valid_points)

    o3d.io.write_point_cloud(filename, pcd)
    print(f"✅ Saved full point cloud ({len(valid_points)} points) → {filename}")


# CuPy 사용 여부
if gl.GPU_ACCELERATION_AVAILABLE:
    import cupy as cp

# 전역 상태들 (스레드/Grab 루프 공유)
lock = Lock()
run_signal = False
exit_signal = False
image_net = None
detections = []


def save_pointcloud_from_bbox(point_cloud, bbox, filename, image_resolution, point_cloud_resolution):
    # 1. 스케일링 팩터 계산
    scale_x = point_cloud_resolution.width / image_resolution.width
    scale_y = point_cloud_resolution.height / image_resolution.height

    # 2. 바운딩 박스 좌표를 포인트 클라우드 해상도로 스케일링
    bbox_scaled = bbox.copy()
    bbox_scaled[:, 0] = bbox[:, 0] * scale_x
    bbox_scaled[:, 1] = bbox[:, 1] * scale_y

    # 3. 스케일링된 바운딩 박스로부터 영역 정의
    x_min_scaled = int(max(0, min(bbox_scaled[:, 0])))
    x_max_scaled = int(min(point_cloud_resolution.width, max(bbox_scaled[:, 0])))
    y_min_scaled = int(max(0, min(bbox_scaled[:, 1])))
    y_max_scaled = int(min(point_cloud_resolution.height, max(bbox_scaled[:, 1])))

    points = []

    # 4. 스케일링된 영역에서 포인트 추출
    for y in range(y_min_scaled, y_max_scaled):
        for x in range(x_min_scaled, x_max_scaled):
            success, value = point_cloud.get_value(x, y)
            if success == sl.ERROR_CODE.SUCCESS:
                X, Y, Z, _ = value
                if not np.isnan(X) and not np.isnan(Y) and not np.isnan(Z):
                    points.append([X, Y, Z])

    if len(points) == 0:
        print("⚠ No valid points found in bbox after scaling.")
        return None

    # Save to PCD
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))

    o3d.io.write_point_cloud(filename, pcd)
    print(f"✅ Saved point cloud → {filename}")
    return pcd


def depth_to_color(z, z_min=0.2, z_max=5.0):
    """
    Z-depth -> RGB color (blue=near, red=far)
    """
    z = np.clip((z - z_min) / (z_max - z_min), 0.0, 1.0)
    r = (z * 255).astype(np.uint8)
    g = ((1 - z) * 255).astype(np.uint8)
    b = np.full_like(r, 0)
    return (b, g, r)


def xywh2abcd(xywh):
    output = np.zeros((4, 2))

    # Center / Width / Height -> BBox corners coordinates
    x_min = max(0, xywh[0] - 0.5 * xywh[2])
    x_max = (xywh[0] + 0.5 * xywh[2])
    y_min = max(0, xywh[1] - 0.5 * xywh[3])
    y_max = (xywh[1] + 0.5 * xywh[3])

    # A ------ B
    # | Object |
    # D ------ C

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

        # Creating ingestable objects for the ZED SDK
        obj = sl.CustomBoxObjectData()
        obj.bounding_box_2d = xywh2abcd(xywh)
        obj.label = det.cls
        obj.probability = det.conf
        obj.is_grounded = False
        output.append(obj)
    return output


# =========================================================================
# === YOLO 추론 스레드 ===
# =========================================================================

def torch_thread(weights, img_size, conf_thres=0.2, iou_thres=0.45):
    global image_net, exit_signal, run_signal, detections

    print("Intializing Network...")

    model = YOLO(weights)

    while not exit_signal:
        if run_signal:
            lock.acquire()

            if gl.GPU_ACCELERATION_AVAILABLE:
                img_cupy = cp.asarray(image_net)[:, :, :3]  # Remove alpha channel on GPU
                img = cp.asnumpy(img_cupy)
            else:
                img = cv2.cvtColor(image_net, cv2.COLOR_RGBA2RGB)

            det = model.predict(img, save=False, imgsz=img_size, conf=conf_thres, iou=iou_thres, verbose=False)[
                0].cpu().numpy().boxes

            # ZED CustomBox format
            detections = detections_to_custom_box(det)
            lock.release()
            run_signal = False
        sleep(0.005)


def start_detection_thread(opt):
    """YOLO 추론 스레드를 시작하고 Thread 객체를 반환"""
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


# =========================================================================
# === ZED / Viewer 초기화 ===
# =========================================================================

def get_memory_type(opt):
    """GPU / CPU 메모리 타입 결정"""
    use_gpu = gl.GPU_ACCELERATION_AVAILABLE and not opt.disable_gpu_data_transfer
    mem_type = sl.MEM.GPU if use_gpu else sl.MEM.CPU

    if use_gpu:
        print("🚀 Using GPU data transfer with CuPy")
    else:
        print("💻 Using CPU data transfer")
    return mem_type


def initialize_camera_and_viewers(opt, mem_type):
    """카메라, 뷰어, 관련 객체들 전부 초기화하고 dict로 반환"""
    print("Initializing Camera...")

    zed = sl.Camera()

    input_type = sl.InputType()
    if opt.svo is not None:
        input_type.set_from_svo_file(opt.svo)

    # Init parameters
    init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=True)
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
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

    # Display
    camera_infos = zed.get_camera_information()
    camera_res = camera_infos.camera_configuration.resolution

    # OpenGL viewer
    viewer = gl.GLViewer()
    point_cloud_res = sl.Resolution(min(camera_res.width, 720), min(camera_res.height, 404))
    viewer.init(camera_infos.camera_model, point_cloud_res, obj_param.enable_tracking)
    point_cloud = sl.Mat(point_cloud_res.width, point_cloud_res.height, sl.MAT_TYPE.F32_C4, mem_type)
    image_left = sl.Mat(0, 0, sl.MAT_TYPE.U8_C4, mem_type)

    # 2D display
    display_resolution = sl.Resolution(min(camera_res.width, 1280), min(camera_res.height, 720))
    image_scale = [display_resolution.width / camera_res.width, display_resolution.height / camera_res.height]
    image_left_ocv = np.full((display_resolution.height, display_resolution.width, 4),
                             [245, 239, 239, 255], np.uint8)

    # Tracks view
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


# =========================================================================
# === 렌더링/처리용 헬퍼 ===
# =========================================================================

def colorize_bbox_depth(objects, point_cloud, image_left_ocv):
    """bbox 내부 픽셀에 깊이 기반 컬러 입히기"""
    for obj in objects.object_list:
        if obj.label == 1 or obj.label == 0:  # 클래스 1과 0 모두 시각화
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


def process_and_save_topmost_class1(objects, point_cloud, camera_infos, point_cloud_res, image_left_ocv):
    """s 키 입력 시: 가장 위(화면 기준) Pallet class1 하나 선택 후 저장/처리"""
    print("📥 Saving point cloud and processing for the TOPMOST CLASS 1 object...")

    original_res = camera_infos.camera_configuration.resolution
    pc_res = point_cloud_res

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
        print("⚠️ No object with class ID 1 detected.")
        return

    bbox_2d = np.array(topmost_obj.bounding_box_2d)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. 포인트클라우드 저장 + O3D PCD 얻기
    filename_pcd_bbox = f"pointcloud_class1_BBOX_{timestamp}.pcd"
    pcd_object = save_pointcloud_from_bbox(point_cloud, bbox_2d, filename_pcd_bbox, original_res, pc_res)
    save_pointcloud_from_all(point_cloud, f"pointcloud_class1_{timestamp}.pcd")  # 전체 클라우드 저장

    # 2. PCA / RANSAC 처리
    if pcd_object is not None:
        center, normal, pca_v1, pcd_filtered = process_class1_pointcloud(pcd_object)

        if center is not None:
            print("--------------------------------------------------")
            print("🌟 FINAL CLASS 1 PROCESSING RESULTS 🌟")
            print(f"Center Point: {center}")
            print(f"Normal Vector (RANSAC/PCA): {normal}")
            print(f"Principal Axis (PCA V1): {pca_v1}")
            print("--------------------------------------------------")

    # 3. Bounding Box 영역 이미지 저장
    bbox_int = bbox_2d.astype(int)
    x_min = max(0, np.min(bbox_int[:, 0]))
    x_max = min(image_left_ocv.shape[1], np.max(bbox_int[:, 0]))
    y_min = max(0, np.min(bbox_int[:, 1]))
    y_max = min(image_left_ocv.shape[0], np.max(bbox_int[:, 1]))

    cropped_image = image_left_ocv[y_min:y_max, x_min:x_max]

    if cropped_image.size > 0:
        filename_img = f"image_class1_{timestamp}.png"
        cv2.imwrite(filename_img, cropped_image)
        print(f"🖼️ Saved image crop → {filename_img}")
    else:
        print("⚠ Cropped image is empty.")


# =========================================================================
# === 프레임 단위 처리 함수 ===
# =========================================================================

def process_single_frame(ctx, mem_type):
    """
    한 프레임 단위로:
      - 이미지 grab
      - YOLO 결과 ingest
      - point cloud / image / tracking 업데이트
      - s/q 키 처리
    """
    global image_net, exit_signal, run_signal, detections

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

    # 이미지 가져와서 YOLO 스레드에 던짐
    lock.acquire()
    zed.retrieve_image(image_left_tmp, sl.VIEW.LEFT, mem_type)
    image_net = image_left_tmp.get_data(memory_type=mem_type, deep_copy=False)
    lock.release()
    run_signal = True

    # YOLO 추론 끝날 때까지 대기
    while run_signal:
        sleep(0.001)

    # YOLO 결과 ingest
    lock.acquire()
    zed.ingest_custom_box_objects(detections)
    lock.release()
    zed.retrieve_custom_objects(objects, obj_runtime_param)

    # 뎁스 / 이미지 / pose
    zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, mem_type, point_cloud_res)
    zed.retrieve_image(image_left, sl.VIEW.LEFT, mem_type, display_resolution)
    zed.get_position(cam_w_pose, sl.REFERENCE_FRAME.WORLD)

    # 3D 렌더링
    ctx["viewer"].updateData(point_cloud, objects)

    # 2D 이미지 복사
    if mem_type == sl.MEM.GPU:
        ctx["image_left_ocv"][:] = cp.asnumpy(cp.asarray(image_left.get_data(memory_type=mem_type, deep_copy=False)))
    else:
        np.copyto(image_left_ocv, image_left.get_data(memory_type=mem_type, deep_copy=False))

    # 2D bbox & ID 렌더링
    cv_viewer.render_2D(image_left_ocv, image_scale, objects, obj_param.enable_tracking)

    # bbox 영역 깊이 컬러링
    colorize_bbox_depth(objects, point_cloud, image_left_ocv)

    # Tracking view
    track_view_generator.generate_view(objects, cam_w_pose, image_track_ocv, objects.is_tracked)

    # 최종 화면 합치기
    global_image = cv2.hconcat([image_left_ocv, image_track_ocv])

    cv2.imshow("ZED | 2D View and Birds View", global_image)
    key = cv2.waitKey(1)

    # 키 처리
    if key == ord('s') or key == ord('S'):
        process_and_save_topmost_class1(objects, point_cloud, camera_infos, point_cloud_res, image_left_ocv)

    if key == 27 or key == ord('q') or key == ord('Q'):
        exit_signal = True

    return True


# =========================================================================
# === 전체 실행 루프 ===
# =========================================================================

def run(opt):
    """외부에서 호출 가능한 상위 실행 함수"""
    global exit_signal

    mem_type = get_memory_type(opt)

    # YOLO 스레드 시작
    capture_thread = start_detection_thread(opt)

    # 카메라/뷰어 초기화
    ctx = initialize_camera_and_viewers(opt, mem_type)

    viewer = ctx["viewer"]
    zed = ctx["zed"]

    # 메인 루프
    while viewer.is_available() and not exit_signal:
        if not process_single_frame(ctx, mem_type):
            break

    # 종료 처리
    viewer.exit()
    exit_signal = True
    zed.close()
    capture_thread.join()


# =========================================================================
# === 진입점 ===
# =========================================================================

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
