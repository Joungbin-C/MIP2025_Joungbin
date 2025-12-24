from isaacsim import SimulationApp

# 0. Isaac Sim App 및 Output 설정
simulation_app = SimulationApp({"headless": False})

import os
import time
import numpy as np
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation as R

import omni.kit.app

from isaacsim.core.api import World
from isaacsim.core.utils.types import ArticulationAction

from isaacsim.robot.manipulators.examples.universal_robots.controllers.pick_place_controller import (
    PickPlaceController,
)
from isaacsim.robot.manipulators.examples.universal_robots.controllers.rmpflow_controller import (
    RMPFlowController,
)

from modules.pallet_filling import PalletFilling
from modules.perception import load_yolo
from modules.visualizer import create_visualizer, visualize_side_cluster

import omni.replicator.core as rep
from pxr import UsdGeom, Sdf, UsdLux  

# 출력 디렉토리 설정
OUTPUT_DIR = "perception_output_1208_5"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global Variables
vis = create_visualizer()
print(f"[{OUTPUT_DIR}] 폴더에 인식 결과가 저장됩니다.")


# Helper: rotation matrix → Isaac quaternion (w, x, y, z)
def rotation_matrix_to_quat_isaac(rot_matrix: np.ndarray) -> np.ndarray:
    r = R.from_matrix(rot_matrix)
    x, y, z, w = r.as_quat()
    return np.array([w, x, y, z])


# Camera intrinsics + fixed camera pose
def get_fixed_camera_transform(camera_pos):
    width, height = 1024, 1024
    fx = fy = width
    cx, cy = width / 2.0, height / 2.0

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    rot_mat = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)

    pose_matrix = np.eye(4, dtype=np.float32)
    pose_matrix[:3, :3] = rot_mat
    pose_matrix[:3, 3] = np.array(camera_pos, dtype=np.float32)

    return K, pose_matrix


#  Isaac World + Task + Robot
world = World(stage_units_in_meters=1.0)
task = PalletFilling()
world.add_task(task)
world.reset()

params = task.get_params()
robot_name = params["robot_name"]["value"]

robot = world.scene.get_object(robot_name)

controller = PickPlaceController(
    name="pick_place_controller", gripper=robot.gripper, robot_articulation=robot
)
articulation = robot.get_articulation_controller()

rmp = RMPFlowController(
    name="rmp_controller",
    robot_articulation=robot,
    attach_gripper=True,
)

# Camera + Replicator 설정 + 돔 라이트 추가
stage = world.stage
camera_path = "/World/TopViewCamera"
CAM_POS = (0.55, 1.25, 8.0)

camera_xf = UsdGeom.Xform.Define(stage, Sdf.Path(camera_path))
camera_xf.AddTranslateOp().Set(CAM_POS)
camera_prim_path = Sdf.Path(camera_path + "/Camera")
UsdGeom.Camera.Define(stage, camera_prim_path)
rp = rep.create.render_product(camera_prim_path.pathString, (1024, 1024))

annot_rgb = rep.annotators.get("rgb")
annot_depth = rep.annotators.get("distance_to_image_plane")

world.step(render=True)
annot_rgb.attach(rp)
annot_depth.attach(rp)

# 돔 라이트 추가 로직
print("Adding Dome Light...")
dome_light_path = Sdf.Path("/World/DomeLight")
dome_light = UsdLux.DomeLight.Define(stage, dome_light_path)
dome_light.GetIntensityAttr().Set(2000.0)

for _ in range(10):
    world.step(render=True)
    rep.orchestrator.step()

# Perception Model (YOLO) 및 객체 제거
world.reset()
print("Loading YOLO model...")
model = load_yolo()

# 객체 제거 및 프레임 업데이트 로직
print("Removing Bin and Pipe for perception screenshot...")

try:
    bin_path = str(params["bin_name"]["value"])
    pipe_path = "/World/pipe"

    # 객체 제거
    if world.stage.GetPrimAtPath(bin_path):
        world.stage.RemovePrim(bin_path)
        print(f"Removed: {bin_path}")
    else:
        print(f"Warning: {bin_path} not found.")

    if world.stage.GetPrimAtPath(pipe_path):
        world.stage.RemovePrim(pipe_path)
        print(f"Removed: {pipe_path}")
    else:
        print(f"Warning: {pipe_path} not found.")

except Exception as e:
    print(f"[ERROR] Failed to remove objects: {e}. Check task/scene paths.")

# 렌더링 업데이트 강제 실행
print("Forcing rendering updates after removal...")
for _ in range(30):
    world.step(render=True)
    rep.orchestrator.step()

# State Machine 설정
state = "INITIAL_WAIT"  # 새로운 초기 상태
start_time = time.time()

pallet_index = 0
MAX_PALLETS = 3

place_positions = [
    np.array([1.0, 0, -0.35]),
    np.array([1.0, 0, -0.32 + 0.075]),
    np.array([1.0, 0, -0.32 + 0.22]),
    np.array([1.0, 0, -0.32 + 0.225]),
]

target_pos = None
target_quat = None
latest_perception_data = {}

# 첫 프레임 저장 플래그
first_frame_captured = False

default_joint_pos = np.array(
    [-np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi / 2, 0.0]
)
reset_action = ArticulationAction(joint_positions=default_joint_pos)


# 인식 데이터 저장 함수
def save_perception_data_steps(data: dict, save_id: str):
    rgb_raw = data['rgb_raw']
    depth = data['depth']
    bbox = data['bbox']
    points_array = data['points_array']
    surface_pos = data['surface_pos']
    rot_matrix = data['rot_matrix']
    pallet_center = data['pallet_center']
    points_world = data['points_world']

    x1, y1, x2, y2 = bbox

    rgb_raw = data['rgb_raw']
    depth = data['depth']
    # 새로운 변수 할당
    K = data['K']
    pose_matrix = data['pose_matrix']

    # 1. Raw 사진 및 Raw Point Cloud (Depth) 저장
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{save_id}_step_1_raw_rgb.png"),
                cv2.cvtColor(rgb_raw.copy(), cv2.COLOR_RGB2BGR))
    np.save(os.path.join(OUTPUT_DIR, f"{save_id}_step_1_raw_depth.npy"), depth)
    print(f"  [SAVE 1] Raw data saved.")
    H, W = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    # 전체 픽셀 그리드 생성
    u_grid, v_grid = np.meshgrid(np.arange(W), np.arange(H))

    z_c = depth
    valid_mask = (z_c > 0)  # 0보다 큰 모든 깊이 사용 (노이즈 제외는 Step 3에서)

    # 2D 픽셀을 카메라 좌표계 3D 포인트로 투영
    x_c = (u_grid - cx) * z_c / fx
    y_c = (v_grid - cy) * z_c / fy

    # 유효한 포인트만 추출 (Nx3)
    points_cam = np.stack(
        [x_c[valid_mask], y_c[valid_mask], z_c[valid_mask]], axis=-1
    )

    # 카메라 좌표계 Point Cloud를 월드 좌표계로 변환 (Step 3에서 사용하는 로직과 동일)
    ones = np.ones((points_cam.shape[0], 1))
    raw_points_world = (np.hstack([points_cam, ones]) @ pose_matrix.T)[:, :3]

    # Raw 3D Point Cloud 저장 (Raw Depth의 3D 버전)
    np.save(os.path.join(OUTPUT_DIR, f"{save_id}_step_1_raw_pc_world.npy"), raw_points_world)
    print(f"  [SAVE 1-3D] Raw 3D Point Cloud saved.")

    # 2. 바운딩 박스 쳐진 사진 저장
    rgb_bbox_vis = rgb_raw.copy()
    cv2.rectangle(rgb_bbox_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{save_id}_step_2_bbox_rgb.png"),
                cv2.cvtColor(rgb_bbox_vis, cv2.COLOR_RGB2BGR))
    print(f"  [SAVE 2] Bounding Box image saved.")

    # 3. 바운딩 박스 안의 Point Cloud 저장
    np.save(os.path.join(OUTPUT_DIR, f"{save_id}_step_3_bbox_pc.npy"), points_world)
    print(f"  [SAVE 3] BBox Point Cloud saved.")

    # 4. DBSCAN 거친 Point Cloud 저장 (PCD 파일)
    pcd_dbscan = o3d.geometry.PointCloud()
    pcd_dbscan.points = o3d.utility.Vector3dVector(points_array)
    o3d.io.write_point_cloud(os.path.join(OUTPUT_DIR, f"{save_id}_step_4_dbscan_pc.pcd"), pcd_dbscan)
    print(f"  [SAVE 4] DBSCAN PC saved.")

    # 5 & 6. PCA 결과 및 최종 벡터 시각화
    visualize_side_cluster(vis, points_array, surface_pos, rot_matrix, surface_pos)
    print(f"  [SAVE 5&6] Visualizer output generated.")

    # 7. 사이드 중심과 벡터값 TXT 저장
    target_quat_saved = rotation_matrix_to_quat_isaac(rot_matrix)
    with open(os.path.join(OUTPUT_DIR, f"{save_id}_step_7_picking_pose.txt"), "w") as f:
        f.write(f"Picking Pallet Index: {pallet_index}\n")
        f.write("-" * 30 + "\n")
        f.write(f"Picking Position (XYZ): {surface_pos}\n")
        f.write(f"Picking Orientation (WXYZ): {target_quat_saved}\n")
        f.write(f"Pallet Center (Estimated): {pallet_center}\n")
    print(f"  [SAVE 7] Picking Pose TXT saved.")


# Perception Logic
def perform_perception(rgb_raw, depth):
    # YOLO 실행
    results = model(rgb_raw)
    final_res = results[0] if isinstance(results, list) else results

    boxes = None
    if hasattr(final_res, "boxes") and len(final_res.boxes) > 0:
        boxes = final_res.boxes.xyxy.cpu().numpy()
    elif hasattr(final_res, "xyxy") and len(final_res.xyxy[0]) > 0:
        boxes = final_res.xyxy[0].cpu().numpy()

    if boxes is None or len(boxes) == 0:
        print("  (!) YOLO detection failed.")
        return None

    print("  Processing Side Picking...")
    bbox = list(map(int, boxes[0][:4]))
    x1, y1, x2, y2 = bbox

    K, pose_matrix = get_fixed_camera_transform(CAM_POS)

    base_pos = task._bin_initial_position
    pallet_size = task._bin_size
    pallet_height = pallet_size[2]
    layer_from_top = MAX_PALLETS - 1 - pallet_index
    pallet_center = base_pos + np.array([0, 0, pallet_height * layer_from_top])

    # BBox Point Cloud 계산
    depth_crop = depth[y1:y2, x1:x2]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u_grid, v_grid = np.meshgrid(np.arange(x1, x2), np.arange(y1, y2))
    z_c = depth_crop
    valid_mask = (z_c > 5.0) & (z_c < 12.0)

    if np.sum(valid_mask) < 50:
        print("  (!) Point cloud points too few.")
        return None

    x_c = (u_grid - cx) * z_c / fx
    y_c = (v_grid - cy) * z_c / fy
    points_cam = np.stack(
        [x_c[valid_mask], y_c[valid_mask], z_c[valid_mask]], axis=-1
    )
    ones = np.ones((points_cam.shape[0], 1))
    points_world = (np.hstack([points_cam, ones]) @ pose_matrix.T)[:, :3]

    # DBSCAN 및 PCA 로직
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_world)
    labels = np.array(pcd.cluster_dbscan(eps=0.1, min_points=30))

    if len(labels) == 0 or np.all(labels == -1):
        print("  (!) DBSCAN failed.")
        return None

    largest = np.argmax(np.bincount(labels[labels >= 0]))
    pallet_pcd = pcd.select_by_index(np.where(labels == largest)[0])
    points_array = np.asarray(pallet_pcd.points)

    z_values = points_array[:, 2]
    threshold = np.percentile(z_values, 20)
    top_edge_points = points_array[z_values >= threshold]
    if len(top_edge_points) >= 30:
        points_array = top_edge_points

    # PCA
    cov = np.cov(points_array.T)
    vals, vecs = np.linalg.eig(cov)
    idx = np.argsort(vals)[::-1]
    vecs = vecs[:, idx]

    long_axis = vecs[:, 0]
    long_axis[2] = 0.0
    long_axis /= np.linalg.norm(long_axis)
    side_normal = np.cross([0.0, 0.0, 1.0], long_axis)
    side_normal /= np.linalg.norm(side_normal)

    # Picking Surface 계산
    pc = np.array(pallet_center)
    L, W, H = pallet_size
    short_half = W * 0.5 - 0.1
    z_half = H * 0.5
    face1 = pc + side_normal * short_half
    face2 = pc - side_normal * short_half
    surface_pos = face1 if face1[1] < face2[1] else face2
    side_normal = -side_normal if face1[1] < face2[1] else side_normal
    surface_pos[2] = pc[2] + z_half

    x_axis = side_normal.copy()
    x_axis[2] = 0.0
    x_axis /= np.linalg.norm(x_axis)
    z_axis = np.array([0.0, 0.0, 1.0])
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    rot_matrix = np.column_stack((x_axis, y_axis, z_axis))

    # 최종 포즈와 데이터 반환
    return {
        'rgb_raw': rgb_raw, 'depth': depth, 'bbox': bbox, 'K': K,
        'pose_matrix': pose_matrix, 'pallet_center': pallet_center,
        'points_world': points_world, 'points_array': points_array,
        'surface_pos': surface_pos, 'rot_matrix': rot_matrix,
        'target_quat': rotation_matrix_to_quat_isaac(rot_matrix)
    }


# Main Loop
while simulation_app.is_running():
    world.step(render=True)

    # 초기 프레임 캡처 로직
    if world.is_playing() and not first_frame_captured:
        print("시뮬레이션 시작 감지-> 첫 프레임 데이터를 수집합니다.")

        rep.orchestrator.step()
        rgb = annot_rgb.get_data()
        depth = annot_depth.get_data()

        if rgb is not None and depth is not None:
            # 데이터 전처리
            if rgb.ndim == 4:
                rgb = rgb[0]
                depth = depth[0]

            rgb_raw = rgb[..., :3]
            if rgb_raw.dtype != np.uint8:
                rgb_raw = (rgb_raw * 255).astype(np.uint8)

            # Perception 수행
            data = perform_perception(rgb_raw, depth)

            if data is not None:
                target_pos = data['surface_pos']
                target_quat = data['target_quat']
                latest_perception_data = data

                # 데이터 저장
                save_id = f"initial_pallet_{pallet_index}"
                save_perception_data_steps(latest_perception_data, save_id)

                # 상태 업데이트
                first_frame_captured = True
                state = "STABILIZING"  # 안정화 단계로 진입
                start_time = time.time()  # 안정화 타이머 시작
                print(">>> 첫 프레임 캡처 및 저장 완료. 안정화 단계로 진입합니다.")
            else:
                print(">>> 첫 프레임 인식 실패. Play 버튼을 다시 눌러주세요.")

    if not world.is_playing():
        continue

    # 1) 안정화 대기
    if state == "STABILIZING":
        if time.time() > start_time + 3.0:
            state = "MOVE_PICK_UP"  # 포즈가 이미 준비되었으므로 바로 이동
        continue

    # 2) Perception: 데이터 획득 및 포즈 계산
    if state == "PERCEPTION":
        rep.orchestrator.step()

        rgb = annot_rgb.get_data()
        depth = annot_depth.get_data()

        if rgb is None or depth is None:
            continue

        if rgb.ndim == 4:
            rgb = rgb[0]
            depth = depth[0]

        rgb_raw = rgb[..., :3]
        if rgb_raw.dtype != np.uint8:
            rgb_raw = (rgb_raw * 255).astype(np.uint8)

        data = perform_perception(rgb_raw, depth)

        if data is not None:
            target_pos = data['surface_pos']
            target_quat = data['target_quat']
            latest_perception_data = data

            print(f"  ✔ Side Surface Pos (World): {target_pos}")
            print("  ▶ Perception 성공. 저장 및 동작을 시작합니다...")

            # 인식 결과 저장
            save_id = f"pallet_{pallet_index}"
            save_perception_data_steps(latest_perception_data, save_id)

            state = "MOVE_PICK_UP"
        continue

    # 3) Pallet 옆면 pick (surface suction)
    if state == "MOVE_PICK_UP":
        obs = world.get_observations()

        # pallet_index 2번은 특수 로직 유지
        if pallet_index == 2:
            modified_target_pos = np.array([0.55, 0.85, -0.4])
            target_quat = np.array([-0.6963, 0.1227, -0.1227, -0.6963])

            actions = controller.forward(
                picking_position=modified_target_pos,
                placing_position=np.array([0.55, 0.75, 0.35]),
                current_joint_positions=obs[robot_name]["joint_positions"],
                end_effector_orientation=target_quat,
            )
        else:
            actions = controller.forward(
                picking_position=target_pos,
                placing_position=np.array([0.55, 0.75, 0.35]),
                current_joint_positions=obs[robot_name]["joint_positions"],
                end_effector_orientation=target_quat,
            )

        articulation.apply_action(actions)

        if controller.get_current_event() == 6:
            print("PICK COMPLETED (suction still on)")
            controller.reset()
            state = "MOVE_PICK_SIDE"

    # 4) 옆으로 이동
    if state == "MOVE_PICK_SIDE":
        side_target = np.array([0.75, 0.75, 0.35])

        if pallet_index == 2:
            place_quat = np.array([-0.9848, 0.0, -0.1736, 0.0])
        else:
            place_quat = np.array([ -1.0, 0.0, 0.0, 0.0 ])

        obs = world.get_observations()
        actions = rmp.forward(
            target_end_effector_position=side_target,
            target_end_effector_orientation=place_quat,
        )
        articulation.apply_action(actions)

        ee_pos = obs[robot_name]["end_effector_position"]
        if np.linalg.norm(ee_pos - side_target) < 0.3:
            print("SIDE MOVE DONE")
            state = "MOVE_PICK_DOWN"

    # 5) stack 위치로 내려놓기 (place)
    if state == "MOVE_PICK_DOWN":
        obs = world.get_observations()

        if pallet_index == 2:
            place_quat = np.array([-0.9848, 0.0, -0.1736, 0.0])
        else:
            place_quat = np.array([ -1.0, 0.0, 0.0, 0.0 ])

        actions = controller.forward(
            picking_position=np.array([0.75, 0.75, 0.35]),
            placing_position=place_positions[pallet_index],
            current_joint_positions=obs[robot_name]["joint_positions"],
            end_effector_orientation=place_quat,
        )
        articulation.apply_action(actions)

        if controller.is_done():
            print(f"[{pallet_index}] PLACE DONE")

            pallet_index += 1

            for _ in range(50):
                articulation.apply_action(reset_action)
                world.step(render=True)
            controller.reset()

            if pallet_index >= MAX_PALLETS:
                print("ALL PALLETS MOVED!")
                state = "FINISHED"
            else:
                state = "PERCEPTION"  # 다음 팔레트 인식을 위해 PERCEPTION 상태로 전환

    if state == "FINISHED":
        pass

simulation_app.close()