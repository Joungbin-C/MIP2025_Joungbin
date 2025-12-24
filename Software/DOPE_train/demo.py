# predict_dope_multistage.py
# DOPE Multi-Stage Prediction + GT Visualization + Cuboid Lines
# + Final Stage Heatmap & Overlay Save
# + DOPE 6D Pose (PnP) + Cuboid Projection Visualization

import os
import json
import warnings
import numpy as np
from PIL import Image
import cv2

import torch
import torchvision.transforms as transforms
from scipy.spatial.transform import Rotation as R  # ✅ 자세(Euler) 계산용

from train import (
    DopeMultiStageNet,
    NUM_KEYPOINTS,
    NUM_AFFINITY,
    NUM_STAGES,
    IMAGE_SIZE,
    OUTPUT_SIZE
)

warnings.filterwarnings("ignore")

# PATHS
MODEL_PATH = r"C:\DOPE_Training\train_test_final_251207\best_model_multistage.pth"
INPUT_IMAGE = r"C:\isaacsim\FinalProject\pose_compare_output\01_yolo_bbox.png"
INPUT_JSON  = r"C:\Isaac_Data\Pallet_Dataset_Total\003140.json"

SAVE_VIS          = r"C:\DOPE_Training\train_test3\pred_vis_lines.png"          # GT + Pred keypoints
SAVE_DOPE_ORIENT  = r"C:\DOPE_Training\train_test3\pred_dope_orientation.png"   # DOPE 6D pose cuboid

SAVE_HEATMAP_DIR = r"C:\DOPE_Training\train_test3\heatmaps"
os.makedirs(SAVE_HEATMAP_DIR, exist_ok=True)

# 카메라
FX = FY = float(IMAGE_SIZE)
CX = CY = float(IMAGE_SIZE) / 2.0
K = np.array([[FX, 0, CX],
              [0, FY, CY],
              [0,  0,  1]], dtype=np.float32)

# 팔레트 cuboid 3D 모델 좌표(m)
CUBOID_3D = np.array([
    [-0.25, -0.35, -0.10],
    [ 0.25, -0.35, -0.10],
    [ 0.25,  0.35, -0.10],
    [-0.25,  0.35, -0.10],
    [-0.25, -0.35,  0.10],
    [ 0.25, -0.35,  0.10],
    [ 0.25,  0.35,  0.10],
    [-0.25,  0.35,  0.10],
], dtype=np.float32)

# Cuboid 라인 연결 규칙 (8점 기준)
CUBOID_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,0),
    (4,5), (5,6), (6,7), (7,4),
    (0,4), (1,5), (2,6), (3,7)
]


def fix_rectangle_points(raw_pts):
    pts = np.unique(raw_pts, axis=0)  # 중복 제거

    # 이미 4점이면 그대로 사용
    if len(pts) == 4:
        return pts

    # 3점일 때만 보정
    if len(pts) == 3:
        A, B, C = pts

        dAB = np.linalg.norm(A - B)
        dAC = np.linalg.norm(A - C)
        dBC = np.linalg.norm(B - C)

        # 가장 긴 변이 대각선이라고 가정
        if dAB >= dAC and dAB >= dBC:
            # AB가 대각선, C가 나머지 → D = A + (B - C)
            D = A + (B - C)
        elif dAC >= dAB and dAC >= dBC:
            # AC가 대각선, B가 나머지 → D = A + (C - B)
            D = A + (C - B)
        else:
            # BC가 대각선, A가 나머지 → D = B + (C - A)
            D = B + (C - A)

        pts = np.vstack([pts, D])

    return pts[:4]


def order_rectangle_points(pts):
    pts = np.array(pts, dtype=np.float32)

    # 중심점 계산
    cx = np.mean(pts[:, 0])
    cy = np.mean(pts[:, 1])

    # 각도 계산
    angles = np.arctan2(pts[:,1] - cy, pts[:,0] - cx)

    # 반시계 방향 정렬
    sort_idx = np.argsort(angles)
    return pts[sort_idx]
def draw_orientation_arrow(img, rect_pts, color=(0,0,255), thickness=3):
    img = img.copy()

    # 중심
    cx = np.mean(rect_pts[:,0])
    cy = np.mean(rect_pts[:,1])
    center = (int(cx), int(cy))

    # 기존 축 벡터
    v = rect_pts[1] - rect_pts[0]

    # +90° 회전
    v = np.array([-v[1], v[0]])

    # 정규화
    v = v / np.linalg.norm(v)

    # 화살표 길이
    arrow_len = 120
    end_point = (int(cx + v[0]*arrow_len), int(cy + v[1]*arrow_len))

    cv2.arrowedLine(img, center, end_point, color, thickness, tipLength=0.3)
    return img

def draw_simple_rectangle(img_np, pred_kp, save_path):
    img = img_np.copy()
    raw_pts = pred_kp[:4]

    # 사각형으로 재정렬
    pts = fix_rectangle_points(raw_pts)
    pts = order_rectangle_points(pts).astype(int)

    # 사각형 그리기
    for i in range(4):
        pt1 = tuple(pts[i])
        pt2 = tuple(pts[(i + 1) % 4])
        cv2.line(img, pt1, pt2, (0, 255, 0), 3)
    img = draw_orientation_arrow(img, pts, color=(0,0,255), thickness=4)

    cv2.imwrite(save_path, img)
    print("Rectangle saved to:", save_path)

def load_ground_truth(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return np.array(data["objects"][0]["projected_cuboid"], dtype=np.float32)

# Belief → Keypoint
def extract_keypoints_from_belief(belief_map):
    keypoints = []
    for i in range(NUM_KEYPOINTS):
        heat = belief_map[i]
        y, x = np.unravel_index(np.argmax(heat), heat.shape)

        x = x * (IMAGE_SIZE / OUTPUT_SIZE)
        y = y * (IMAGE_SIZE / OUTPUT_SIZE)
        keypoints.append([x, y])
    return np.array(keypoints, dtype=np.float32)

# 이미지 전처리
def preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    img = Image.open(img_path).convert("RGB")
    if img.size != (IMAGE_SIZE, IMAGE_SIZE):
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))

    return transform(img).unsqueeze(0), np.array(img)

# DOPE Inference (run_prediction)
def run_prediction(model_path, img_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DopeMultiStageNet(
        num_keypoints=NUM_KEYPOINTS,
        num_affinity=NUM_AFFINITY,
        num_stages=NUM_STAGES,
        pretrained_backbone=False
    ).to(device)

    print("Loading model:", model_path)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    img_tensor, img_np = preprocess_image(img_path)

    with torch.no_grad():
        pred_beliefs_list, _ = model(img_tensor.to(device))

    final_belief = pred_beliefs_list[-1].cpu().numpy()[0]  # (K, H, W)
    keypoints_pred = extract_keypoints_from_belief(final_belief)

    return keypoints_pred, img_np, final_belief

# 3D → 2D Projection
def project_points(points_3d, Rm, tvec, K):
    pts_cam = (Rm @ points_3d.T + tvec.reshape(3,1)).T
    x, y, z = pts_cam[:,0], pts_cam[:,1], pts_cam[:,2]

    u = K[0,0] * (x / z) + K[0,2]
    v = K[1,1] * (y / z) + K[1,2]

    return np.vstack([u, v]).T

# PnP로 6D Pose 계산 (DOPE keypoint → R, t)
def solve_pnp_from_keypoints(pred_kp):

    kp_2d = pred_kp[:8].astype(np.float32)  # (8,2)

    success, rvec, tvec = cv2.solvePnP(
        CUBOID_3D, kp_2d, K, None, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        raise RuntimeError("PnP failed for DOPE keypoints.")

    Rm, _ = cv2.Rodrigues(rvec)      # (3,3)
    tvec = tvec.reshape(3)           # (3,)
    return Rm, tvec

# DOPE Pose Visualization
def draw_dope_orientation(img_np, pred_kp, Rm, tvec):
    img = img_np.copy()

    # 예측 keypoint 표시
    for (x, y) in pred_kp:
        cv2.circle(img, (int(x), int(y)), 5, (0,255,0), -1)

    # 예측 cuboid projection
    pts_2d = project_points(CUBOID_3D, Rm, tvec, K).astype(int)

    for (a, b) in CUBOID_CONNECTIONS:
        pt1 = tuple(pts_2d[a])
        pt2 = tuple(pts_2d[b])
        cv2.line(img, pt1, pt2, (0,255,0), 2)

    return img

# 시각화 (GT + Pred Keypoints + GT Cuboid)
def visualize_prediction(img_np, pred_kp, gt_kp, save_path):
    vis = img_np.copy()

    # GT keypoints (BLUE)
    for i, (x, y) in enumerate(gt_kp):
        cv2.circle(vis, (int(x), int(y)), 5, (255,0,0), -1)
        cv2.putText(vis, f"G{i}", (int(x)+5, int(y)+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

    # Pred keypoints (GREEN)
    for i, (x, y) in enumerate(pred_kp):
        cv2.circle(vis, (int(x), int(y)), 5, (0,255,0), -1)
        cv2.putText(vis, f"P{i}", (int(x)+5, int(y)+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # GT cuboid lines (BLUE)
    for (a, b) in CUBOID_CONNECTIONS:
        pt1 = (int(gt_kp[a][0]), int(gt_kp[a][1]))
        pt2 = (int(gt_kp[b][0]), int(gt_kp[b][1]))
        cv2.line(vis, pt1, pt2, (255,0,0), 2)

    cv2.imwrite(save_path, vis)
    print("Visualization saved to:", save_path)

# Heatmap 저장
def save_belief_maps(belief_map, img_np, base_name):
    for i in range(NUM_KEYPOINTS):
        heat = belief_map[i]

        heat_norm = cv2.normalize(heat, None, 0, 255, cv2.NORM_MINMAX)
        heat_uint8 = heat_norm.astype(np.uint8)
        heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
        heat_color = cv2.resize(heat_color, (IMAGE_SIZE, IMAGE_SIZE))

        cv2.imwrite(os.path.join(SAVE_HEATMAP_DIR, f"{base_name}_kp{i}_heat.png"), heat_color)
        overlay = cv2.addWeighted(img_np, 0.6, heat_color, 0.4, 0)
        cv2.imwrite(os.path.join(SAVE_HEATMAP_DIR, f"{base_name}_kp{i}_overlay.png"), overlay)

    print(f"Heatmaps saved to: {SAVE_HEATMAP_DIR}")

# Entry
if __name__ == "__main__":

    # 1) DOPE inference
    pred_kp, img_np, final_belief = run_prediction(MODEL_PATH, INPUT_IMAGE)

    # 2) GT 2D keypoints
    gt_kp = load_ground_truth(INPUT_JSON)

    # 3) GT vs Pred keypoints + GT cuboid
    visualize_prediction(img_np, pred_kp, gt_kp, SAVE_VIS)

    # 4) PnP로 DOPE 6D Pose (R, t) 계산
    R_dope, t_dope = solve_pnp_from_keypoints(pred_kp)

    # 5) Euler 각도 (카메라 기준, xyz 순)
    euler_deg = R.from_matrix(R_dope).as_euler("xyz", degrees=True)

    print("\n===== DOPE 6D Pose (Camera ← Pallet) =====")
    print("Rotation matrix R:\n", R_dope)
    print("\nEuler angles (deg, xyz): Roll={:.2f}, Pitch={:.2f}, Yaw={:.2f}".format(
        euler_deg[0], euler_deg[1], euler_deg[2]
    ))
    print("\nTranslation t (camera frame):", t_dope)
    print("==========================================\n")

    # 6) DOPE 6D pose 기반 cuboid projection 시각화
    dope_img = draw_dope_orientation(img_np, pred_kp, R_dope, t_dope)
    cv2.imwrite(SAVE_DOPE_ORIENT, dope_img)
    print(" DOPE Orientation image (pose) saved:", SAVE_DOPE_ORIENT)

    # 7) Heatmaps 저장
    base_name = os.path.splitext(os.path.basename(INPUT_IMAGE))[0]
    save_belief_maps(final_belief, img_np, base_name)

    print("\nPredicted Keypoints:\n", pred_kp)
    print("\nGround Truth Keypoints:\n", gt_kp)

    SAVE_RECT = r"C:\DOPE_Training\train_test3\pred_rectangle.png"

    draw_simple_rectangle(img_np, pred_kp, SAVE_RECT)

    print("\nPose estimation + visualization finished.")
