import open3d as o3d
import numpy as np
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

def create_visualizer():
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Point Cloud Visualization")
    opt = vis.get_render_option()
    opt.background_color = (0.1, 0.1, 0.1)
    opt.point_size = 3.0
    return vis

def visualize_side_cluster(vis, points, center, normal, surface_pos):
    # points      : DBSCAN으로 골라낸 팔렛 옆면 포인트들
    # center      : PCA center
    # normal      : PCA로 구한 side normal (world)
    # surface_pos : suction target

    vis.clear_geometries()

    # 1) Point Cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0.1, 0.7, 1.0])
    vis.add_geometry(pcd)

    # 2) Normal vector arrow visualization
    arrow_len = 1
    end = surface_pos - normal * arrow_len

    line_points = np.vstack([surface_pos, end])
    line = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(line_points),
        lines=o3d.utility.Vector2iVector([[0, 1]]),
    )
    line.colors = o3d.utility.Vector3dVector([[1, 0, 0]])
    vis.add_geometry(line)

    # 3) Suction Target sphere
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.04)
    sphere.paint_uniform_color([1, 1, 0])
    sphere.translate(surface_pos)
    vis.add_geometry(sphere)

    # Update window
    vis.poll_events()
    vis.update_renderer()

def draw_gripper_axes(vis, pos, Rot, length=0.15):
    # pos: EE position (3)
    # Rot: 3x3 rotation matrix

    axes = []
    colors = [[1,0,0], [0,1,0], [0,0,1]]

    origin = pos
    for i in range(3):
        end = origin + Rot[:, i] * length
        line = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector([origin, end]),
            lines=o3d.utility.Vector2iVector([[0, 1]])
        )
        line.colors = o3d.utility.Vector3dVector([colors[i]])
        axes.append(line)

    for a in axes:
        vis.add_geometry(a)
    vis.poll_events()
    vis.update_renderer()

def draw_vectors(vis, side_info, target_side_name):
    if not side_info:
        return

    arrow_length = 0.3  # 화살표 길이 (30cm)

    for name, info in side_info.items():
        start_point = info["center"]
        normal_vector = info["normal"]

        # 화살표의 끝점 계산
        end_point = start_point + normal_vector * arrow_length

        line_points = [start_point, end_point]
        line_indices = [[0, 1]]

        if name == target_side_name:
            color = [1, 0, 0]
        else:
            color = [0, 0, 1]

        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(line_points),
            lines=o3d.utility.Vector2iVector(line_indices)
        )
        line_set.colors = o3d.utility.Vector3dVector([color])

        vis.add_geometry(line_set)