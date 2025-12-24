# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from isaacsim import SimulationApp
# 시뮬레이션 앱 시작
simulation_app = SimulationApp({"headless": False})
import numpy as np
import time
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.grippers import SurfaceGripper
from isaacsim.storage.native import get_assets_root_path
from controller.rmpflow import RMPFlowController



# 월드 생성
my_world = World(stage_units_in_meters=1.0)
my_world.reset()
task_params = my_world.get_task("ur10e_follow_target").get_params()
target_name = task_params["target_name"]["value"]
ur10e_name = task_params["robot_name"]["value"]
my_ur10e = my_world.scene.get_object(ur10e_name)
articulation_controller = my_ur10e.get_articulation_controller()

# Gripper 생성
gripper = SurfaceGripper(
    end_effector_prim_path="/World/UR10/ee_link",
    surface_gripper_path="/World/UR10/ee_link/SurfaceGripper"
)

# UR10 Manipulator 추가
ur10 = my_world.scene.add(
    SingleManipulator(
        prim_path="/World/UR10",
        name="my_ur10",
        end_effector_prim_path="/World/UR10/ee_link",
        gripper=gripper
    )
)

# Stage 초기화 및 articulation 준비
max_wait_time = 5.0
waited = 0.0
step_time = 0.05

while ur10.get_articulation_controller() is None and waited < max_wait_time:
    my_world.step(render=True)
    time.sleep(step_time)
    waited += step_time

articulation_controller  = RMPFlowController(name="target_follower_controller", robot_articulation=my_ur10e)
articulation_controller.reset()

if articulation_controller is None:
    print("Articulation 초기화 실패")
    simulation_app.close()
    exit()

# joint_positions가 None인지 확인 후 대기
while ur10.get_joint_positions() is None:
    my_world.step(render=True)
    time.sleep(step_time)

# 초기 joint 상태 설정
joint_positions = np.array([-np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2, np.pi/2, 0])
ur10.set_joints_default_state(joint_positions)

# sinus 움직임 변수
t = 0.0

# 시뮬레이션 loop
while simulation_app.is_running():
    my_world.step(render=True)

    current_joints = ur10.get_joint_positions()
    if current_joints is not None:
        target_joints = current_joints.copy()
        target_joints[0] = 0.5 * np.sin(t)
        articulation_controller.apply_action(target_joints)
        t += 0.05

simulation_app.close()
