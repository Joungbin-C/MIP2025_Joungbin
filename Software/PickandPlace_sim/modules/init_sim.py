from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import omni
from pxr import UsdGeom, Sdf, UsdPhysics, PhysxSchema, Gf
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
import random

# setup_robot 함수를 사용하기 위해 필요한 임포트
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.grippers import SurfaceGripper

ROBOT_POSITION = (-1.1, -0.15, 0)


def setup_robot(world, assets_root_path):
    ur10_usd = assets_root_path + "/Isaac/Robots/UniversalRobots/ur10/ur10.usd"
    robot_prim = add_reference_to_stage(ur10_usd, "/World/UR10")
    robot_prim.GetVariantSet("Gripper").SetVariantSelection("Short_Suction")

    robot_xform = UsdGeom.Xform(robot_prim)
    translate_ops = [op for op in robot_xform.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
    if translate_ops:
        translate_ops[0].Set(Gf.Vec3d(*ROBOT_POSITION))
    else:
        robot_xform.AddTranslateOp().Set(Gf.Vec3d(*ROBOT_POSITION))

    surface_path = "/World/UR10/ee_link/SurfaceGripper"

    gripper = SurfaceGripper(
        end_effector_prim_path="/World/UR10/ee_link",
        surface_gripper_path=surface_path,
    )

    ur10 = world.scene.add(
        SingleManipulator(
            prim_path="/World/UR10",
            name="ur10_robot",
            end_effector_prim_path="/World/UR10/ee_link",
            gripper=gripper
        )
    )

    ur10.set_joints_default_state([0, -1.57, 2.09, -0.52, 1.57, 0])
    articulation_controller = ur10.get_articulation_controller()

    return ur10, articulation_controller


def init_simulation(headless=False):
    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        raise RuntimeError("Isaac Sim 에셋 경로를 찾을 수 없습니다. Nucleus 서버 연결을 확인하세요.")

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    stage = world.stage
    dome_light = stage.DefinePrim("/World/DomeLight", "DomeLight")
    dome_light.GetAttribute("inputs:intensity").Set(1000.0)

    # 로봇 셋업을 팔레트보다 먼저 실행
    ur10, articulation_controller = setup_robot(world, assets_root_path)

    asset_path = assets_root_path + "/Isaac/Props/Pallet/pallet.usd"

    pallet_scale = (0.3, 0.3, 0.3)
    base_height = 0.05
    stacking_gap = 0.05

    for i in range(5):
        parent_path = f"/World/PalletParent_{i}"
        ref_path = f"{parent_path}/Pallet_{i}"

        parent_xform = UsdGeom.Xform.Define(stage, Sdf.Path(parent_path))
        parent_prim = stage.GetPrimAtPath(parent_path)

        rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(parent_prim)
        mass_api = UsdPhysics.MassAPI.Apply(parent_prim)
        mass_api.GetMassAttr().Set(10.0)

        physx_rigid_body_api = PhysxSchema.PhysxRigidBodyAPI.Apply(parent_prim)
        physx_rigid_body_api.CreateLinearDampingAttr().Set(0.8)
        physx_rigid_body_api.CreateAngularDampingAttr().Set(0.8)

        current_z = base_height + (i * stacking_gap)
        parent_xform.AddTranslateOp().Set((0, 0, current_z))
        # parent_xform.AddRotateZOp().Set(random.uniform(0, 360))
        parent_xform.AddScaleOp().Set(pallet_scale)

        add_reference_to_stage(asset_path, ref_path)

        pallet_prim = stage.GetPrimAtPath(ref_path)

        collision_api = UsdPhysics.MeshCollisionAPI.Apply(pallet_prim)
        collision_api.GetApproximationAttr().Set(UsdPhysics.Tokens.convexHull)

    return simulation_app, world, assets_root_path, ur10, articulation_controller