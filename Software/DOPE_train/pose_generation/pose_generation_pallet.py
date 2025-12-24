# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate a [DOPE] synthetic datasets for PALLET
Modified by Gemini: Fixed mesh_path error & Removed invalid writer arguments & Forced Pallet Config
"""

import argparse
import datetime
import os
import signal
import numpy as np
import torch
import yaml
from isaacsim import SimulationApp

# ==============================================================================
# [사용자 설정] 팔레트 USD 경로 (S3 링크 사용)
# ==============================================================================
PALLET_USD_URL = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/Props/Pallet/pallet.usd"
# ==============================================================================

parser = argparse.ArgumentParser("Pose Generation data generator")
parser.add_argument("--num_mesh", type=int, default=1000, help="Number of frames to record")
parser.add_argument("--num_dome", type=int, default=0, help="Number of frames to record similar to DOME dataset")
parser.add_argument("--dome_interval", type=int, default=1, help="Interval for dome background switching")
parser.add_argument("--output_folder", "-o", type=str, default=r"C:\Isaac_Data\Pallet_Dataset_warehouse_background",
                    help="Output directory.")
parser.add_argument("--use_s3", action="store_true", help="Saves output to s3 bucket.") 
parser.add_argument("--bucket", type=str, default=None, help="Bucket name")
parser.add_argument("--s3_region", type=str, default="us-east-1", help="s3 region.")
parser.add_argument("--endpoint", "--endpoint_url", type=str, default=None, help="s3 endpoint.")
parser.add_argument("--writer", type=str, default="dope", help="Which writer to use [DOPE]")
parser.add_argument("--debug", action="store_true", help="Write debug images.")
parser.add_argument("--test", action="store_true", help="Generates data for testing.")

args, unknown_args = parser.parse_known_args()

if args.test:
    args.use_s3 = False

CONFIG_FILES = {
    "dope": "config/dope_config.yaml",
    # "ycbvideo": "config/ycb_config.yaml",
    # "centerpose": "config/centerpose_config.yaml",
}
TEST_CONFIG_FILES = {
    "dope": "pose_tests/dope/test_dope_config.yaml",
    # "ycbvideo": "pose_tests/ycbvideo/test_ycb_config.yaml",
}

cf_map = TEST_CONFIG_FILES if args.test else CONFIG_FILES
CONFIG_FILE = cf_map.get(args.writer.lower(), "config/dope_config.yaml")
CONFIG_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), CONFIG_FILE)

# YAML 파일 로드
with open(CONFIG_FILE_PATH) as f:
    config_data = yaml.full_load(f)

# ==============================================================================
# [설정 강제 수정] YAML 파일을 무시하고 팔레트 설정을 덮어씁니다.
# ==============================================================================
print(f"[User Mod] 팔레트 설정을 적용합니다. 타겟 URL: {PALLET_USD_URL}")

# 1. 학습 물체 변경 (크래커 상자 -> 팔레트)
config_data["OBJECTS_TO_GENERATE"] = [
    {"part_name": "Pallet_Custom", "num": 1, "prim_type": "pallet"}
]

# 2. 카메라 거리 대폭 증가
config_data["MIN_DISTANCE"] = 2.5
config_data["MAX_DISTANCE"] = 6.0

# 3. 방해물(Distractor) 제거 (깨끗한 학습을 위해)
config_data["MESH_FILENAMES"] = []
config_data["NUM_MESH_SHAPES"] = 0
config_data["NUM_MESH_OBJECTS"] = 0

# 4. 회전 범위 제한 (바닥에 놓인 상태 유지)
config_data["MIN_ROTATION_RANGE"] = [-180, -20, -180]
config_data["MAX_ROTATION_RANGE"] = [180, 20, 180]
# ==============================================================================

OBJECTS_TO_GENERATE = config_data["OBJECTS_TO_GENERATE"]

kit = SimulationApp(launch_config=config_data["CONFIG"])

import math
import carb
import omni.replicator.core as rep
from isaacsim.core.api import World
from isaacsim.core.prims import XFormPrim
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.semantics import add_labels
from isaacsim.replicator.writers import PoseWriter, YCBVideoWriter
from isaacsim.storage.native import get_assets_root_path

world = World(physics_dt=1.0 / 30.0)
world.reset()

from flying_distractors.collision_box import CollisionBox
from flying_distractors.dynamic_object import DynamicObject
from flying_distractors.flying_distractors import FlyingDistractors
from isaacsim.core.utils.random import get_random_world_pose_in_view
from isaacsim.core.utils.transformations import get_world_pose_from_relative
from pose_tests.test_utils import clean_output_dir, run_pose_generation_test


class RandomScenario(torch.utils.data.IterableDataset):
    def __init__(self, num_mesh, num_dome, dome_interval, output_folder, use_s3=False, endpoint="",
                 s3_region="us-east-1", writer="dope", bucket="", test=False, debug=False):
        self.ENV_URLS = {
            "Warehouse": "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/Environments/Simple_Warehouse/warehouse.usd",
            "FlatPlane": "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/Environments/Terrains/flat_plane.usd"
            # "Rivermark": "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/Environments/Outdoor/Rivermark/rivermark.usd"
        }
        self.test = test
        self.writer_format = writer.lower()
        self.debug = debug

        if writer == "ycbvideo":
            self.writer_helper = YCBVideoWriter
        elif writer == "dope" or writer == "centerpose":
            self.writer_helper = PoseWriter
        else:
            raise Exception("Invalid writer specified.")

        self.result = True
        assets_root_path = get_assets_root_path()

        self.dome_texture_path = ""
        if assets_root_path:
            self.dome_texture_path = assets_root_path + config_data["DOME_TEXTURE_PATH"]
            self.distractor_asset_path = assets_root_path + config_data["DISTRACTOR_ASSET_PATH"]
            self.train_asset_path = assets_root_path + config_data["TRAIN_ASSET_PATH"]
        else:
            # 에셋 경로를 못 찾아도 진행하도록 빈 값 설정
            print("[Warning] Isaac Sim assets not found. Backgrounds might be black.")
            self.distractor_asset_path = ""
            self.train_asset_path = ""

        self.train_parts = []
        self.train_part_mesh_path_to_prim_path_map = {}
        self.mesh_distractors = FlyingDistractors()
        self.dome_distractors = FlyingDistractors()
        self.current_distractors = None

        self.num_mesh = max(0, num_mesh) if not self.test else 5
        self.num_dome = max(0, num_dome) if not self.test else 0
        self.train_size = self.num_mesh + self.num_dome
        self.dome_interval = dome_interval

        self._output_folder = output_folder if use_s3 else os.path.join(os.getcwd(), output_folder)
        self.use_s3 = use_s3
        self.endpoint = endpoint
        self.s3_region = s3_region
        self.bucket = bucket

        self._setup_world()
        self.cur_idx = 0
        self.exiting = False
        self.last_frame_reached = False

        if not self.use_s3 and self.test:
            clean_output_dir(self._output_folder)

        self._carb_settings = carb.settings.get_settings()
        self._carb_settings.set("/omni/replicator/captureOnPlay", False)
        self._carb_settings.set("/app/asyncRendering", False)
        self._carb_settings.set("/omni/replicator/asyncRendering", False)
        signal.signal(signal.SIGINT, self._handle_exit)

    def _handle_exit(self, *args, **kwargs):
        print("[SDG] Exiting dataset generation..")
        self.exiting = True

    def _setup_world(self):
        self._setup_camera()
        rep.settings.set_render_rtx_realtime()
        world.get_physics_context().set_gravity(0.0)
        collision_box = self._setup_collision_box()
        world.scene.add(collision_box)

        # Distractors setup (설정에서 껐으므로 패스)
        self._setup_distractors(collision_box)
        self._load_environments_usd()
        # [핵심] 팔레트 로드
        self._setup_train_objects()

        self._setup_randomizers()

        for _ in range(5):
            kit.app.update()

        if self.writer_helper == PoseWriter:
            self.writer = rep.WriterRegistry.get("PoseWriter")
            # [수정] 에러 발생시키는 bounding_box_3d 옵션 제거함
            self.writer.initialize(
                output_dir=self._output_folder,
                write_debug_images=self.debug,
                format=self.writer_format,
                skip_empty_frames=False,
                # bounding_box_3d=True,  <-- 제거됨
                use_s3=self.use_s3,
                s3_bucket=self.bucket,
                s3_endpoint_url=self.endpoint,
                s3_region=self.s3_region,
            )
        else:
            self.writer_helper.register_pose_annotator(config_data=config_data)
            self.writer = self.writer_helper.setup_writer(
                config_data=config_data,
                writer_config={
                    "output_folder": self._output_folder,
                    "train_size": self.train_size,
                },
            )
        self.writer.attach([self.render_product])
        self.dome_distractors.set_visible(False)
        rep.orchestrator.preview()

    def _load_environments_usd(self):
        """USD 환경들을 로드하고 초기에는 모두 숨깁니다."""
        from isaacsim.core.utils.stage import add_reference_to_stage
        from pxr import UsdGeom

        self.env_prim_paths = []
        print("[SDG] Loading 3D Environments from S3...")

        for name, url in self.ENV_URLS.items():
            # 이름에 공백이나 특수문자가 없도록 처리
            safe_name = name.replace(" ", "_")
            prim_path = f"/World/Environments/{safe_name}"

            # 스테이지에 참조로 추가
            add_reference_to_stage(usd_path=url, prim_path=prim_path)
            self.env_prim_paths.append(prim_path)

            # 일단 모두 숨김 처리 (Imageable 속성 사용)
            prim = world.stage.GetPrimAtPath(prim_path)
            if prim.IsValid():
                imageable = UsdGeom.Imageable(prim)
                imageable.MakeInvisible()
                print(f"  - Loaded: {name}")
            else:
                print(f"  - [Error] Failed to load {name}")

    def _setup_camera(self):
        focal_length_mm = (config_data["F_X"] + config_data["F_Y"]) * config_data["pixel_size"] / 2
        horiztonal_aperture_mm = config_data["pixel_size"] * config_data["WIDTH"]

        self.camera = rep.create.camera(
            position=(0, 0, 9.0),
            rotation=(0, -180, 0),
            # rotation=np.array(config_data["CAMERA_RIG_ROTATION"]),
            focal_length=focal_length_mm,
            horizontal_aperture=horiztonal_aperture_mm,
            clipping_range=(0.01, 10000),
        )
        self.render_product = rep.create.render_product(self.camera, (config_data["WIDTH"], config_data["HEIGHT"]))
        camera_rig_path = str(rep.utils.get_node_targets(self.camera.node, "inputs:primsIn")[0])
        self.camera_path = camera_rig_path + "/Camera"

        with rep.get.prims(prim_types=["Camera"]):
            rep.modify.pose(
                rotation=rep.distribution.uniform(
                    np.array(config_data["CAMERA_ROTATION"]), np.array(config_data["CAMERA_ROTATION"])
                )
            )
        self.rig = XFormPrim(camera_rig_path)

    def _setup_collision_box(self):
        self.fov_x = 2 * math.atan(config_data["WIDTH"] / (2 * config_data["F_X"]))
        self.fov_y = 2 * math.atan(config_data["HEIGHT"] / (2 * config_data["F_Y"]))
        theta_x = self.fov_x / 2.0
        theta_y = self.fov_y / 2.0

        collision_box_width = max(2 * config_data["MAX_DISTANCE"] * math.tan(theta_x), 1.3)
        collision_box_height = max(2 * config_data["MAX_DISTANCE"] * math.tan(theta_y), 1.3)
        collision_box_depth = max(config_data["MAX_DISTANCE"] - config_data["MIN_DISTANCE"], 0.1)

        collision_box_path = "/World/collision_box"
        collision_box_translation = np.array([0, 0, (config_data["MIN_DISTANCE"] + config_data["MAX_DISTANCE"]) / 2.0])
        collision_box_rotation = np.array([0, 0, 0])
        collision_box_quat = euler_angles_to_quat(collision_box_rotation, degrees=True)

        camera_prim = world.stage.GetPrimAtPath(self.camera_path)
        center, orient = get_world_pose_from_relative(camera_prim, collision_box_translation, collision_box_quat)

        return CollisionBox(collision_box_path, "collision_box", position=center, orientation=orient,
                            width=collision_box_width, height=collision_box_height, depth=collision_box_depth)

    def _setup_distractors(self, collision_box):
        pass

    def _setup_train_objects(self):
        train_part_idx = 0

        for object in OBJECTS_TO_GENERATE:
            for prim_idx in range(object["num"]):
                prim_type = object["prim_type"]  # "pallet"

                # 기존 경로 무시하고 팔레트 URL 강제 사용
                ref_path = PALLET_USD_URL
                print(f"[SDG] Loading Custom Pallet from: {ref_path}")

                path = "/World/" + prim_type + f"_{prim_idx}"

                # 내부 구조를 추측하지 않고, 불러온 루트(path) 자체를 mesh_path로 설정
                mesh_path = path

                name = f"train_part_{train_part_idx}"
                self.train_part_mesh_path_to_prim_path_map[mesh_path] = path

                train_part = DynamicObject(
                    usd_path=ref_path,
                    prim_path=path,
                    mesh_path=mesh_path,
                    name=name,
                    position=np.array([0.0, 0.0, 0.0]),
                    scale=config_data["TRAIN_PART_SCALE"],
                    mass=10.0,
                )

                if len(train_part.prims) > 0:
                    try:
                        train_part.prims[0].GetAttribute("physics:rigidBodyEnabled").Set(True)
                    except:
                        pass

                self.train_parts.append(train_part)

                # 라벨 부착
                mesh_prim = world.stage.GetPrimAtPath(mesh_path)
                add_labels(mesh_prim, labels=[prim_type], instance_name="class")
                print(f"[SDG] Label 'class={prim_type}' added to {mesh_prim.GetPath()}")

                train_part_idx += 1

    def _setup_randomizers(self):
        import random
        from pxr import UsdGeom

        # 1. 보조 조명
        def randomize_sphere_lights():
            lights = rep.create.light(
                light_type="Sphere",
                color=rep.distribution.uniform((0.8, 0.8, 0.8), (1.0, 1.0, 1.0)),
                intensity=rep.distribution.uniform(100000, 300000),
                position=rep.distribution.uniform((-300, -300, 300), (300, 300, 600)),
                scale=rep.distribution.uniform(5, 15),
                count=2,
            )
            return lights.node

        # 2. [수정] 배경 디버깅 모드
        def randomize_environment_switch():
            # 환경 로드 실패 시 방어 코드
            if not self.env_prim_paths:
                print("[Error] 환경 파일이 로드되지 않았습니다. 기본 조명만 켭니다.")
                dummy = rep.create.light(light_type="Dome", color=(0.2, 0, 0), intensity=1000)  # 로드 실패시 '빨간색' 배경
                return dummy.node

            selected_path = next((p for p in self.env_prim_paths if "Warehouse" in p), self.env_prim_paths[0])

            print(f"[SDG] Selected Environment: {selected_path}")

            # 선택된 환경 켜기 / 나머지 끄기
            for path in self.env_prim_paths:
                prim = world.stage.GetPrimAtPath(path)
                if not prim.IsValid(): continue

                imageable = UsdGeom.Imageable(prim)
                if path == selected_path:
                    imageable.MakeVisible()
                else:
                    imageable.MakeInvisible()

            lights = rep.create.light(
                light_type="Dome",
                texture="",
                color=(1, 1, 1),
                intensity=1500  # 빛 세기도 줄임
            )

            return lights.node

        rep.randomizer.register(randomize_sphere_lights, override=True)
        rep.randomizer.register(randomize_environment_switch, override=True)

        with rep.trigger.on_frame():
            rep.randomizer.randomize_sphere_lights()
            rep.randomizer.randomize_environment_switch()

    def _setup_dome_randomizers(self):
        pass

    def randomize_movement_in_view(self, prim):
        if not self.test:
            camera_prim = world.stage.GetPrimAtPath(self.camera_path)
            rig_prim = world.stage.GetPrimAtPath(self.rig.prim_paths[0])

            translation, orientation = get_random_world_pose_in_view(
                camera_prim,
                config_data["MIN_DISTANCE"],
                config_data["MAX_DISTANCE"],
                self.fov_x,
                self.fov_y,
                config_data["FRACTION_TO_SCREEN_EDGE"],
                rig_prim,
                np.array(config_data["MIN_ROTATION_RANGE"]),
                np.array(config_data["MAX_ROTATION_RANGE"]),
            )
            translation[2] = np.random.uniform(0.7, 6.5)
        else:
            translation, orientation = np.array([0.0, 0.0, 0.5]), np.array([0.0, 0.0, 0.0, 1.0])
            translation[2] = np.random.uniform(0.5, 7.5)

        prim.set_world_poses(np.array([translation]), np.array([orientation]))

    def __iter__(self):
        return self

    def __next__(self):
        if self.cur_idx == self.num_mesh:
            pass

        for train_part in self.train_parts:
            self.randomize_movement_in_view(train_part)

        for _ in range(60):
            world.step(render=False)

        print(f"[SDG] ID: {self.cur_idx}/{self.train_size - 1}")
        rep.orchestrator.step(rt_subframes=64)
        self.cur_idx += 1

        if self.cur_idx >= self.train_size:
            print(f"[SDG] Done. Output: {self._output_folder}")
            self.last_frame_reached = True


dataset = RandomScenario(
    num_mesh=args.num_mesh,
    num_dome=args.num_dome,
    dome_interval=args.dome_interval,
    output_folder=args.output_folder,
    use_s3=args.use_s3,
    bucket=args.bucket,
    s3_region=args.s3_region,
    endpoint=args.endpoint,
    writer=args.writer.lower(),
    test=args.test,
    debug=args.debug,
)

if dataset.result:
    print("[SDG] Loading materials. Will generate data soon...")
    start_time = datetime.datetime.now()
    if dataset.train_size > 0:
        for _ in dataset:
            if dataset.last_frame_reached or dataset.exiting:
                break
    print("[SDG] Total time taken:", str(datetime.datetime.now() - start_time).split(".")[0])

kit.close()