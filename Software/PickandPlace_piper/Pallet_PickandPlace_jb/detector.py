#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import sys
import cv2
import numpy as np
import pyzed.sl as sl

# 전역 변수
point_cloud = sl.Mat()
clicked_point = None
click_signal = False


def on_mouse(event, x, y, flags, param):
    global clicked_point, click_signal
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)
        click_signal = True


def main():
    # 1. ZED 카메라 설정
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL  # 가장 정밀한 깊이 모드
    init_params.coordinate_units = sl.UNIT.METER  # 단위: 미터
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP  # Y가 위쪽

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"ZED Open Failed: {err}")
        return

    # 카메라 원본 해상도 가져오기
    camera_config = zed.get_camera_information().camera_configuration
    res = camera_config.resolution

    image_sl = sl.Mat()
    runtime_params = sl.RuntimeParameters()

    print("\n=======================================================")
    print(" 🖱️  Click-Based Calibration Tool (Fixed)")
    print(" 1. 화면에서 측정하고 싶은 지점(모서리 등)을 '클릭'하세요.")
    print(" 2. 터미널에 출력된 좌표를 기록하세요.")
    print(" 3. 'q'를 누르면 종료합니다.")
    print("=======================================================\n")

    cv2.namedWindow("ZED Calibration - Click Point")
    cv2.setMouseCallback("ZED Calibration - Click Point", on_mouse)

    global click_signal, clicked_point

    while True:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # 데이터 획득 (해상도 인자 제거 -> 기본 해상도 사용)
            zed.retrieve_image(image_sl, sl.VIEW.LEFT)
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

            # 이미지 변환 (OpenCV용)
            image_cv = image_sl.get_data()
            image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGRA2BGR)  # 채널 맞춤

            # 클릭 이벤트 처리
            if click_signal and clicked_point is not None:
                x, y = clicked_point

                # 이미지 해상도와 Point Cloud 해상도가 같으므로 스케일링 불필요
                # (만약 다를 경우를 대비해 비율 계산 코드는 유지)
                pc_width = point_cloud.get_width()
                pc_height = point_cloud.get_height()
                img_width = image_cv.shape[1]
                img_height = image_cv.shape[0]

                scale_x = pc_width / img_width
                scale_y = pc_height / img_height

                pc_x = int(x * scale_x)
                pc_y = int(y * scale_y)

                # 범위 체크
                if 0 <= pc_x < pc_width and 0 <= pc_y < pc_height:
                    # 깊이 값 가져오기
                    err, value = point_cloud.get_value(pc_x, pc_y)

                    if err == sl.ERROR_CODE.SUCCESS:
                        p_x, p_y, p_z, _ = value

                        # 유효한 값인지 확인 (NaN 체크)
                        if np.isnan(p_x) or np.isnan(p_y) or np.isnan(p_z):
                            print(f"\r⚠️ [Invalid] 깊이 값을 읽을 수 없는 영역입니다. 다시 클릭하세요.", end="")
                        else:
                            # 좌표 출력
                            print("\n" + "=" * 40)
                            print(f"📍 [Selected Point] (Meter)")
                            print(f"   X: {p_x:.5f}")
                            print(f"   Y: {p_y:.5f}")
                            print(f"   Z: {p_z:.5f}")
                            print("=" * 40 + "\n")

                            # 화면 표시
                            cv2.circle(image_cv, (x, y), 5, (0, 0, 255), -1)
                            cv2.putText(image_cv, f"[{p_x:.2f}, {p_y:.2f}, {p_z:.2f}]", (x + 10, y),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            cv2.imshow("ZED Calibration - Click Point", image_cv)
                            cv2.waitKey(500)
                else:
                    print("범위 밖 클릭")

                click_signal = False  # 리셋

            # 십자선 그리기
            h, w = image_cv.shape[:2]
            cv2.line(image_cv, (w // 2 - 20, h // 2), (w // 2 + 20, h // 2), (0, 255, 0), 1)
            cv2.line(image_cv, (w // 2, h // 2 - 20), (w // 2, h // 2 + 20), (0, 255, 0), 1)

            cv2.imshow("ZED Calibration - Click Point", image_cv)

            key = cv2.waitKey(10)
            if key == ord('q') or key == 27:
                break

    zed.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()