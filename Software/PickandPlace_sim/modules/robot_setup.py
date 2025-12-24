def set_gripper_state(robot, state: bool):
    robot.gripper._enabled = state
    print("Gripper =", state)
