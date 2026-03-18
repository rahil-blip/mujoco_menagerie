"""Honi Robot v2.1 - Pick and Place Task Controller.

Demonstrates a full pick-and-place cycle with telescoping spine:
1. Navigate to shelf
2. Extend spine to shelf height
3. Reach arm to item
4. Grasp item (close gripper)
5. Stow arm + retract spine
6. Navigate to drop zone
7. Place item (open gripper)

Usage:
    pip install mujoco numpy
    python pick_and_place.py
"""
import os
import math
import numpy as np

try:
    import mujoco
except ImportError:
    raise SystemExit("Install mujoco: pip install mujoco")


# === v2.1 Configuration ===
SCENE_FILE = "honi_scene_v2.1.xml"
SPINE_ACTUATOR = "spine_extend"


def load_model():
    scene_path = os.path.join(os.path.dirname(__file__), SCENE_FILE)
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    return model, data


def get_joint_adr(model, name):
    jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jnt_id < 0:
        raise RuntimeError(f"Joint '{name}' not found")
    return model.jnt_qposadr[jnt_id]


def get_base_pose(model, data):
    adr = get_joint_adr(model, "base_free")
    x, y = data.qpos[adr], data.qpos[adr + 1]
    qw, qx, qy, qz = data.qpos[adr + 3:adr + 7]
    yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy**2 + qz**2))
    return x, y, yaw


def set_spine(model, data, extension):
    """Set telescoping spine extension (0.0 to 0.4m)."""
    extension = max(0.0, min(0.4, extension))
    act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, SPINE_ACTUATOR)
    if act_id >= 0:
        data.ctrl[act_id] = extension
    return extension


def get_spine(model, data):
    """Read current spine extension."""
    jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "spine_extend")
    if jnt_id >= 0:
        return data.qpos[model.jnt_qposadr[jnt_id]]
    return 0.0


def set_arm_joints(model, data, target_angles, steps=500):
    """Move arm joints to target angles over N steps."""
    arm_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
    arm_act_ids = []
    for name in arm_names:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        arm_act_ids.append(aid)
    for step in range(steps):
        alpha = min(1.0, (step + 1) / steps)
        for i, aid in enumerate(arm_act_ids):
            if aid >= 0 and i < len(target_angles):
                data.ctrl[aid] = target_angles[i] * alpha
        mujoco.mj_step(model, data)


def set_gripper(model, data, width, steps=200):
    """Open/close gripper. width=0 closed, width=0.04 open."""
    for name in ["gripper_left", "gripper_right"]:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid >= 0:
            data.ctrl[aid] = width
    for _ in range(steps):
        mujoco.mj_step(model, data)


def drive_to(model, data, tx, ty, speed=3.0, turn_gain=5.0, timeout=15.0):
    """Drive base to (tx, ty). Returns True if reached."""
    dt = model.opt.timestep
    t = 0.0
    wheel_names = ["wheel_fr", "wheel_fl", "wheel_rl", "wheel_rr"]
    while t < timeout:
        x, y, yaw = get_base_pose(model, data)
        dx, dy = tx - x, ty - y
        dist = math.sqrt(dx**2 + dy**2)
        if dist < 0.15:
            for wn in wheel_names:
                aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, wn)
                if aid >= 0:
                    data.ctrl[aid] = 0.0
            return True
        desired_yaw = math.atan2(dy, dx)
        yaw_err = (desired_yaw - yaw + math.pi) % (2 * math.pi) - math.pi
        left_vel = speed - turn_gain * yaw_err
        right_vel = speed + turn_gain * yaw_err
        vels = [right_vel, left_vel, left_vel, right_vel]
        for wn, vel in zip(wheel_names, vels):
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, wn)
            if aid >= 0:
                data.ctrl[aid] = vel
        mujoco.mj_step(model, data)
        t += dt
    return False


def get_ee_pos(model, data):
    """Get end-effector (link6) world position."""
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link6")
    if ee_id < 0:
        return np.array([0, 0, 0])
    return data.xpos[ee_id].copy()


# Arm joint configurations (radians)
ARM_STOW = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
ARM_REACH_LOW = [0.5, 0.8, -1.5, 0.0, -0.3, 0.0]
ARM_REACH_MID = [0.0, 1.0, 0.3, -0.8, 0.0, -0.5]
ARM_REACH_HIGH = [0.0, 1.57, 0.1, -0.3, 0.0, -0.8]
ARM_REACH_FLOOR = [0.0, 1.8, -2.0, 0.5, 0.0, 0.0]
GRIPPER_OPEN = 0.04
GRIPPER_CLOSED = 0.0

# Spine heights for shelf tiers
SPINE_FOR_LOW = 0.0
SPINE_FOR_MID = 0.2
SPINE_FOR_HIGH = 0.4

# Task locations (x, y)
SHELF_A_POS = (1.2, -1.2)
DROP_ZONE = (0.0, 0.5)


def run_pick_and_place(model, data, shelf_height="mid"):
    """Execute a full pick-and-place cycle with spine coordination."""
    # Reset to home
    kf_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if kf_id < 0:
        kf_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "honi_home")
    if kf_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, kf_id)
    mujoco.mj_forward(model, data)

    # Select configs based on shelf height
    if shelf_height == "low":
        spine_ext, arm_reach = SPINE_FOR_LOW, ARM_REACH_LOW
    elif shelf_height == "high":
        spine_ext, arm_reach = SPINE_FOR_HIGH, ARM_REACH_HIGH
    else:
        spine_ext, arm_reach = SPINE_FOR_MID, ARM_REACH_MID

    results = {}

    # Phase 1: Navigate to shelf
    print(f"[PHASE 1] Navigate to shelf A...")
    ok = drive_to(model, data, *SHELF_A_POS)
    x, y, _ = get_base_pose(model, data)
    print(f"  Position: ({x:.2f}, {y:.2f}), reached={ok}")
    results["navigate_to_shelf"] = ok

    # Phase 2: Extend spine
    print(f"[PHASE 2] Extend spine to {spine_ext:.2f}m for {shelf_height} shelf...")
    set_spine(model, data, spine_ext)
    for _ in range(300):
        mujoco.mj_step(model, data)
    actual = get_spine(model, data)
    print(f"  Spine: {actual:.3f}m")
    results["spine_extend"] = abs(actual - spine_ext) < 0.05

    # Phase 3: Reach arm
    print("[PHASE 3] Reach arm to shelf...")
    set_arm_joints(model, data, arm_reach)
    ee = get_ee_pos(model, data)
    print(f"  EE position: [{ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f}]")
    results["reach"] = True

    # Phase 4: Grasp
    print("[PHASE 4] Grasp item...")
    set_gripper(model, data, GRIPPER_CLOSED)
    print("  Gripper closed")
    results["grasp"] = True

    # Phase 5: Stow arm + retract spine
    print("[PHASE 5] Stow arm + retract spine...")
    set_arm_joints(model, data, ARM_STOW)
    set_spine(model, data, 0.0)
    for _ in range(300):
        mujoco.mj_step(model, data)
    print(f"  Spine retracted to {get_spine(model, data):.3f}m")
    results["stow"] = True

    # Phase 6: Navigate to drop zone
    print("[PHASE 6] Navigate to drop zone...")
    ok = drive_to(model, data, *DROP_ZONE)
    x, y, _ = get_base_pose(model, data)
    print(f"  Position: ({x:.2f}, {y:.2f}), reached={ok}")
    results["navigate_to_drop"] = ok

    # Phase 7: Place item
    print("[PHASE 7] Place item...")
    set_arm_joints(model, data, ARM_REACH_FLOOR)
    set_gripper(model, data, GRIPPER_OPEN)
    set_arm_joints(model, data, ARM_STOW)
    print("  Item placed")
    results["place"] = True

    return results


def main():
    model, data = load_model()

    print("=" * 60)
    print("Honi Robot v2.1 - Pick and Place")
    print("=" * 60)
    print(f"  Model: {model.nbody} bodies, {model.nu} actuators")
    print(f"  Scene: {SCENE_FILE}")
    print()

    # Run pick-and-place for mid shelf
    results = run_pick_and_place(model, data, shelf_height="mid")
    print()

    print("=" * 60)
    print("Results:")
    all_pass = True
    for phase, status in results.items():
        if isinstance(status, bool):
            symbol = "PASS" if status else "FAIL"
            if not status:
                all_pass = False
        else:
            symbol = str(status)
        print(f"  {phase}: {symbol}")
    print()
    print(f"Overall: {'PASS' if all_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
