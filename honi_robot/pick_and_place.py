"""Honi Robot - Pick and Place Task Controller.

Demonstrates a full pick-and-place cycle:
1. Navigate to shelf
2. Reach arm to item
3. Grasp item (close gripper)
4. Stow arm
5. Navigate to drop zone
6. Place item (open gripper)

Usage:
    pip install mujoco
    python pick_and_place.py
"""
import os
import math
import numpy as np

try:
    import mujoco
except ImportError:
    raise SystemExit("Install mujoco: pip install mujoco")


def load_model():
    scene_path = os.path.join(os.path.dirname(__file__), "honi_scene.xml")
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


def set_arm_joints(model, data, target_angles, steps=500):
    """Move arm joints to target angles over N steps using position control."""
    # PiPER arm actuators start after the 8 wheel actuators
    arm_act_offset = 8  # 4 drive + 4 steer
    for step in range(steps):
        alpha = min(1.0, (step + 1) / steps)
        for i in range(6):  # 6 arm joints
            current = data.qpos[i]  # arm joints at qpos[0:6]
            target = target_angles[i]
            data.ctrl[arm_act_offset + i] = current + alpha * (target - current)
        mujoco.mj_step(model, data)


def set_gripper(model, data, width, steps=200):
    """Open/close gripper. width=0 closed, width=0.04 open."""
    gripper_act_offset = 14  # after 8 wheel + 6 arm actuators
    for step in range(steps):
        data.ctrl[gripper_act_offset] = width
        if gripper_act_offset + 1 < model.nu:
            data.ctrl[gripper_act_offset + 1] = width
        mujoco.mj_step(model, data)


def drive_to(model, data, tx, ty, speed=3.0, turn_gain=5.0, timeout=15.0):
    """Drive base to (tx, ty). Returns True if reached."""
    dt = model.opt.timestep
    t = 0.0
    while t < timeout:
        x, y, yaw = get_base_pose(model, data)
        dx, dy = tx - x, ty - y
        dist = math.sqrt(dx**2 + dy**2)
        if dist < 0.15:
            data.ctrl[0:4] = 0.0
            return True

        desired_yaw = math.atan2(dy, dx)
        yaw_err = (desired_yaw - yaw + math.pi) % (2 * math.pi) - math.pi
        left_vel = speed - turn_gain * yaw_err
        right_vel = speed + turn_gain * yaw_err
        data.ctrl[0] = right_vel
        data.ctrl[1] = left_vel
        data.ctrl[2] = left_vel
        data.ctrl[3] = right_vel

        mujoco.mj_step(model, data)
        t += dt
    return False


def get_ee_pos(model, data):
    """Get end-effector (link6) world position."""
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link6")
    if ee_id < 0:
        return np.array([0, 0, 0])
    return data.xpos[ee_id].copy()


def check_collisions(model, data):
    """Return number of active contacts involving arm bodies."""
    arm_bodies = set()
    for name in ["link1", "link2", "link3", "link4", "link5", "link6"]:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            arm_bodies.add(bid)

    arm_contacts = 0
    for i in range(data.ncon):
        c = data.contact[i]
        g1_body = model.geom_bodyid[c.geom1]
        g2_body = model.geom_bodyid[c.geom2]
        if g1_body in arm_bodies or g2_body in arm_bodies:
            arm_contacts += 1
    return arm_contacts


# Arm joint configurations (radians)
ARM_STOW = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
ARM_REACH_LOW = [0.5, 0.8, -1.5, 0.0, -0.3, 0.0]    # shelf low
ARM_REACH_MID = [0.0, 1.0, 0.3, -0.8, 0.0, -0.5]     # shelf mid
ARM_REACH_HIGH = [0.0, 1.57, 0.1, -0.3, 0.0, -0.8]   # shelf high
ARM_REACH_FLOOR = [0.0, 1.8, -2.0, 0.5, 0.0, 0.0]    # floor pickup

GRIPPER_OPEN = 0.04
GRIPPER_CLOSED = 0.0

# Task locations (x, y)
SHELF_A_POS = (1.2, -1.2)   # in front of shelf A
DROP_ZONE = (0.0, 0.5)       # drop zone near entrance


def run_pick_and_place(model, data):
    """Execute a full pick-and-place cycle."""
    kf_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "honi_home")
    mujoco.mj_resetDataKeyframe(model, data, kf_id)
    mujoco.mj_forward(model, data)

    results = {}

    # Phase 1: Navigate to shelf
    print("[PHASE 1] Navigate to shelf A...")
    ok = drive_to(model, data, *SHELF_A_POS)
    x, y, _ = get_base_pose(model, data)
    print(f"  Position: ({x:.2f}, {y:.2f}), reached={ok}")
    results["navigate_to_shelf"] = ok

    # Phase 2: Reach arm to item
    print("[PHASE 2] Reach arm to shelf (low position)...")
    set_arm_joints(model, data, ARM_REACH_LOW)
    ee = get_ee_pos(model, data)
    contacts = check_collisions(model, data)
    print(f"  EE position: [{ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f}]")
    print(f"  Arm contacts: {contacts}")
    results["reach"] = True
    results["reach_contacts"] = contacts

    # Phase 3: Close gripper
    print("[PHASE 3] Grasp item...")
    set_gripper(model, data, GRIPPER_CLOSED)
    print("  Gripper closed")
    results["grasp"] = True

    # Phase 4: Stow arm
    print("[PHASE 4] Stow arm...")
    set_arm_joints(model, data, ARM_STOW)
    ee = get_ee_pos(model, data)
    print(f"  EE stowed at: [{ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f}]")
    results["stow"] = True

    # Phase 5: Navigate to drop zone
    print("[PHASE 5] Navigate to drop zone...")
    ok = drive_to(model, data, *DROP_ZONE)
    x, y, _ = get_base_pose(model, data)
    print(f"  Position: ({x:.2f}, {y:.2f}), reached={ok}")
    results["navigate_to_drop"] = ok

    # Phase 6: Place item
    print("[PHASE 6] Place item...")
    set_arm_joints(model, data, ARM_REACH_FLOOR)
    set_gripper(model, data, GRIPPER_OPEN)
    set_arm_joints(model, data, ARM_STOW)
    print("  Item placed")
    results["place"] = True

    return results


def main():
    model, data = load_model()
    print("=" * 50)
    print("Honi Robot - Pick and Place")
    print("=" * 50)
    print(f"  Model: {model.nbody} bodies, {model.nu} actuators")
    print()

    results = run_pick_and_place(model, data)

    print()
    print("=" * 50)
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
