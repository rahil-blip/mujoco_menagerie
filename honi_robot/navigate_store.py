"""Honi Robot v2.1 - Store Navigation Controller.

Waypoint-following navigation through store aisles.
Drives the mecanum base to waypoints with telescoping spine coordination.
Automatically adjusts spine height for shelf interaction zones.

Usage:
    pip install mujoco numpy
    python navigate_store.py
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
SPINE_RETRACTED = 0.0    # spine fully retracted (travel mode)
SPINE_LOW_SHELF = 0.1    # spine extension for low shelves
SPINE_MID_SHELF = 0.25   # spine extension for mid shelves
SPINE_HIGH_SHELF = 0.4   # spine fully extended for high shelves
SPINE_ACTUATOR = "spine_extend"  # actuator name in v2.1 scene


def load_model():
    scene_path = os.path.join(os.path.dirname(__file__), SCENE_FILE)
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    return model, data


def get_base_pose(model, data):
    """Return (x, y, yaw) of the base."""
    jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "base_free")
    adr = model.jnt_qposadr[jnt_id]
    x, y = data.qpos[adr], data.qpos[adr + 1]
    # quaternion -> yaw
    qw, qx, qy, qz = data.qpos[adr + 3:adr + 7]
    yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy**2 + qz**2))
    return x, y, yaw


def set_spine_height(model, data, extension):
    """Set the telescoping spine extension (0.0 to 0.4m)."""
    extension = max(0.0, min(0.4, extension))
    act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, SPINE_ACTUATOR)
    if act_id >= 0:
        data.ctrl[act_id] = extension
    return extension


def get_spine_height(model, data):
    """Read current spine extension from joint sensor."""
    jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "spine_extend")
    if jnt_id >= 0:
        adr = model.jnt_qposadr[jnt_id]
        return data.qpos[adr]
    return 0.0


def drive_to_waypoint(model, data, target_x, target_y, speed=3.0, turn_gain=5.0):
    """Drive the base toward (target_x, target_y). Returns True when reached."""
    x, y, yaw = get_base_pose(model, data)
    dx = target_x - x
    dy = target_y - y
    dist = math.sqrt(dx**2 + dy**2)

    if dist < 0.15:  # close enough
        # Stop all wheel actuators
        for i in range(4):
            act_name = f"wheel_{['fr','fl','rl','rr'][i]}"
            act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
            if act_id >= 0:
                data.ctrl[act_id] = 0.0
        return True

    # desired heading
    desired_yaw = math.atan2(dy, dx)
    yaw_err = desired_yaw - yaw
    yaw_err = (yaw_err + math.pi) % (2 * math.pi) - math.pi

    # differential drive: forward + turn
    forward = speed
    turn = turn_gain * yaw_err

    # mecanum wheel mapping
    left_vel = forward - turn
    right_vel = forward + turn

    wheel_names = ["wheel_fr", "wheel_fl", "wheel_rl", "wheel_rr"]
    wheel_vels = [right_vel, left_vel, left_vel, right_vel]
    for name, vel in zip(wheel_names, wheel_vels):
        act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if act_id >= 0:
            data.ctrl[act_id] = vel

    return False


def navigate_route(model, data, waypoints, max_time=60.0):
    """Follow waypoints with spine coordination. Returns log and success flag."""
    # Reset to home keyframe
    kf_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if kf_id < 0:
        kf_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "honi_home")
    if kf_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, kf_id)
    mujoco.mj_forward(model, data)

    log = []
    wp_idx = 0
    t = 0.0
    dt = model.opt.timestep

    print(f"Navigating {len(waypoints)} waypoints...")
    x0, y0, yaw0 = get_base_pose(model, data)
    print(f"  Start: ({x0:.2f}, {y0:.2f}), yaw={math.degrees(yaw0):.1f} deg")
    print(f"  Spine: {get_spine_height(model, data):.3f}m")

    while wp_idx < len(waypoints) and t < max_time:
        wp = waypoints[wp_idx]
        tx, ty = wp[0], wp[1]
        spine_target = wp[2] if len(wp) > 2 else SPINE_RETRACTED

        # Set spine height for this waypoint
        set_spine_height(model, data, spine_target)

        reached = drive_to_waypoint(model, data, tx, ty)
        mujoco.mj_step(model, data)
        t += dt

        if int(t / dt) % 500 == 0:
            x, y, yaw = get_base_pose(model, data)
            spine_h = get_spine_height(model, data)
            log.append((t, x, y, spine_h))

        if reached:
            x, y, yaw = get_base_pose(model, data)
            spine_h = get_spine_height(model, data)
            print(f"  WP {wp_idx}: ({tx:.2f}, {ty:.2f}) reached at t={t:.2f}s, "
                  f"pos=({x:.2f}, {y:.2f}), spine={spine_h:.3f}m")
            wp_idx += 1

    if wp_idx >= len(waypoints):
        print(f"  Route complete in {t:.2f}s")
    else:
        print(f"  Timeout after {t:.2f}s at waypoint {wp_idx}")

    return log, wp_idx == len(waypoints)


# Store aisle waypoints: (x, y, spine_extension)
# Layout: shelves along y-axis, aisles along x-axis
STORE_ROUTE = [
    (0.5, 0.0, SPINE_RETRACTED),     # move forward, spine down
    (0.5, -1.0, SPINE_RETRACTED),    # turn into aisle 1
    (1.0, -1.0, SPINE_LOW_SHELF),    # approach low shelf, extend spine
    (1.5, -1.0, SPINE_MID_SHELF),    # mid-aisle, mid shelf height
    (1.5, 0.0, SPINE_RETRACTED),     # end aisle 1, retract
    (1.5, 1.0, SPINE_RETRACTED),     # turn into aisle 2
    (1.0, 1.0, SPINE_HIGH_SHELF),    # high shelf zone
    (0.5, 1.0, SPINE_RETRACTED),     # traverse back, retract
    (0.0, 0.0, SPINE_RETRACTED),     # return to start
]


def main():
    model, data = load_model()

    print("=" * 60)
    print("Honi Robot v2.1 - Store Navigation")
    print("=" * 60)
    print(f"  Model: {model.nbody} bodies, {model.nu} actuators")
    print(f"  Timestep: {model.opt.timestep}s")
    print(f"  Scene: {SCENE_FILE}")
    print()

    log, success = navigate_route(model, data, STORE_ROUTE)
    print()

    if success:
        print("Navigation: PASS")
    else:
        print("Navigation: FAIL")

    # Summary stats
    if log:
        xs = [p[1] for p in log]
        ys = [p[2] for p in log]
        spines = [p[3] for p in log]
        print(f"  X range: [{min(xs):.2f}, {max(xs):.2f}]")
        print(f"  Y range: [{min(ys):.2f}, {max(ys):.2f}]")
        print(f"  Spine range: [{min(spines):.3f}, {max(spines):.3f}]m")
        total_dist = sum(
            math.sqrt((log[i][1] - log[i-1][1])**2 + (log[i][2] - log[i-1][2])**2)
            for i in range(1, len(log))
        )
        print(f"  Approx distance traveled: {total_dist:.2f}m")


if __name__ == "__main__":
    main()
