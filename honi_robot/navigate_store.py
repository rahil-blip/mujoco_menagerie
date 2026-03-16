"""Honi Robot - Store Navigation Controller.

Waypoint-following navigation through store aisles.
Drives the Ranger Mini base to waypoints while keeping the arm stowed.

Usage:
    pip install mujoco
    python navigate_store.py
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


def get_base_pose(model, data):
    """Return (x, y, yaw) of the base."""
    jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "base_free")
    adr = model.jnt_qposadr[jnt_id]
    x, y = data.qpos[adr], data.qpos[adr + 1]
    # quaternion -> yaw
    qw, qx, qy, qz = data.qpos[adr + 3:adr + 7]
    yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy**2 + qz**2))
    return x, y, yaw


def drive_to_waypoint(model, data, target_x, target_y, speed=3.0, turn_gain=5.0):
    """Drive the base toward (target_x, target_y). Returns True when reached."""
    x, y, yaw = get_base_pose(model, data)
    dx = target_x - x
    dy = target_y - y
    dist = math.sqrt(dx**2 + dy**2)

    if dist < 0.15:  # close enough
        data.ctrl[0:4] = 0.0
        return True

    # desired heading
    desired_yaw = math.atan2(dy, dx)
    yaw_err = desired_yaw - yaw
    # wrap to [-pi, pi]
    yaw_err = (yaw_err + math.pi) % (2 * math.pi) - math.pi

    # differential drive: forward + turn
    forward = speed
    turn = turn_gain * yaw_err

    # mecanum wheel mapping: all wheels drive forward, steering for turn
    # actuators 0-3: fr_drive, fl_drive, rl_drive, rr_drive
    # actuators 4-7: fr_steer, fl_steer, rl_steer, rr_steer (if present)
    left_vel = forward - turn
    right_vel = forward + turn
    data.ctrl[0] = right_vel  # fr drive
    data.ctrl[1] = left_vel   # fl drive
    data.ctrl[2] = left_vel   # rl drive
    data.ctrl[3] = right_vel  # rr drive

    return False


def navigate_route(model, data, waypoints, max_time=60.0):
    """Follow a sequence of waypoints. Returns list of (time, x, y) logs."""
    kf_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "honi_home")
    mujoco.mj_resetDataKeyframe(model, data, kf_id)
    mujoco.mj_forward(model, data)

    log = []
    wp_idx = 0
    t = 0.0
    dt = model.opt.timestep

    print(f"Navigating {len(waypoints)} waypoints...")
    x0, y0, yaw0 = get_base_pose(model, data)
    print(f"  Start: ({x0:.2f}, {y0:.2f}), yaw={math.degrees(yaw0):.1f} deg")

    while wp_idx < len(waypoints) and t < max_time:
        tx, ty = waypoints[wp_idx]
        reached = drive_to_waypoint(model, data, tx, ty)

        mujoco.mj_step(model, data)
        t += dt

        if int(t / dt) % 500 == 0:  # log every 500 steps
            x, y, yaw = get_base_pose(model, data)
            log.append((t, x, y))

        if reached:
            x, y, yaw = get_base_pose(model, data)
            print(f"  WP {wp_idx}: ({tx:.2f}, {ty:.2f}) reached at t={t:.2f}s, pos=({x:.2f}, {y:.2f})")
            wp_idx += 1

    if wp_idx >= len(waypoints):
        print(f"  Route complete in {t:.2f}s")
    else:
        print(f"  Timeout after {t:.2f}s at waypoint {wp_idx}")

    return log, wp_idx == len(waypoints)


# Store aisle waypoints (x, y) in meters
# Layout: shelves along y-axis, aisles along x-axis
STORE_ROUTE = [
    (0.5, 0.0),    # move forward from start
    (0.5, -1.0),   # turn into aisle 1
    (1.5, -1.0),   # traverse aisle 1
    (1.5, 0.0),    # end of aisle 1
    (1.5, 1.0),    # turn into aisle 2
    (0.5, 1.0),    # traverse aisle 2 back
    (0.0, 0.0),    # return to start
]


def main():
    model, data = load_model()
    print("=" * 50)
    print("Honi Robot - Store Navigation")
    print("=" * 50)
    print(f"  Model: {model.nbody} bodies, {model.nu} actuators")
    print(f"  Timestep: {model.opt.timestep}s")
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
        print(f"  X range: [{min(xs):.2f}, {max(xs):.2f}]")
        print(f"  Y range: [{min(ys):.2f}, {max(ys):.2f}]")
        total_dist = sum(
            math.sqrt((log[i][1] - log[i-1][1])**2 + (log[i][2] - log[i-1][2])**2)
            for i in range(1, len(log))
        )
        print(f"  Approx distance traveled: {total_dist:.2f}m")


if __name__ == "__main__":
    main()
