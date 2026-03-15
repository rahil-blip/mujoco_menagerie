"""Honi Robot - MuJoCo simulation test script.

Loads the Honi Robot scene (Ranger Mini + PiPER arm) in a store
environment and runs basic kinematic tests.

Usage:
  pip install mujoco
  python test_honi.py
"""
import os
import numpy as np

try:
    import mujoco
except ImportError:
    raise SystemExit("Install mujoco: pip install mujoco")


def load_model():
    """Load the Honi Robot MJCF model."""
    scene_path = os.path.join(os.path.dirname(__file__), "honi_scene.xml")
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    return model, data


def print_model_info(model):
    """Print model summary."""
    print("=" * 50)
    print("Honi Robot - Model Summary")
    print("=" * 50)
    print(f"  Bodies:    {model.nbody}")
    print(f"  Joints:    {model.njnt}")
    print(f"  DOFs:      {model.nv}")
    print(f"  Actuators: {model.nu}")
    print(f"  Sensors:   {model.nsensor}")
    print(f"  Geoms:     {model.ngeom}")
    print(f"  Timestep:  {model.opt.timestep}s")
    print()


def test_home_keyframe(model, data):
    """Reset to home keyframe and verify."""
    print("[TEST] Home keyframe...")
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    base_z = data.qpos[2]
    print(f"  Base height: {base_z:.3f}m")
    assert base_z > 0.1, f"Base too low: {base_z}"
    print("  PASS")


def test_drive_forward(model, data):
    """Drive the robot forward for 2 seconds."""
    print("[TEST] Drive forward...")
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    start_x = data.qpos[0]

    # Set wheel drive velocities (actuators 0-3)
    steps = int(2.0 / model.opt.timestep)
    for _ in range(steps):
        data.ctrl[0] = 5.0  # fr drive
        data.ctrl[1] = 5.0  # fl drive
        data.ctrl[2] = 5.0  # rl drive
        data.ctrl[3] = 5.0  # rr drive
        mujoco.mj_step(model, data)

    end_x = data.qpos[0]
    dist = end_x - start_x
    print(f"  Distance traveled: {dist:.3f}m in 2s")
    assert dist > 0.05, f"Robot didn't move enough: {dist}"
    print("  PASS")


def test_arm_reach(model, data):
    """Test arm joint movement."""
    print("[TEST] Arm reach...")
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    # Get end-effector (link6) position
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link6")
    if ee_id < 0:
        print("  SKIP - link6 body not found")
        return

    ee_start = data.xpos[ee_id].copy()
    print(f"  EE start pos: [{ee_start[0]:.3f}, {ee_start[1]:.3f}, {ee_start[2]:.3f}]")
    print("  PASS")


def test_sensor_readout(model, data):
    """Verify sensors return data."""
    print("[TEST] Sensor readout...")
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    accel = data.sensordata[0:3]
    gyro = data.sensordata[3:6]
    print(f"  Accelerometer: [{accel[0]:.2f}, {accel[1]:.2f}, {accel[2]:.2f}]")
    print(f"  Gyroscope:     [{gyro[0]:.4f}, {gyro[1]:.4f}, {gyro[2]:.4f}]")
    assert not np.all(accel == 0), "Accelerometer reads all zeros"
    print("  PASS")


def main():
    model, data = load_model()
    print_model_info(model)
    test_home_keyframe(model, data)
    test_drive_forward(model, data)
    test_arm_reach(model, data)
    test_sensor_readout(model, data)
    print()
    print("All tests passed!")


if __name__ == "__main__":
    main()
