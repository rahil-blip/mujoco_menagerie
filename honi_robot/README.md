# Honi Robot - Mobile Manipulator (MJCF)

Combined MuJoCo model of the **Honi Robot**: a mobile manipulator for autonomous store cleaning operations.

## Overview

The Honi Robot combines:
- **AgileX Ranger Mini v3** - 4-wheel steered mobile base
- **AgileX PiPER** - 6-DOF robotic arm with parallel-jaw gripper
- **Custom spine** - Cylindrical column connecting base to arm

## Architecture

```
ranger_base (freejoint)
  +-- chassis (box 500x350x200mm)
  +-- fr_wheel (steering + drive)
  +-- fl_wheel (steering + drive)
  +-- rl_wheel (steering + drive)
  +-- rr_wheel (steering + drive)
  +-- spine_mount
      +-- spine (cylinder r=30mm, h=300mm)
      +-- arm_mount
          +-- arm_plate (150x100x20mm)
          +-- piper_attachment (site)
          +-- head_camera (site)

base_link (PiPER arm, welded to arm_mount)
  +-- link1 -> link2 -> ... -> link6
  +-- link7 (left finger)
  +-- link8 (right finger)
```

## Joints & Actuators

| Component | Joints | Actuator Type |
|-----------|--------|---------------|
| Base wheels (x4) | steering (hinge) + drive (hinge) | position + velocity |
| PiPER arm (6-DOF) | joint1-joint6 (hinge) | position (from piper.xml) |
| Gripper | joint7/joint8 (slide, coupled) | position (from piper.xml) |

**Total**: 8 base joints + 8 arm joints = 16 joints, 15 actuators

## Sensors

- IMU (accelerometer + gyro) at arm mount
- Joint position sensors for all steering joints
- Joint velocity sensors for all drive joints
- Arm joint position sensors (joint1-6 + gripper)
- Gripper touch sensor
- Camera frame pose (position + quaternion)
- Magnetometer at base
- Velocimeter at base

## Store Environment

The scene includes a simulated convenience store aisle with:
- Two shelf units with multiple shelves at different heights
- Graspable items (boxes, cans) on shelves with free joints
- A fallen item on the floor as a cleaning target
- Store walls forming an aisle corridor
- Realistic floor friction for wheel traction

## Testing

Run the validation test suite:

```bash
cd honi_robot
python test_honi.py
```

Tests include: home keyframe verification, forward driving, arm reach check, and sensor readout validation.

## Usage

```python
import mujoco

model = mujoco.MjModel.from_xml_path('honi_robot/honi_scene.xml')
data = mujoco.MjData(model)

# Reset to home position
mujoco.mj_resetDataKeyframe(model, data, 0)

# Step simulation
mujoco.mj_step(model, data)
```

## Dependencies

- MuJoCo 2.3.4 or later
- PiPER arm model (`../agilex_piper/`)

## Files

- `honi_scene.xml` - Combined MJCF scene
- `README.md` - This file
- `honi_robot.yaml` - Robot configuration and metadata
- `test_honi.py` - Simulation test and validation script

## License

See repository root LICENSE file.
