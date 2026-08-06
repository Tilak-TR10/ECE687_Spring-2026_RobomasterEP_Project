# 🏒 Hockey Player RoboMaster EP

ECE687 Final Project
An autonomous hockey-playing RoboMaster EP robot developed using ROS2, OptiTrack motion capture, Approximate Linearization, and CLF-CBF-QP-based obstacle avoidance.

YouTube link: https://youtu.be/7PON9QAs5Pc
---

## Team Members

**Tilak Raval** (21208713)  
**Ahmed Elsaigh** (20671110)

Department of Electrical and Computer Engineering  
University of Waterloo

---

## Project Overview

The objective of this project was to develop an autonomous RoboMaster EP robot capable of:

- Navigating to a hockey stick.
- Picking up the hockey stick using the robotic arm and gripper.
- Navigating toward a known puck location.
- Positioning itself for a shot.
- Shooting the puck into a goal.
- Avoiding neighboring robots during navigation.

The system was implemented using ROS2 and OptiTrack motion capture feedback.

---

## Finite State Machine

The robot behavior was implemented using a finite state machine:

```text
GET_CHECKPOINT
        ↓
GET_STICK
        ↓
GRAB_STICK
        ↓
GET_GOAL_FAR
        ↓
GET_GOAL_NEAR
        ↓
SWING_SHOT
        ↓
DONE
```

### State Description

**GET_CHECKPOINT**  
Navigate to an intermediate waypoint near the hockey stick.

**GET_STICK**  
Perform the final approach toward the hockey stick.

**GRAB_STICK**  
Execute the arm extension, gripper closure, arm lifting, and reverse motion sequence.

**GET_GOAL_FAR**  
Navigate toward a waypoint near the goal.

**GET_GOAL_NEAR**  
Move into the final shooting configuration.

**SWING_SHOT**  
Rotate rapidly to transfer momentum through the hockey stick and strike the puck.

**DONE**  
Stop all robot motion.

---

## Approximate Linearization Controller

The RoboMaster EP was modeled as a unicycle robot.

### Virtual Point

A virtual point located in front of the robot was used to simplify control design.

```math
p_x=x+l\cos\theta
```

```math
p_y=y+l\sin\theta
```

### Proportional Controller

```math
u_x=k_p(x_d-p_x)
```

```math
u_y=k_p(y_d-p_y)
```

### Control Inputs

```math
v=u_x\cos\theta+u_y\sin\theta
```

```math
\omega=\frac{-u_x\sin\theta+u_y\cos\theta}{L_s}
```

### Parameters

- Control Frequency: **50 Hz**
- Proportional Gain: **kp = 1.2**
- Linear Velocity Limit: **±0.5 m/s**
- Angular Velocity Limit: **±1.0 rad/s**

---

## Stick Pickup Strategy

The stick pickup sequence consisted of the following actions:

1. Extend the robotic arm.
2. Close the gripper.
3. Lift the hockey stick.
4. Reverse the robot to create clearance.

A timer-based state machine was used to coordinate the arm and gripper actions.

---

## Custom Arm Simulation

Since testing time with the physical RoboMaster EP was limited, a custom arm simulator was developed using the measured dimensions of the robot arm and gripper.

The simulator was used to:

- Verify workspace reachability.
- Validate gripper placement.
- Test pickup configurations.
- Reduce hardware debugging time.

### Forward Kinematics

```math
x_e=L_1\cos\theta_1+L_2\cos(\theta_1+\theta_2)
```

```math
y_e=L_1\sin\theta_1+L_2\sin(\theta_1+\theta_2)
```

---

## CLF-CBF-QP Obstacle Avoidance

To provide collision-free navigation, a CLF-CBF-QP-inspired controller was integrated with the navigation system.

### Barrier Function

```math
h_i=(p_x-x_i)^2+(p_y-y_i)^2-d_{safe}^2
```

### Safety Constraint

```math
\dot{h}_i+\gamma h_i \ge 0
```

Negative CBF values indicate that the robot is moving toward an unsafe region.

### Optimization Problem

```math
u^*=\arg\min_u ||u-u_{nom}||^2
```

subject to

```math
\dot{h}_i+\gamma h_i \ge 0
```

A projection-based safety filter was implemented to generate safe control actions while preserving waypoint-tracking performance.

---

## Simulation Enhancements

The provided simulation environment was extended to include:

- Hockey stick visualization
- Hockey puck visualization
- Goal visualization
- Obstacle robots
- Safety circles
- Robot trajectory plotting
- Custom arm simulation

These additions significantly simplified controller development and debugging.

---

## Development Utilities

### ros_startup.sh

Automates:

- ROS daemon restart
- Cache cleanup
- Environment setup
- Workspace sourcing

Used to resolve ROS2 daemon communication issues encountered during development.

### lazy_colcon.sh

Automates:

- Workspace rebuilding
- Package sourcing
- Node execution

Used to accelerate debugging and controller development.

---

## Results

Successfully demonstrated:

✅ Navigation to hockey stick

✅ Hockey stick pickup

✅ Waypoint-based navigation

✅ Swing-shot goal scoring

✅ CLF-CBF-QP obstacle avoidance

✅ Full finite-state-machine execution

---

## Repository Contents

```text
.
├── project_node.py
├── simulator.py
├── ros_startup.sh
├── lazy_colcon.sh
├── All_Important_Commands.txt
├── trajectory_plot.png
├── arm Movement trajectory.png
├── Arm Initial position.png
├── Robomaster_ECE687_Report.pdf
└── README.md
```

---

## Build

```bash
colcon build
source install/setup.bash
```

## Run

```bash
ros2 run ece687_project_pkg project_node
```

or

```bash
./lazy_colcon.sh
```

---

## Team Contributions

### Tilak Raval

- Resolved ROS2 environment and daemon communication issues.
- Developed automation utilities (ros_startup.sh and lazy_colcon.sh).
- Designed and implemented the CLF-CBF-QP obstacle avoidance framework.
- Developed the custom arm simulation environment.
- Generated simulation plots used in the report.
- Implemented waypoint-based positioning strategies.
- Maintained project documentation and development notes.
- Prepared major portions of the project report.

### Ahmed Elsaigh

- Implemented the approximate linearization navigation controller.
- Integrated arm and gripper control.
- Added hockey objects to the simulation environment.
- Developed stick pickup logic using experimentally determined offsets.
- Implemented waypoint navigation and swing-shot behavior.
- Performed hardware testing and system validation.
- Prepared major portions of the project report.

### Joint Contributions

- Experimental testing and validation.
- System integration and debugging.
- Final controller tuning.
- Discussion and final report preparation.

---

## Additional Resources

Project videos, screenshots, simulation results, source code, and development progress are available in this repository.

---

## Acknowledgements

This project was completed as part of **ECE687 – Collaborative Autonomous and Intelligent Systems** at the University of Waterloo.

Microsoft Copilot was used as a supplementary tool for debugging assistance, LaTeX formatting, proofreading, and technical writing support.
