# Robot Manipulators: Kinematics, Motion Planning, and Reinforcement Learning

This repository provides introductory Python examples for learning robotic manipulation using MuJoCo. The examples are designed to help students understand fundamental robotics concepts through mathematical modeling, simulation, and learning-based control.

## Topics Covered

- Forward Kinematics (FK)
- Denavit–Hartenberg (DH) Convention
- Inverse Kinematics (IK)
- Jacobian-Based Control
- Motion Planning and Trajectory Generation
- Reinforcement Learning (RL)
- Serial and Parallel Manipulators

## Repository Contents

| File | Description |
|---|---|
| `fk_UR5e_dh.py` | Forward kinematics of the UR5e using DH parameters. |
| `fk_UR5e_mujoco.py` | Forward kinematics of the UR5e using MuJoCo. |
| `basic_RL.py` | Introductory implementation of reinforcement learning for robotic control. |
| `fivebar.xml` | MuJoCo model of a five-bar planar parallel manipulator. |
| `test_fivebar.py` | Load, visualize, and interact with the five-bar manipulator. |

Additional examples will be added as the course progresses.

## Installation

### 1. Prerequisites

- Python 3.11
- Visual Studio Code
- Python extension for VS Code

### 2. Install Python Libraries

```bash
pip install mujoco numpy matplotlib
```

### 3. Download MuJoCo Menagerie

Download the ZIP file from the [MuJoCo Menagerie repository](https://github.com/google-deepmind/mujoco_menagerie).

Extract the robot models into a `models` folder and ensure that the model paths in the Python scripts match your local directory structure.

## Running the Examples

Open the repository folder in VS Code and run the desired Python script from the terminal.

For example:

```bash
python fk_UR5e_dh.py
```

To visualize the five-bar parallel manipulator:

```bash
python test_fivebar.py
```

To explore reinforcement learning:

```bash
python basic_RL.py
```

## Learning Objectives

- Understand the relationship between joint configurations and end-effector motion.
- Develop and validate mathematical models of robotic manipulators.
- Simulate serial and parallel robotic mechanisms.
- Explore conventional and learning-based approaches to robot control.
- Understand how robots can learn motion strategies through interaction with their environment.

## Educational Purpose

These examples are intentionally simplified for teaching and experimentation. Students are encouraged to modify robot configurations, control inputs, target positions, and reward functions to investigate robot behavior.

Developed as part of robotics and Physical AI teaching activities at the University of Michigan–Dearborn.
