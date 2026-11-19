
import mujoco
import mujoco.viewer
import time
import numpy as np
from pathlib import Path

# Load UR5e from Menagerie
robot_path = Path(
    "C:/Users/kpatnaik/Documents/ECE_3641/models/mujoco_menagerie/universal_robots_ur5e/scene.xml"
)

model = mujoco.MjModel.from_xml_path(str(robot_path))
data = mujoco.MjData(model)

# Joint names
joint_names = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint"
]

# Same joint angles as DH calculation
q = [0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0]

# Set joint angles
for i in range(6):

    joint_id = model.joint(joint_names[i]).id
    qpos_id = model.jnt_qposadr[joint_id]

    data.qpos[qpos_id] = q[i]

# Update forward kinematics
mujoco.mj_forward(model, data)

# End-effector position
ee_id = model.site("attachment_site").id

p_ee = data.site_xpos[ee_id]
R_ee = data.site_xmat[ee_id].reshape(3, 3)

T = np.eye(4)
T[:3, :3] = R_ee
T[:3, 3] = p_ee

print("MuJoCo end-effector transformation:")
print(np.round(T, 4))

print("MuJoCo end-effector position:")
print(np.round(p_ee, 4))

######################################################
#### Open MuJoCo viewer

with mujoco.viewer.launch_passive(model, data) as viewer:
# with mujoco.viewer.launch(model, data) as viewer:

    # Display coordinate frames
    with viewer.lock():
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY

    # Keep viewer open
    while viewer.is_running():
        viewer.sync()
        time.sleep(0.05)
