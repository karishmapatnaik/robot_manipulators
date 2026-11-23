
import mujoco
import mujoco.viewer
import numpy as np
import time

# Load robot
model = mujoco.MjModel.from_xml_path("fivebar.xml")
data = mujoco.MjData(model)

# Initialize simulation
mujoco.mj_forward(model, data)

# Get end-effector sites
left_id = model.site("tip_left").id
right_id = model.site("tip_right").id

print("MuJoCo model loaded successfully!")
print("Number of joints:", model.njnt)
print("Number of actuators:", model.nu)

# Launch viewer
with mujoco.viewer.launch(model, data) as viewer:

    while viewer.is_running():

        # Keep motors at zero torque initially
        data.ctrl[:] = 0

        # Step simulation
        mujoco.mj_step(model, data)

        # Read end-effector positions
        p_left = data.site_xpos[left_id].copy()
        p_right = data.site_xpos[right_id].copy()

        # Check loop closure error
        error = np.linalg.norm(p_left - p_right)

        if int(data.time * 100) % 100 == 0:
            print(
                f"Time: {data.time:.2f} | "
                f"EE: {p_left.round(3)} | "
                f"Closure error: {error:.6f} m"
            )

        viewer.sync()
        time.sleep(model.opt.timestep)
