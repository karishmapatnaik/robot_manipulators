
import mujoco
import mujoco.viewer
import numpy as np
import time

# -----------------------------
# 1. Load UR5e
# -----------------------------
path = "models/mujoco_menagerie/universal_robots_ur5e/scene.xml"

model = mujoco.MjModel.from_xml_path(path)
data = mujoco.MjData(model)

joint_names = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint"
]

q_ids = [model.joint(name).qposadr[0] for name in joint_names]

q_start = np.array([0, -1.57, 1.57, -1.57, -1.57, 0])
q_goal = np.array([0.5, -1.57, 1.57, -1.57, -1.57, 0])

def set_robot(q):
    data.qpos[q_ids] = q
    mujoco.mj_forward(model, data)

def get_ee():
    return data.site("attachment_site").xpos.copy()

# Obtain target position using FK
set_robot(q_goal)
target = get_ee()
set_robot(q_start)

print("Target:", target)

# -----------------------------
# 2. RL policy
# -----------------------------
# State = joint angles + EE position error (9)
# Action = change in 6 joint angles
# Policy mean = W @ state

rng = np.random.default_rng(0)

W = np.zeros((6, 9))
learning_rate = 0.0005
sigma = 0.15
step_size = 0.05

episodes = 3000
max_steps = 80
gamma = 0.99

def get_state():
    q = data.qpos[q_ids].copy()
    error = target - get_ee()
    return np.concatenate((q, error))

def policy(state):
    mean = W @ state
    action = mean + rng.normal(0, sigma, 6)
    return action, mean

# -----------------------------
# 3. Train using REINFORCE
# -----------------------------
for episode in range(episodes):
    set_robot(q_start)

    states = []
    actions = []
    means = []
    rewards = []

    for step in range(max_steps):
        state = get_state()

        action, mean = policy(state)

        # Small joint-angle update
        dq = step_size * np.tanh(action)
        q_new = data.qpos[q_ids].copy() + dq

        # Respect joint limits
        for i, name in enumerate(joint_names):
            jid = model.joint(name).id
            if model.jnt_limited[jid]:
                low, high = model.jnt_range[jid]
                q_new[i] = np.clip(q_new[i], low, high)

        set_robot(q_new)

        distance = np.linalg.norm(target - get_ee())
        reward = -distance

        states.append(state)
        actions.append(action)
        means.append(mean)
        rewards.append(reward)

        if distance < 0.02:
            break

    # Discounted returns
    returns = np.zeros(len(rewards))
    G = 0

    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * G
        returns[t] = G

    # Normalize returns
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    # REINFORCE gradient update
    gradient = np.zeros_like(W)

    for state, action, mean, G in zip(
        states, actions, means, returns
    ):
        grad_log_policy = np.outer(
            (action - mean) / sigma**2, state
        )
        gradient += G * grad_log_policy

    W += learning_rate * gradient / len(rewards)

    if episode % 100 == 0:
        print(
            f"Episode {episode}, "
            f"Distance = {distance:.4f} m"
        )

# -----------------------------
# 4. Test learned policy
# -----------------------------
print("\nTesting trained policy")

set_robot(q_start)

with mujoco.viewer.launch_passive(model, data) as viewer:
    for step in range(max_steps):
        state = get_state()

        # Deterministic action (no exploration)
        action = W @ state

        dq = step_size * np.tanh(action)
        q_new = data.qpos[q_ids].copy() + dq

        for i, name in enumerate(joint_names):
            jid = model.joint(name).id
            if model.jnt_limited[jid]:
                low, high = model.jnt_range[jid]
                q_new[i] = np.clip(q_new[i], low, high)

        with viewer.lock():
            set_robot(q_new)

        viewer.sync()
        time.sleep(0.05)

        distance = np.linalg.norm(target - get_ee())
        print(f"Step {step}: distance = {distance:.4f} m")

        if distance < 0.02:
            print("Goal reached!")
            break

    while viewer.is_running():
        viewer.sync()
        time.sleep(0.05)
