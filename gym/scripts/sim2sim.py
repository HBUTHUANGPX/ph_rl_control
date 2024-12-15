# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.


import math
import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
from gym import LEGGED_GYM_ROOT_DIR
from gym.envs import Ph_1_ControllerCfg
import torch


class cmd:
    vx = 0.4
    vy = 0.0
    dyaw = 0.0


class Sim2simCfg(Ph_1_ControllerCfg):
    class env(Ph_1_ControllerCfg.env):
        num_single_obs = 21
        frame_stack = 51
        num_observations = num_single_obs * frame_stack

    class sim_config:
        mujoco_model_path = (
            # f"{LEGGED_GYM_ROOT_DIR}/resources/robots/hector_description/mjcf/rl_fixedbase.xml"
            f"{LEGGED_GYM_ROOT_DIR}/resources/robots/hector_description/mjcf/rl.xml"
        )
        sim_duration = 60.0
        dt = 0.001
        decimation = 10

    class robot_config:
        kps = np.array([20, 4000, 0] * (2), dtype=np.double)
        kds = np.array([1.0, 80, 1.0] * (2), dtype=np.double)
        tau_limit = 200.0 * np.ones(6, dtype=np.double)


def quaternion_to_euler_array(quat):
    # Ensure quaternion is in the correct format [x, y, z, w]
    x, y, z, w = quat

    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)

    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)

    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)

    # Returns roll, pitch, yaw in a NumPy array in radians
    return np.array([roll_x, pitch_y, yaw_z])


def get_obs(data):
    """Extracts an observation from the mujoco data structure"""
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    omega = data.sensor("angular-velocity").data.astype(np.double)
    gvec = r.apply(np.array([0.0, 0.0, -1.0]), inverse=True).astype(np.double)
    return (q, dq, quat, v, omega, gvec)


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    # print("p:", (target_q - q) * kp )
    # print("d", (target_dq - dq) * kd)
    return (target_q - q) * kp + (target_dq - dq) * kd


def quat_rotate_inverse(q, v):
    q_w = q[-1]
    q_vec = q[:3]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c


def reset_simulation(data, model):
    """
    Resets the simulation to its initial state.

    Args:
        data: The MjData object containing the simulation state.
        model: The MjModel object containing the simulation model.

    Returns:
        None
    """
    # Reset the simulation state
    mujoco.mj_resetData(model, data)
    # Optionally, you can set specific initial conditions here
    # For example, setting initial joint positions or velocities
    # data.qpos[:] = np.zeros(7 + 6, dtype=np.double)
    # data.qpos[7 + 1] = 0.08
    # data.qpos[7 + 4] = 0.08
    # data.qvel[:] = initial_velocities
    mujoco.mj_forward(model, data)  # Recompute positions and velocities

def init_hist_obs(cfg:Sim2simCfg):
    hist_obs = deque()
    for _ in range(cfg.env.frame_stack):
        hist_obs.append(np.zeros([1, cfg.env.num_single_obs], dtype=np.double))
    return hist_obs


def run_mujoco(policy, cfg: Sim2simCfg):
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        policy: The policy used for controlling the simulation.
        cfg: The configuration object containing simulation settings.

    Returns:
        None
    """
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    viewer = mujoco_viewer.MujocoViewer(model, data)

    target_q = np.zeros((cfg.env.num_actuators), dtype=np.double)
    target_dq = np.zeros((cfg.env.num_actuators), dtype=np.double)
    action = np.zeros((cfg.env.num_actuators), dtype=np.double)

    hist_obs = init_hist_obs(cfg)

    count_lowlevel = 0

    for _ in tqdm(
        range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)),
        desc="Simulating...",
    ):

        # Obtain an observation
        q, dq, quat, v, omega, gvec = get_obs(data)
        q = q[-cfg.env.num_actuators :]
        dq = dq[-cfg.env.num_actuators :]
        _q = quat
        _v = np.array([0.0, 0.0, -1.0])
        projected_gravity = quat_rotate_inverse(_q, _v)
        if count_lowlevel % cfg.sim_config.decimation == 0:
            obs = np.zeros([1, cfg.env.num_single_obs], dtype=np.float32)
            eu_ang = quaternion_to_euler_array(quat)
            eu_ang[eu_ang > math.pi] -= 2 * math.pi

            obs[0, 0] = cmd.vx * cfg.scaling.commands
            obs[0, 1] = cmd.vy * cfg.scaling.commands
            obs[0, 2] = cmd.dyaw * cfg.scaling.commands
            obs[0, 3:6] = omega * cfg.scaling.base_ang_vel
            obs[0, 6:9] = torch.tensor(projected_gravity, dtype=torch.double)  # 3
            obs[0, 9] = math.sin(2 * math.pi * count_lowlevel * cfg.sim_config.dt / 0.1)
            obs[0, 10] = math.cos(
                2 * math.pi * count_lowlevel * cfg.sim_config.dt / 0.1
            )
            hip_ind = [0, 3]
            leg_ind = [1, 4]
            whl_ind = [2, 5]
            obs[0, 11:13] = q[hip_ind] * cfg.scaling.dof_pos
            obs[0, 13:15] = dq[hip_ind] * cfg.scaling.dof_vel
            obs[0, 15:17] = q[leg_ind] * cfg.scaling.dof_pos
            obs[0, 17:19] = dq[leg_ind] * cfg.scaling.dof_vel
            obs[0, 19:21] = dq[whl_ind] * cfg.scaling.dof_vel

            obs = np.clip(
                obs,
                -20,
                20,
            )
            hist_obs.append(obs)
            hist_obs.popleft()

            policy_input = np.zeros([1, cfg.env.num_observations], dtype=np.float32)
            for i in range(cfg.env.frame_stack):
                policy_input[
                    0, i * cfg.env.num_single_obs : (i + 1) * cfg.env.num_single_obs
                ] = hist_obs[i][0, :]
            action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()
            action = np.clip(
                action, -cfg.scaling.clip_actions, cfg.scaling.clip_actions
            )
            target_q = action
            # target_q = action * 0
            # target_q[hip_ind[0]] = target_q[hip_ind[1]]

            # 髋关节

            # 平移关节
            # target_q[leg_ind] = target_q[leg_ind] /15

            derta_z = target_q[1] / 15
            derta_roll = target_q[4] / 30

            derta_roll_z = np.sin(derta_roll)

            target_q[leg_ind[0]] = derta_z - derta_roll_z
            target_q[leg_ind[1]] = derta_z + derta_roll_z
            # target_q[leg_ind] = 0

            # 轮
            w_z = target_dq[2]
            w_z = np.clip(
                w_z, -cfg.commands.ranges.yaw_vel, cfg.commands.ranges.yaw_vel
            )
            v_x = target_dq[5]
            v_x = np.clip(
                v_x,
                cfg.commands.ranges.lin_vel_x[0],
                cfg.commands.ranges.lin_vel_x[1],
            )
            target_dq[whl_ind[0]] = (v_x - w_z * 0.26) / 0.15
            target_dq[whl_ind[1]] = (v_x + w_z * 0.26) / 0.15
            target_dq[whl_ind] = np.clip(target_q[whl_ind], -10, 10)
        tau = pd_control(
            target_q, q, cfg.robot_config.kps, target_dq, dq, cfg.robot_config.kds
        )  # Calc torques
        tau = np.clip(
            tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit
        )  # Clamp torques

        # 遍历所有接触点
        d = data
        ground_geom_id = 0
        has_contact = False
        for i in range(d.ncon):
            contact = d.contact[i]

            # 获取接触的两个几何体 ID
            geom1 = contact.geom1
            geom2 = contact.geom2

            # 检查是否有一个几何体是地面
            if geom1 == ground_geom_id or geom2 == ground_geom_id:
                # 如果是，检查另一个几何体是否属于机器人
                # 你需要根据你的模型文件来确认哪些几何体属于机器人
                if geom1 != ground_geom_id:
                    robot_geom_id = geom1
                else:
                    robot_geom_id = geom2

                # 在这里，你可以添加更多逻辑来判断 robot_geom_id 是否属于机器人
                # 例如，检查 robot_geom_id 是否在一个特定的范围内
                if robot_geom_id == 18 or robot_geom_id == 24:
                    ...
                else:
                    has_contact = True

        if has_contact:
            print("Unwanted contact detected, resetting simulation.")
            reset_simulation(data, model)
            hist_obs = init_hist_obs(cfg)
            
            continue  # Skip the rest of the loop and start from the initial state

        for i in range(3):
            tmptau = tau[i]
            tau[i] = tau[i + 3]
            tau[i + 3] = tmptau
        data.ctrl = tau
        # print(tau)

        mujoco.mj_step(model, data)
        viewer.render()
        count_lowlevel += 1

    viewer.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Deployment script.")
    parser.add_argument(
        "--load_model",
        type=str,
        required=False,
        help="Run to load from.",
        default=f"{LEGGED_GYM_ROOT_DIR}/logs/ph_Controller/exported/policy.pt",
    )
    parser.add_argument("--terrain", action="store_true", help="terrain or plane")
    args = parser.parse_args()

    policy = torch.jit.load(args.load_model)
    run_mujoco(policy, Sim2simCfg())
