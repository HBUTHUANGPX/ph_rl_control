"""
Hierarchical structure for Deep Stepper for Humanoid
1) Low-level policy: Step controller trained by PPO
    - It is divided into two section. (1) Only one-step controller (2) Continuous-step controller
2) High-level policy: Step planner trained by SAC

Purpose: Given a base velocity command (linear x,y velocity, angular velocity), 
         robot determines stepping locations to follow the commanded velocity

This script serves as a Low-level policy which actuate the robot to take a step

* All variables are calculated w.r.t world frame
* However, when the variables are put into observation, it is converted w.r.t base frame

post_physics_step 更新状态变量和历史数据
    _post_physics_step_callback 进行resample_commands、measure_heights、push_robots
        _update_robot_states 更新机器人的状态变量
            _calculate_foot_states 计算足端状态
        _calculate_CoM 计算机器人的质心
        _calculate_raibert_heuristic 计算 Raibert 启发式步态命令
        _calculate_ICP 计算机器人的瞬时捕获点
        _measure_success_rate 测量步态命令的成功率
        _update_commands 根据当前的步态状态和环境条件，生成新的步态命令
            _generate_step_command_by_3DLIPM_XCoM 生成新的步态命令
            _update_LIPM_CoM 根据当前的步态状态和环境条件，生成新的步态命令
            _adjust_foot_collision 将脚移动到边界上的最近点，以避免两个脚之间的碰撞
    check_termination 根据接触力的大小判断机器人是否需要reset
_reset_system
    _calculate_foot_states 计算足端状态
"""

import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from gym.envs.ph_1.ph_1_controller_config import (
    Ph_1_ControllerCfg,
    Ph_1_ControllerRunnerCfg,
)
from gym.utils.math import *
from gym.envs import LeggedRobot
from isaacgym import gymapi, gymutil
import numpy as np
from typing import Tuple, Dict
from .ph_1_utils import (
    FootStepGeometry,
    SimpleLineGeometry,
    VelCommandGeometry,
    smart_sort,
)
from gym.utils import XCoMKeyboardInterface
from scipy.signal import correlate
import torch.nn.functional as F


class Ph_1_Controller(LeggedRobot):
    cfg: Ph_1_ControllerCfg
    run_cfg: Ph_1_ControllerRunnerCfg

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.run_cfg = Ph_1_ControllerRunnerCfg()

    def _setup_keyboard_interface(self):
        self.keyboard_interface = XCoMKeyboardInterface(self)

    def _init_buffers(self):
        super()._init_buffers()
        # * Robot states
        self.base_height = self.root_states[:, 2:3]
        self.right_hip_pos = self.rigid_body_state[
            :, self.rigid_body_idx["r_mt_link"], :3
        ]
        self.left_hip_pos = self.rigid_body_state[
            :, self.rigid_body_idx["l_mt_link"], :3
        ]
        self.CoM = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.foot_states = torch.zeros(
            self.num_envs,
            len(self.feet_ids),
            7,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )  # num_envs x (right & left foot) x (x, y, z, quat)
        self.foot_states_right = torch.zeros(
            self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False
        )  # num_envs x (x, y, z, heading, projected_gravity)
        self.foot_states_left = torch.zeros(
            self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False
        )  # num_envs x (x, y, z, heading, projected_gravity)
        self.foot_heading = torch.zeros(
            self.num_envs,
            len(self.feet_ids),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )  # num_envs x (right & left foot heading)
        self.foot_projected_gravity = torch.stack(
            (self.gravity_vec, self.gravity_vec), dim=1
        )  # (num_envs x 2 x 3), [0., 0., -1.]
        self.foot_contact = torch.zeros(
            self.num_envs,
            len(self.feet_ids),
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )  # contacts on right & left sole

        self.ankle_vel_history = torch.zeros(
            self.num_envs,
            len(self.feet_ids),
            2 * 3,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.base_heading = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.base_lin_vel_world = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False
        )

        # * Step commands
        self.step_commands = torch.zeros(
            self.num_envs,
            len(self.feet_ids),
            3,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )  # (right & left foot) x (x, y, heading) wrt base x,y-coordinate
        self.step_commands_right = torch.zeros(
            self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False
        )  # (right foot) x (x, y, heading) wrt base x,y-coordinate
        self.step_commands_left = torch.zeros(
            self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False
        )  # (left foot) x (x, y, heading) wrt base x,y-coordinate
        self.foot_on_motion = torch.zeros(
            self.num_envs,
            len(self.feet_ids),
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )  # True foot is on command
        self.step_period = torch.zeros(
            self.num_envs, 1, dtype=torch.long, device=self.device, requires_grad=False
        )
        self.full_step_period = torch.zeros(
            self.num_envs, 1, dtype=torch.long, device=self.device, requires_grad=False
        )  # full_step_period = 2 * step_period
        self.ref_foot_trajectories = torch.zeros(
            self.num_envs,
            len(self.feet_ids),
            3,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )  # (right & left foot) x (x, y, heading) wrt base x,y-coordinate

        # * Step states
        self.current_step = torch.zeros(
            self.num_envs,
            len(self.feet_ids),
            3,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )  # (right & left foot) x (x, y, heading) wrt base x,y-coordinate
        self.prev_step_commands = torch.zeros(
            self.num_envs,
            len(self.feet_ids),
            3,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )  # (right & left foot) x (x, y, heading) wrt base x,y-coordinate
        self.step_location_offset = torch.zeros(
            self.num_envs,
            len(self.feet_ids),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )  # num_envs x (right & left foot)
        self.step_heading_offset = torch.zeros(
            self.num_envs,
            len(self.feet_ids),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )  # num_envs x (right & left foot)
        self.succeed_step_radius = torch.tensor(
            self.cfg.commands.succeed_step_radius,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.succeed_step_angle = torch.tensor(
            np.deg2rad(self.cfg.commands.succeed_step_angle),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.semi_succeed_step = torch.zeros(
            self.num_envs,
            len(self.feet_ids),
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )  # whether foot is close to step_command
        self.succeed_step = torch.zeros(
            self.num_envs,
            len(self.feet_ids),
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )  # whether steps are successful
        self.already_succeed_step = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False
        )  # check if robot succeed given step command
        self.had_wrong_contact = torch.zeros(
            self.num_envs,
            len(self.feet_ids),
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )  # check if it has had wrong contact
        self.step_stance = torch.zeros(
            self.num_envs, 1, dtype=torch.long, device=self.device, requires_grad=False
        )  # step_stance = previous step_period

        # * Others
        self.update_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device, requires_grad=False
        )  # Number of transition since the beginning of the episode
        self.update_commands_ids = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False
        )  # envs whose step commands are updated
        self.phase_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device, requires_grad=False
        )  # Number of phase progress
        self.update_phase_ids = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False
        )  # envs whose phases are updated
        self.phase = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False
        )  # phase of current step in a whole gait cycle
        self.ICP = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False
        )  # Instantaneous Capture Point (ICP) for the robot
        self.raibert_heuristic = torch.zeros(
            self.num_envs,
            len(self.feet_ids),
            3,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )  # step_location & angle by raibert heuristic
        self.w = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )  # eigenfrequency of the inverted pendulum
        self.step_length = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False
        )  # step length
        self.step_width = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False
        )  # step width
        self.dstep_length = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False
        )  # desired step length
        self.dstep_width = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False
        )  # desired step width
        self.support_foot_pos = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False
        )  # position of the support foot
        self.prev_support_foot_pos = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False
        )  # position of the support foot
        self.LIPM_CoM = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False
        )  # base position of the Linear Inverted Pendulum model

        # * Observation variables
        self.phase_sin = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.phase_cos = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.contact_schedule = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False
        )

        # * Dof control type mask
        self.dof_ctr_type_mask = torch.zeros(
            self.num_envs, 1, dtype=torch.int32, device=self.device, requires_grad=False
        )
        mask_list = [
            1 if control_type == "pos" else 0
            for control_type in self.cfg.init_state.dof_ctr_type.values()
        ]

        # 将 mask_list 转换为张量，并扩展到 (num_envs, len(mask_list)) 的形状
        dof_ctr_type_mask = torch.tensor(
            mask_list, dtype=torch.int32, device=self.device, requires_grad=False
        )
        self.dof_ctr_type_mask_pos = dof_ctr_type_mask.unsqueeze(0).expand(
            self.num_envs, -1
        )
        self.dof_ctr_type_mask_vel = 1 - self.dof_ctr_type_mask_pos
        # print(self.dof_ctr_type_mask_pos)
        # print(self.dof_ctr_type_mask_vel)
        hip_ind = [0, 3]
        leg_ind = [1, 4]
        whe_ind = [2, 5]
        self.hip_pos = self.dof_pos[:, hip_ind]
        self.hip_vel = self.dof_vel[:, hip_ind]
        self.leg_length = self.dof_pos[:, leg_ind]
        self.leg_vel = self.dof_vel[:, leg_ind]
        self.wheel_vel = self.dof_vel[:, whe_ind]

    def _compute_torques(self):
        self.desired_pos_target = self.dof_pos_target + self.default_dof_pos
        # TODO 241211
        # 将 desired_pos_target 改为 desired_action_target
        q = self.dof_pos.clone()
        qd = self.dof_vel.clone()
        q_des = self.desired_pos_target * self.dof_ctr_type_mask_pos
        qd_des = self.desired_pos_target * self.dof_ctr_type_mask_vel
        kp = self.p_gains.clone()
        kd = self.d_gains.clone()
        ############################# 髋关节
        hip_indices = [0, 3]
        # q_des[:, hip_indices[0]] = q_des[:, hip_indices[1]]

        ############################# 平移关节
        p_indices = [1, 4]
        # print(self.desired_pos_target[0, p_indices])
        derta_z = self.desired_pos_target[:, 1] / 30
        derta_roll = self.desired_pos_target[:, 4] / 30

        derta_roll_z = torch.sin(derta_roll)

        q_des[:, p_indices[0]] = derta_z - derta_roll_z
        q_des[:, p_indices[1]] = derta_z + derta_roll_z
        # q_des = torch.clip(q_des, -3, 3)

        ############################ 轮
        wh_indices = [2, 5]
        # print(self.desired_pos_target[0, wh_indices])
        w_z = self.desired_pos_target[:, 2] / 30
        # print(self.cfg.commands.ranges.name)
        # print("w_z: ",w_z)
        # print("self.cfg.commands.ranges.yaw_vel: ",self.cfg.commands.ranges.yaw_vel)
        w_z = torch.clip(
            w_z, -self.cfg.commands.ranges.yaw_vel, self.cfg.commands.ranges.yaw_vel
        )
        # print(w_z)
        v_x = self.desired_pos_target[:, 5] / 30
        # print(v_x)
        v_x = torch.clip(
            v_x,
            self.cfg.commands.ranges.lin_vel_x[0],
            self.cfg.commands.ranges.lin_vel_x[1],
        )
        # print(v_x)
        qd_des[:, wh_indices[0]] = (v_x - w_z * 0.26) / 0.15
        qd_des[:, wh_indices[1]] = (v_x + w_z * 0.26) / 0.15

        # qd_des[:, wh_indices] = qd_des[:, wh_indices]

        torques = kp * (q_des - q) + kd * (qd_des - qd)
        # print(kp)
        # print(kd)
        # print(q_des[0,:])
        # print(q[0,wh_indices])
        # print("qd:\n", qd[0, wh_indices])
        # print("qd_des:\n", qd_des[0, :])
        # print(torques[0, :])
        torques = torch.clip(torques, -self.torque_limits, self.torque_limits)
        # print(torques[0,:])
        tt = torques.view(self.torques.shape)
        # print(tt[0])
        # print(self._reward_base_axis_xy_orientation())
        # print(self._reward_ang_vel_xy())
        # print(self.wheel_vel)
        return tt

    def _resample_commands(self, env_ids):
        """Randomly select foot step commands one/two steps ahead"""
        super()._resample_commands(env_ids)

        self.step_period[env_ids] = torch.randint(
            low=self.command_ranges["sample_period"][0],
            high=self.command_ranges["sample_period"][1],
            size=(len(env_ids), 1),
            device=self.device,
        )
        self.full_step_period = 2 * self.step_period

        self.step_stance[env_ids] = torch.clone(self.step_period[env_ids])

        # * Randomly select the desired step width
        self.dstep_width[env_ids] = torch_rand_float(
            self.command_ranges["dstep_width"][0],
            self.command_ranges["dstep_width"][1],
            (len(env_ids), 1),
            self.device,
        )

    def _reset_system(self, env_ids):
        super()._reset_system(env_ids)
        # * Robot states
        self.foot_projected_gravity[env_ids, 0] = self.gravity_vec[env_ids]
        self.foot_projected_gravity[env_ids, 1] = self.gravity_vec[env_ids]

        # * Step commands
        self.step_commands[env_ids, 0] = self.env_origins[env_ids] + torch.tensor(
            [0.0, -0.1, 0.0], device=self.device
        )  # right foot initializatoin
        self.step_commands[env_ids, 1] = self.env_origins[env_ids] + torch.tensor(
            [0.0, 0.1, 0.0], device=self.device
        )  # left foot initializatoin
        self.foot_on_motion[env_ids, 0] = False
        self.foot_on_motion[env_ids, 1] = True  # When resample, left feet is swing foot

        # * Step states
        self.current_step[env_ids] = torch.clone(
            self.step_commands[env_ids]
        )  # current_step initializatoin
        self.prev_step_commands[env_ids] = torch.clone(self.step_commands[env_ids])
        self.semi_succeed_step[env_ids] = False
        self.succeed_step[env_ids] = False
        self.already_succeed_step[env_ids] = False
        self.had_wrong_contact[env_ids] = False

        # * Others
        self.update_count[env_ids] = 0
        self.update_commands_ids[env_ids] = False
        self.phase_count[env_ids] = 0
        self.update_phase_ids[env_ids] = False
        self.phase[env_ids] = 0
        self.ICP[env_ids] = 0.0
        self.raibert_heuristic[env_ids] = 0.0
        self.w[env_ids] = 0.0
        self.dstep_length[env_ids] = self.cfg.commands.dstep_length
        self.dstep_width[env_ids] = self.cfg.commands.dstep_width

    # ============================post_physics_step==============================
    def post_physics_step(self):
        # print("post_physics_step")
        super().post_physics_step()
        # self._log_info()

    # ======================== _post_physics_step_callback ================
    def _post_physics_step_callback(self):
        # print("_post_physics_step_callback")
        super()._post_physics_step_callback()

        self._update_robot_states()
        # self._log_info()

    def _update_robot_states(self):
        # print("_update_robot_states")
        """Update robot state variables"""
        self.base_height[:] = self.root_states[:, 2:3]
        forward = quat_apply(self.base_quat, self.forward_vec)
        self.base_heading = torch.atan2(forward[:, 1], forward[:, 0]).unsqueeze(1)
        self.right_hip_pos = self.rigid_body_state[
            :, self.rigid_body_idx["r_mt_link"], :3
        ]
        self.left_hip_pos = self.rigid_body_state[
            :, self.rigid_body_idx["l_mt_link"], :3
        ]

        right_foot_forward = quat_apply(self.foot_states[:, 0, 3:7], self.forward_vec)
        left_foot_forward = quat_apply(self.foot_states[:, 1, 3:7], self.forward_vec)
        right_foot_heading = wrap_to_pi(
            torch.atan2(right_foot_forward[:, 1], right_foot_forward[:, 0])
        )
        left_foot_heading = wrap_to_pi(
            torch.atan2(left_foot_forward[:, 1], left_foot_forward[:, 0])
        )
        self.foot_heading[:, 0] = right_foot_heading
        self.foot_heading[:, 1] = left_foot_heading

        self.foot_projected_gravity[:, 0] = quat_rotate_inverse(
            self.foot_states[:, 0, 3:7], self.gravity_vec
        )
        self.foot_projected_gravity[:, 1] = quat_rotate_inverse(
            self.foot_states[:, 1, 3:7], self.gravity_vec
        )

        self.update_count += 1
        self.phase_count += 1
        self.phase += 1 / self.full_step_period

        # * Ground truth foot contact
        self.foot_contact = torch.gt(self.contact_forces[:, self.feet_ids, 2], 0)

        # * Phase-based foot contact
        self.contact_schedule = self.smooth_sqr_wave(self.phase)

        # * Update current step
        current_step_masked = self.current_step[self.foot_contact]
        current_step_masked[:, :2] = self.foot_states[self.foot_contact][:, :2]
        current_step_masked[:, 2] = self.foot_heading[self.foot_contact]
        self.current_step[self.foot_contact] = current_step_masked

        naxis = 3
        self.ankle_vel_history[:, 0, naxis:] = self.ankle_vel_history[:, 0, :naxis]
        self.ankle_vel_history[:, 0, :naxis] = self.rigid_body_state[
            :, self.rigid_body_idx["r_wheel_link"], 7:10
        ]
        self.ankle_vel_history[:, 1, naxis:] = self.ankle_vel_history[:, 1, :naxis]
        self.ankle_vel_history[:, 1, :naxis] = self.rigid_body_state[
            :, self.rigid_body_idx["l_wheel_link"], 7:10
        ]
        self.phase_sin = torch.sin(2 * torch.pi * self.phase)
        self.phase_cos = torch.cos(2 * torch.pi * self.phase)

        self.base_lin_vel_world = self.root_states[:, 7:10].clone()
        hip_ind = [0, 3]
        leg_ind = [1, 4]
        whe_ind = [2, 5]
        self.hip_pos = self.dof_pos[:, hip_ind]
        self.hip_vel = self.dof_vel[:, hip_ind]
        self.leg_length = self.dof_pos[:, leg_ind]
        self.leg_vel = self.dof_vel[:, leg_ind]
        self.wheel_vel = self.dof_vel[:, whe_ind]
        # print(self.phase_sin)

    # ======================== _post_physics_step_callback ================
    def check_termination(self):
        # print("check_termination")
        """Check if environments need to be reset"""
        # * Termination for contact
        term_contact = torch.norm(
            self.contact_forces[:, self.termination_contact_indices, :], dim=-1
        )
        self.terminated = torch.any((term_contact > 1.0), dim=1)

        # * Termination for velocities, orientation, and low height
        # self.terminated |= torch.any(
        #     torch.norm(self.base_lin_vel, dim=-1, keepdim=True) > 5.0, dim=1
        # )
        # self.terminated |= torch.any(
        #     torch.norm(self.base_ang_vel, dim=-1, keepdim=True) > 30.0, dim=1
        # )
        # self.terminated |= torch.any(
        #     torch.abs(self.projected_gravity[:, 0:1]) > 0.7, dim=1
        # )
        # self.terminated |= torch.any(
        #     torch.abs(self.projected_gravity[:, 1:2]) > 0.7, dim=1
        # )
        # self.terminated |= torch.any(self.base_pos[:, 2:3] < 0.2, dim=1)
        self.terminated |= torch.any(self.base_pos[:, 2:3] > 0.4, dim=1)

        # * No terminal reward for time-outs
        self.timed_out = self.episode_length_buf > self.max_episode_length

        self.reset_buf = self.terminated | self.timed_out

    # ============================post_physics_step==============================

    def _set_obs_variables(self):
        # print("_set_obs_variables")
        self.foot_states_right[:, :3] = quat_rotate_inverse(
            self.base_quat, self.foot_states[:, 0, :3] - self.base_pos
        )
        self.foot_states_left[:, :3] = quat_rotate_inverse(
            self.base_quat, self.foot_states[:, 1, :3] - self.base_pos
        )
        self.foot_states_right[:, 3] = wrap_to_pi(
            self.foot_heading[:, 0] - self.base_heading.squeeze(1)
        )
        self.foot_states_left[:, 3] = wrap_to_pi(
            self.foot_heading[:, 1] - self.base_heading.squeeze(1)
        )

        self.step_commands_right[:, :3] = quat_rotate_inverse(
            self.base_quat,
            torch.cat(
                (
                    self.step_commands[:, 0, :2],
                    torch.zeros((self.num_envs, 1), device=self.device),
                ),
                dim=1,
            )
            - self.base_pos,
        )
        self.step_commands_left[:, :3] = quat_rotate_inverse(
            self.base_quat,
            torch.cat(
                (
                    self.step_commands[:, 1, :2],
                    torch.zeros((self.num_envs, 1), device=self.device),
                ),
                dim=1,
            )
            - self.base_pos,
        )
        self.step_commands_right[:, 3] = wrap_to_pi(
            self.step_commands[:, 0, 2] - self.base_heading.squeeze(1)
        )
        self.step_commands_left[:, 3] = wrap_to_pi(
            self.step_commands[:, 1, 2] - self.base_heading.squeeze(1)
        )

        self.phase_sin = torch.sin(2 * torch.pi * self.phase)
        self.phase_cos = torch.cos(2 * torch.pi * self.phase)

        self.base_lin_vel_world = self.root_states[:, 7:10].clone()

    def _log_info(self):
        # print("_log_info")
        """Log any information for debugging"""
        self.extras["dof_vel"] = self.dof_vel
        self.extras["step_commands"] = self.step_commands
        self.extras["update_count"] = self.update_count

    # ==========================visualization===========
    def _visualization(self):
        # print("_visualization")
        self.gym.clear_lines(self.viewer)
        # self._draw_heightmap_vis()
        # self._draw_debug_vis()
        self._draw_velocity_arrow_vis()
        self._draw_world_velocity_arrow_vis()
        # self._draw_base_pos_vis()
        # self._draw_CoM_vis()
        # self._draw_raibert_vis()
        # self._draw_step_vis()
        # self._draw_step_command_vis()

    def _draw_debug_vis(self):
        """Draws anything for debugging for humanoid"""
        sphere_origin = gymutil.WireframeSphereGeometry(
            0.02, 4, 4, None, color=(1, 1, 1)
        )
        origins = self.base_pos + quat_apply(
            self.base_quat,
            torch.tensor([0.0, 0.0, 0.5]).repeat(self.num_envs, 1).to(self.device),
        )

        for i in range(self.num_envs):
            env_origin = gymapi.Transform(gymapi.Vec3(*self.env_origins[i]), r=None)
            gymutil.draw_lines(
                sphere_origin, self.gym, self.viewer, self.envs[i], env_origin
            )

    def _draw_velocity_arrow_vis(self):
        """Draws linear / angular velocity arrow for humanoid
        Angular velocity is described by axis-angle representation"""
        origins = self.base_pos + quat_apply(
            self.base_quat,
            torch.tensor([0.0, 0.0, 0.5]).repeat(self.num_envs, 1).to(self.device),
        )
        lin_vel_command = quat_apply(
            self.base_quat,
            torch.cat(
                (
                    self.commands[:, :2],
                    torch.zeros((self.num_envs, 1), device=self.device),
                ),
                dim=1,
            )
            / 5,
        )
        ang_vel_command = quat_apply(
            self.base_quat,
            torch.cat(
                (
                    torch.zeros((self.num_envs, 2), device=self.device),
                    self.commands[:, 2:3],
                ),
                dim=1,
            )
            / 5,
        )
        for i in range(self.num_envs):
            lin_vel_arrow = VelCommandGeometry(
                origins[i], lin_vel_command[i], color=(0, 1, 0)
            )
            ang_vel_arrow = VelCommandGeometry(
                origins[i], ang_vel_command[i], color=(0, 1, 0)
            )
            gymutil.draw_lines(
                lin_vel_arrow, self.gym, self.viewer, self.envs[i], pose=None
            )
            gymutil.draw_lines(
                ang_vel_arrow, self.gym, self.viewer, self.envs[i], pose=None
            )

    def _draw_world_velocity_arrow_vis(self):
        """Draws linear / angular velocity arrow for humanoid
        Angular velocity is described by axis-angle representation"""
        origins = self.base_pos + quat_apply(
            self.base_quat,
            torch.tensor([0.0, 0.0, 0.5]).repeat(self.num_envs, 1).to(self.device),
        )
        lin_vel_command = (
            torch.cat(
                (
                    self.commands[:, :2],
                    torch.zeros((self.num_envs, 1), device=self.device),
                ),
                dim=1,
            )
            / 5
        )
        # ang_vel_command = quat_apply(self.base_quat, torch.cat((torch.zeros((self.num_envs,2), device=self.device), self.commands[:, 2:3]), dim=1)/5)
        for i in range(self.num_envs):
            lin_vel_arrow = VelCommandGeometry(
                origins[i], lin_vel_command[i], color=(0, 1, 0)
            )
            # ang_vel_arrow = VelCommandGeometry(origins[i], ang_vel_command[i], color=(0,1,0))
            gymutil.draw_lines(
                lin_vel_arrow, self.gym, self.viewer, self.envs[i], pose=None
            )
            # gymutil.draw_lines(ang_vel_arrow, self.gym, self.viewer, self.envs[i], pose=None)

    def _draw_base_pos_vis(self):
        sphere_base = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 1))
        sphere_left_hip = gymutil.WireframeSphereGeometry(
            0.02, 4, 4, None, color=(0, 0, 1)
        )
        sphere_right_hip = gymutil.WireframeSphereGeometry(
            0.02, 4, 4, None, color=(1, 0, 0)
        )

        base_projection = torch.cat(
            (self.base_pos[:, :2], torch.zeros((self.num_envs, 1), device=self.device)),
            dim=1,
        )
        right_hip_projection = torch.cat(
            (
                self.right_hip_pos[:, :2],
                torch.zeros((self.num_envs, 1), device=self.device),
            ),
            dim=1,
        )
        left_hip_projection = torch.cat(
            (
                self.left_hip_pos[:, :2],
                torch.zeros((self.num_envs, 1), device=self.device),
            ),
            dim=1,
        )
        for i in range(self.num_envs):
            base_loc = gymapi.Transform(gymapi.Vec3(*base_projection[i]), r=None)
            gymutil.draw_lines(
                sphere_base, self.gym, self.viewer, self.envs[i], base_loc
            )
            right_hip_loc = gymapi.Transform(
                gymapi.Vec3(*right_hip_projection[i]), r=None
            )
            gymutil.draw_lines(
                sphere_right_hip, self.gym, self.viewer, self.envs[i], right_hip_loc
            )
            left_hip_loc = gymapi.Transform(
                gymapi.Vec3(*left_hip_projection[i]), r=None
            )
            gymutil.draw_lines(
                sphere_left_hip, self.gym, self.viewer, self.envs[i], left_hip_loc
            )

    def _draw_CoM_vis(self):
        sphere_CoM = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 1))
        CoM_projection = torch.cat(
            (self.CoM[:, :2], torch.zeros((self.num_envs, 1), device=self.device)),
            dim=1,
        )
        for i in range(self.num_envs):
            CoM_loc = gymapi.Transform(gymapi.Vec3(*CoM_projection[i]), r=None)
            gymutil.draw_lines(sphere_CoM, self.gym, self.viewer, self.envs[i], CoM_loc)

    def _draw_raibert_vis(self):
        sphere_right_raibert = gymutil.WireframeSphereGeometry(
            0.02, 4, 4, None, color=(1, 0, 0)
        )
        sphere_left_raibert = gymutil.WireframeSphereGeometry(
            0.02, 4, 4, None, color=(0, 0, 1)
        )

        for i in range(self.num_envs):
            right_raibert_loc = gymapi.Transform(
                gymapi.Vec3(*self.raibert_heuristic[i, 0]), r=None
            )
            gymutil.draw_lines(
                sphere_right_raibert,
                self.gym,
                self.viewer,
                self.envs[i],
                right_raibert_loc,
            )

            left_raibert_loc = gymapi.Transform(
                gymapi.Vec3(*self.raibert_heuristic[i, 1]), r=None
            )
            gymutil.draw_lines(
                sphere_left_raibert,
                self.gym,
                self.viewer,
                self.envs[i],
                left_raibert_loc,
            )

    def _draw_step_vis(self):
        """Draws current foot steps for humanoid"""
        for i in range(self.num_envs):
            right_foot_step = FootStepGeometry(
                self.current_step[i, 0, :2], self.current_step[i, 0, 2], color=(1, 0, 1)
            )  # Right foot: Pink
            left_foot_step = FootStepGeometry(
                self.current_step[i, 1, :2], self.current_step[i, 1, 2], color=(0, 1, 1)
            )  # Left foot: Cyan
            gymutil.draw_lines(
                left_foot_step, self.gym, self.viewer, self.envs[i], pose=None
            )
            gymutil.draw_lines(
                right_foot_step, self.gym, self.viewer, self.envs[i], pose=None
            )

    def _draw_step_command_vis(self):
        """Draws step command for humanoid"""
        for i in range(self.num_envs):
            right_step_command = FootStepGeometry(
                self.step_commands[i, 0, :2],
                self.step_commands[i, 0, 2],
                color=(1, 0, 0),
            )  # Right foot: Red
            left_step_command = FootStepGeometry(
                self.step_commands[i, 1, :2],
                self.step_commands[i, 1, 2],
                color=(0, 0, 1),
            )  # Left foot: Blue
            gymutil.draw_lines(
                left_step_command, self.gym, self.viewer, self.envs[i], pose=None
            )
            gymutil.draw_lines(
                right_step_command, self.gym, self.viewer, self.envs[i], pose=None
            )

    # ==========================visualization===========

    # * ########################## REWARDS ######################## * #

    # * Floating base Rewards * #

    def _reward_base_height(self):
        """Reward tracking specified base height"""
        error = (self.cfg.rewards.base_height_target - self.base_height).flatten()
        return self._negsqrd_exp(error)

    def _reward_base_axis_xy_orientation(self):
        """Reward tracking upright orientation"""
        # ind = [1, 2]
        # aa = self.projected_gravity[:, ind]
        # error = torch.norm(aa, dim=-1)
        error = torch.square(self.projected_gravity[:, 0])
        _rew = self._negsqrd_exp(error, a=1.5)
        # print("_reward_base_axis_y_orientation: \n", aa)
        # print("_reward_base_axis_y_orientation: \n", _rew)
        # print("_reward_base_axis_y_orientation: ",_rew.size())
        # self._rew_base_axis_y_orientation = _rew
        return _rew

    def _reward_tracking_lin_vel_world(self):
        # print("_reward_tracking_lin_vel_world")
        # Reward tracking linear velocity command in world frame
        error = (
            self.commands[:, :2] - self.base_lin_vel[:, :2]
        )  # self.root_states[:, 7:9]
        error *= 1.0 / (1.0 + torch.abs(self.commands[:, :2]))
        _rew = self._negsqrd_exp(error, a=1.0).sum(dim=1)
        # print("_reward_tracking_lin_vel_world: ",_rew.size())
        return _rew

    def _reward_two_hip_same_theta(self):
        # print("_reward_two_leg_same_theta")
        error: Tensor = torch.abs(self.dof_pos[:, 0] - self.dof_pos[:, 3])
        # print(error.size())
        # print(error)
        # print(type(error))
        _rew = self._neg_exp(error, a=1)  # .sum(dim=1)
        # print(_rew.size())
        return _rew

    def _reward_two_hip_zero_theta(self):
        error_1: Tensor = torch.abs(self.dof_pos[:, 0] - 0)
        error_2: Tensor = torch.abs(self.dof_pos[:, 3] - 0)
        _rew = self._neg_exp(error_1 + error_2, a=1)  # .sum(dim=1)
        return _rew

    def _reward_two_leg_same_length(self):
        error: Tensor = torch.abs(self.dof_pos[:, 1] - self.dof_pos[:, 4])
        _rew = self._neg_exp(error, a=0.5)
        return _rew

    def _reward_two_leg_zero_length(self):
        error_1: Tensor = torch.abs(self.dof_pos[:, 1] - 0)
        error_2: Tensor = torch.abs(self.dof_pos[:, 4] - 0)
        _rew = self._neg_exp(error_1 + error_2, a=0.3)
        return _rew

    def _reward_wheel_contact_ground(self):
        _rew = (torch.sum(self.foot_contact, dim=1) - 1) * 2
        # print(_rew[0])
        # print(_rew.size())
        return _rew

    def _reward_actuation_rate(self):
        # Penalize changes in actuations
        nact = self.num_actuators
        dt2 = (self.dt * self.cfg.control.decimation) ** 2
        ind_1 = [
            0,
            1,
            #  2,
            3,
            4,
            #  5.
        ]
        ind_2 = [
            0 + nact,
            1 + nact,
            # 2 + nact,
            3 + nact,
            4 + nact,
            # 5 + nact,
        ]
        ind_3 = [
            0 + 2 * nact,
            1 + 2 * nact,
            # 2 + 2 * nact,
            3 + 2 * nact,
            4 + 2 * nact,
            # 5 + 2 * nact,
        ]
        error = (
            torch.square(
                self.actuation_history[:, ind_1] - self.actuation_history[:, ind_2]
            )
            / dt2
        )
        return -torch.sum(error, dim=1)

    def _reward_actuation_rate2(self):
        # Penalize changes in actuations
        nact = self.num_actuators
        dt2 = (self.dt * self.cfg.control.decimation) ** 2
        ind_1 = [
            0,
            1,
            #  2,
            3,
            4,
            #  5.
        ]
        ind_2 = [
            0 + nact,
            1 + nact,
            # 2 + nact,
            3 + nact,
            4 + nact,
            # 5 + nact,
        ]
        ind_3 = [
            0 + 2 * nact,
            1 + 2 * nact,
            # 2 + 2 * nact,
            3 + 2 * nact,
            4 + 2 * nact,
            # 5 + 2 * nact,
        ]
        error = (
            torch.square(
                self.actuation_history[:, ind_1]
                - 2 * self.actuation_history[:, ind_2]
                + self.actuation_history[:, ind_3]
            )
            / dt2
        )
        return -torch.sum(error, dim=1)

    def _reward_torques(self):
        # Penalize torques
        return -torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        ind_1 = [0, 1, 3, 4]
        return -torch.sum(torch.square(self.dof_vel[:, ind_1]), dim=1)

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return -torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity (x:roll, y:pitch)
        return -torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(
            max=0.0
        )  # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.0)
        return -torch.sum(out_of_limits, dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return -torch.sum(
            (
                torch.abs(self.torques)
                - self.torque_limits * self.cfg.rewards.soft_torque_limit
            ).clip(min=0.0),
            dim=1,
        )

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return -torch.sum(
            (
                torch.abs(self.dof_vel)
                - self.dof_vel_limits * self.cfg.rewards.soft_dof_vel_limit
            ).clip(min=0.0, max=1.0),
            dim=1,
        )

    def _reward_wheel_ang_action_turn_rate(self): ...

    def _reward_wheel_power(self): ...

    def _reward_wheel_vel_diff_from_base(self):
        l = self.dof_vel[:, 2] * 0.15
        r = self.dof_vel[:, 2] * 0.15
        w = 2*(l - r) / 0.52
        v = (l + r) / 2
        diff_w = torch.abs(self.base_ang_vel[:, 2] - w) / 4
        diff_v = torch.abs(self.base_lin_vel[:, 0] - v) / 2
        _rew = torch.exp(-diff_w * 10.0) + torch.exp(-diff_v * 10.0)
        return _rew
    # ##################### HELPER FUNCTIONS ################################## #

    def smooth_sqr_wave(self, phase):
        p = 2.0 * torch.pi * phase
        eps = 0.2
        return torch.sin(p) / torch.sqrt(torch.sin(p) ** 2.0 + eps**2.0)


""" Code Explanation
0.
[Axis] X-axis: Red, Y-axis: Green, Z-axis: Blue

1.
self.base_pos = self.root_states[:, 0:3] : position of the base
self.base_quat = self.root_states[:, 3:7] : quaternion of the base
self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10]) : base linear velocity wrt base frame
self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13]) : base angular velocity wrt base frame

2.                                                    
quat_rotate_inverse() : World frame -> Base frame
quat_rotate(), quat_apply() : Base frame -> World frame

3.
self.rigid_body_state : [num_envs, num_bodies, 13] = [num_envs, 21, 13] 
[position | orientation (Quat) | linear velocity | angular velocity]

self._rigid_body_pos = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[..., 0:3]
self._rigid_body_vel = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[..., 7:10]

4.
21 bodies: base / right_hip_yaw / right_hip_abad / right_upper_leg / right_lower_leg / right_foot / left_hip_yaw / left_hip_abad / left_upper_leg / left_lower_leg / left_foot
                / right_shoulder / right_shoulder_2 / right_upper_arm / right_lower_arm / right_hand / left_shoulder / left_shoulder_2 / left_upper_arm / left_lower_arm / left_hand

right_foot[5] / left_foot[10] are end-effector

5.
self.dof_pos : joint position [num_envs, 10]       
self.dof_vel : joint velocity [num_envs, 10]                     
10 dof: 01_right_hip_yaw / 02_right_hip_abad / 03_right_hip_pitch / 04_right_knee / 05_right_ankle
        06_left_hip_yaw / 07_left_hip_abad / 08_left_hip_pitch / 09_left_knee / 10_left_ankle

6.
self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec) : gravity wrt base frame

7.
self.contact_forces : contact forces of each body parts [num_envs, num_bodies, 3] = [num_envs, 21, 3]
Contact forces are only measured when the collision body is defined. 

self.foot_contact : which foot (right & left foot) are in contact with ground [num_envs, 2]

8.
self.feet_ids: right_foot[5], left_foot[10]
self.end_eff_ids: right_foot[5], left_foot[10]

9.
Maximum reward we can get is "Max value of reward function * reward weight".
Since how it records the reward is "value * weight * dt  * (max_episode_length_s / dt) / max_episode_length_s = value * weight"
"""

""" TODO: 
1) Fix foot_reference_trajectory reward. It forces not to do sprint. 
Because the trajectory always start from the previous step command. Gradually increase the rewards.
2) Systematic training curriculum is necessary
"""
