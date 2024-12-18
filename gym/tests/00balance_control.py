from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import numpy as np
import math
import os
import time

from mat_function import *
from pathlib import Path
from pid import *

# print("Current script directory:", script_dir)
# debug = True
debug = False
_dt = 1.0 / 1000.0

script_path = Path(__file__).resolve()
script_dir = (script_path.parent).parent.parent
print(script_dir)
asset_root = str(script_dir / "resources")
asset_files = "robots/hector_description/urdf/ph_0320_01_rl.urdf"
# asset_files = "urdf/hi_18/urdf/hi_1_18_240731_rl.urdf"
asset_names = "ph_0320_01_rl"
# /home/hpx/Balance/ph_rl_control/resources/robots/hector_description/urdf/ph_0320_01_rl.urdf
custom_parameters = [
    {
        "name": "--controller",
        "type": str,
        "default": "PID",
        "help": "Controller to use for Balance. Options are {PID, LQR, MPC}",
    },
    {
        "name": "--num_envs",
        "type": int,
        "default": 4,
        "help": "Number of environments to create",
    },
]

np.set_printoptions(precision=5, suppress=True, linewidth=100000, threshold=100000)
torch.set_printoptions(precision=4, sci_mode=False, linewidth=500, threshold=20000000)


class balance_controller:
    def __init__(self):
        self.gym = gymapi.acquire_gym()  # initialize gym
        self.init_buff()
        self.init()
        self.warp_tensor()

    def init_args(self):
        args = gymutil.parse_arguments(
            description="Asset and Environment Information",
            custom_parameters=custom_parameters,
        )  # parse arguments
        return args

    def init_sim_params(self):
        sim_params = gymapi.SimParams()  # create simulation context
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.up_axis = gymapi.UP_AXIS_Z
        # sim_params.dt = 1.0 / 100.0
        sim_params.dt = _dt
        sim_params.use_gpu_pipeline = False
        print("sim_params.use_gpu_pipeline:", sim_params.use_gpu_pipeline)
        if self.args.physics_engine == gymapi.SIM_PHYSX:
            sim_params.physx.solver_type = 1
            sim_params.physx.num_position_iterations = 8
            sim_params.physx.num_velocity_iterations = 1
            sim_params.physx.rest_offset = 0.0
            sim_params.physx.contact_offset = 0.001
            sim_params.physx.friction_offset_threshold = 0.001
            sim_params.physx.friction_correlation_distance = 0.0005
            sim_params.physx.num_threads = 32
            sim_params.physx.use_gpu = self.args.use_gpu
        else:
            raise Exception("This example can only be used with PhysX")
        return sim_params

    def init_controller(self):
        self.controller = self.args.controller
        assert self.controller in {
            "PID",
            "LQR",
            "MPC",
            "test",
        }, f"Invalid controller specified -- options are (PID, LQR, MPC). Got: {self.controller}"

    def init_gym(self):
        return self.gym.create_sim(
            self.args.compute_device_id,
            self.args.graphics_device_id,
            self.args.physics_engine,
            self.sim_params,
        )

    def init_viewer(self):
        return self.gym.create_viewer(self.sim, gymapi.CameraProperties())

    def init_plane(self):
        plane_params = gymapi.PlaneParams()  # add ground plane
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def init_asset(self):
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.disable_gravity = False
        # asset_options.fix_base_link = True
        print("asset_root: ", asset_root)
        print("asset_files: ", asset_files)
        return self.gym.load_asset(self.sim, asset_root, asset_files, asset_options)

    def init_Forse_sensor_pp(self):
        sensor_options = gymapi.ForceSensorProperties()
        sensor_options.enable_forward_dynamics_forces = False  # for example gravity
        sensor_options.enable_constraint_solver_forces = True  # for example contacts
        sensor_options.use_world_frame = (
            True  # report forces in world frame (easier to get vertical components)
        )
        return sensor_options

    def init_dof_props(self):
        print("self.current_asset: ", self.current_asset)
        dof_props = self.gym.get_asset_dof_properties(self.current_asset)
        dof_props["driveMode"][:].fill(gymapi.DOF_MODE_EFFORT)
        dof_props["stiffness"][:].fill(0.0)
        dof_props["damping"][:].fill(0.0)
        return dof_props

    def init_envs(self):
        self.num_envs = self.args.num_envs
        num_per_row = int(math.sqrt(self.num_envs))
        spacing = 2.0
        env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        default_dof_state = np.ndarray((6,), dtype=gymapi.DofState.dtype)
        self.default_dof_pos = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
        default_dof_state["pos"] = self.default_dof_pos
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, 0.245)

        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.envs.append(env)
            sensor_pose = gymapi.Transform(gymapi.Vec3(0.0, 0.0, 0.0))
            pai_handle = self.gym.create_actor(
                env, self.current_asset, pose, asset_names, -1, -1
            )
            num_sensors = self.gym.get_actor_force_sensor_count(env, pai_handle)
            self.pai_handles.append(pai_handle)
            print("self.dof_props: ", self.dof_props)
            self.gym.set_actor_dof_properties(env, pai_handle, self.dof_props)
            self.gym.set_actor_dof_states(
                env,
                pai_handle,
                default_dof_state,
                gymapi.STATE_POS,
            )
            self.gym.set_actor_dof_states(
                env,
                pai_handle,
                default_dof_state,
                gymapi.STATE_POS,
            )
            feet_l_idx = self.gym.find_actor_rigid_body_index(
                env, pai_handle, "l_wheel_link", gymapi.DOMAIN_SIM
            )
            self.link_indexs["feet_l"].append(feet_l_idx)
            feet_r_idx = self.gym.find_actor_rigid_body_index(
                env, pai_handle, "r_wheel_link", gymapi.DOMAIN_SIM
            )
            self.link_indexs["feet_r"].append(feet_r_idx)
            base_idx = self.gym.find_actor_rigid_body_index(
                env, pai_handle, "base_link", gymapi.DOMAIN_SIM
            )
            # print(feet_r_idx)
            self.link_indexs["base"].append(base_idx)
            # sensor_idx = self.gym.create_asset_force_sensor(self.current_asset, feet_l_idx, sensor_pose, self.sensor_pp)

        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_R, "reset")
        self.gym.viewer_camera_look_at(
            self.viewer, None, gymapi.Vec3(2, 0, 0.5), gymapi.Vec3(-1, 0, 0)
        )
        self.initial_state = np.copy(
            self.gym.get_sim_rigid_body_states(self.sim, gymapi.STATE_ALL)
        )

    def init(self):
        self.args = self.init_args()
        self.sim_params = self.init_sim_params()
        self.init_controller()
        self.sim = self.init_gym()
        if self.sim is None:
            raise Exception("Failed to create sim")
        self.viewer = self.init_viewer()
        if self.viewer is None:
            raise Exception("Failed to create viewer")
        self.init_plane()
        self.current_asset = self.init_asset()
        self.sensor_pp = self.init_Forse_sensor_pp()
        if self.current_asset is None:
            raise Exception("Failed to load asset")
        self.dof_props = self.init_dof_props()
        self.init_envs()
        self.last_viewer_update_time = self.gym.get_sim_time(self.sim)
        self.init_pid()

    def init_buff(self):
        self.envs = []
        self.pai_handles = []
        self.link_indexs = {"feet_l": [], "feet_r": [], "base": []}
        self.cont = 0
        self.dir = -1
        self.viewer_refresh_rate = 1.0 / 30.0
        self.dof_num = 6

    def warp_tensor(self):
        self.num_bodies = self.gym.get_asset_rigid_body_count(self.current_asset)
        self._rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self._rb_states = gymtorch.wrap_tensor(self._rb_states)
        self.rb_states = self._rb_states.view(self.num_envs, self.num_bodies, 13)
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(actor_root_state)

        _dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_states = gymtorch.wrap_tensor(_dof_states)

        self.dof_pos = (
            self.dof_states[:, 0].view(self.num_envs, self.dof_num, 1).squeeze(-1)
        )
        self.dof_vel = (
            self.dof_states[:, 1].view(self.num_envs, self.dof_num, 1).squeeze(-1)
        )

        self.link_dict = self.gym.get_asset_rigid_body_dict(self.current_asset)
        for key, value in self.link_dict.items():
            print(f"{key}: {value}")

        self.dof_dict = self.gym.get_asset_dof_dict(self.current_asset)
        for key, value in self.dof_dict.items():
            print(f"{key}: {value}")
        for i in range(self.num_envs):
            env = self.envs[i]
            actor_handle = self.pai_handles[i]
            self.actor_joint_dict = self.gym.get_actor_joint_dict(env, actor_handle)
            print(self.actor_joint_dict)
        body_names = self.gym.get_asset_rigid_body_names(self.current_asset)
        body_index = [self.link_dict[s] for s in body_names if "base" in s]
        print("body_index: ", body_index)
        self.feet_index = [self.link_dict[s] for s in body_names if "wheel_link" in s]
        print("feet_index: ", self.feet_index)

        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, asset_names)
        self.jacobian = gymtorch.wrap_tensor(_jacobian)

        self.pos_action = torch.zeros_like(self.dof_pos).squeeze(-1)
        self.effort_action = torch.zeros_like(self.pos_action)

    def evt_handle(self):
        for evt in self.gym.query_viewer_action_events(self.viewer):
            if evt.action == "reset" and evt.value > 0:
                self.gym.set_sim_rigid_body_states(
                    self.sim, self.initial_state, gymapi.STATE_ALL
                )
                self.cont = 0
                self.dir = -1

    def step_physics(self):
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

    def refresh_tensor(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        self.base_pos = self.root_states[:, 0:3]
        self.base_quat = self.root_states[:, 3:7]
        self.base_ang_vel = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 10:13]
        )
        self.base_lin_vel = self.root_states[:, 7:10]
        self.base_lin_vel_world = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 7:10]
        )
        # print("self.base_quat.size(): ",self.base_quat.size())
        self.base_euler_xyz = self.get_euler_xyz_tensor(self.base_quat)
        # print("self.base_euler_xyz.size(): ",self.base_euler_xyz.size())

    def get_euler_xyz_tensor(self, quat):
        r, p, w = get_euler_xyz(quat)
        # stack r, p, w in dim1
        euler_xyz = torch.stack((r, p, w), dim=1)
        euler_xyz[euler_xyz > np.pi] -= 2 * np.pi
        return euler_xyz

    def update_viewer(self):
        current_time = self.gym.get_sim_time(self.sim)
        # if current_time - self.last_viewer_update_time >= self.viewer_refresh_rate:
        #     self.gym.step_graphics(self.sim)
        #     self.gym.draw_viewer(self.viewer, self.sim, False)
        #     self.last_viewer_update_time = current_time
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, False)
        self.gym.sync_frame_time(self.sim)

    def destroy(self):
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

    def query_viewer_has_closed(self):
        return self.gym.query_viewer_has_closed(self.viewer)

    def init_pid(self):

        hip_indices = [0, 3]
        hip_fixed_length = 0 * torch.ones(
            self.num_envs, 2, dtype=torch.float, requires_grad=False
        )
        hip_kp = 30.0 * torch.ones(
            self.num_envs, 2, dtype=torch.float, requires_grad=False
        )
        hip_ki = 0.0 * torch.ones(
            self.num_envs, 2, dtype=torch.float, requires_grad=False
        )
        hip_kd = 5.0 * torch.ones(
            self.num_envs, 2, dtype=torch.float, requires_grad=False
        )
        self.hip_pid = PIDController(hip_kp, hip_ki, hip_kd, _dt)

        leg_indices = [1, 4]
        leg_fixed_length = 0 * torch.ones(
            self.num_envs, 2, dtype=torch.float, requires_grad=False
        )
        leg_kp = 4000.0 * torch.ones(
            self.num_envs, 2, dtype=torch.float, requires_grad=False
        )
        leg_kd = 80.0 * torch.ones(
            self.num_envs, 2, dtype=torch.float, requires_grad=False
        )
        leg_ki = 0.0 * torch.ones(
            self.num_envs, 2, dtype=torch.float, requires_grad=False
        )
        self.leg_pid = PIDController(leg_kp, leg_ki, leg_kd, _dt)

        wheel_pitch_kp = 40.0 * torch.ones(
            self.num_envs, 2, dtype=torch.float, requires_grad=False
        )
        wheel_pitch_kd = 0.1 * torch.ones(
            self.num_envs, 2, dtype=torch.float, requires_grad=False
        )
<<<<<<< HEAD
        wheel_pitch_ki = 0.001 * torch.ones(
=======
        wheel_pitch_ki = 0.01 * torch.ones(
>>>>>>> 73e702472e617e2639264292e19d7a9b39c65f60
            self.num_envs, 2, dtype=torch.float, requires_grad=False
        )
        self.wheel_pitch_pid = PIDController(
            wheel_pitch_kp, wheel_pitch_ki, wheel_pitch_kd, _dt
        )

        wheel_vel_kp = 0.2 * torch.ones(
            self.num_envs, 2, dtype=torch.float, requires_grad=False
        )
        wheel_vel_kd = 0.001 * torch.ones(
            self.num_envs, 2, dtype=torch.float, requires_grad=False
        )
        wheel_vel_ki = 0.0 * torch.ones(
            self.num_envs, 2, dtype=torch.float, requires_grad=False
        )
        self.wheel_vel_pid = PIDController(
            wheel_vel_kp, wheel_vel_kd, wheel_vel_ki, _dt
        )

    def PID_controller(self):
        action = np.zeros((4, 6), dtype=np.float32)
        command = np.array([0.0, 0.0, 0.1], dtype=np.float32)
        command = torch.from_numpy(np.tile(command, (self.num_envs, 1)))
        # print(command)
        pitch_angle = self.base_euler_xyz[:, 1].unsqueeze(-1)
        pitch_angle_vel = self.base_ang_vel[:, 1]
        # hip
        self.hip_indices = [0, 3]
        hip_fixed_length = 0 * torch.ones(
            self.num_envs, 2, dtype=torch.float, requires_grad=False
        )
        hip_kp = 40.0 * torch.ones(
            self.num_envs, 2, dtype=torch.float, requires_grad=False
        )
        hip_kd = 1.0 * torch.ones(
            self.num_envs, 2, dtype=torch.float, requires_grad=False
        )
        hip_now_pos = self.dof_pos[:, self.hip_indices]
        hip_target_pos = pitch_angle * torch.ones_like(hip_now_pos)
        hip_now_vel = self.dof_vel[:, self.hip_indices]
        hip_target_vel = 0 * torch.ones_like(hip_now_vel)
        hip_out = self.hip_pid.compute(hip_target_pos, hip_now_pos)
        action[:, self.hip_indices] = hip_out
        # action[:, hip_indices] = hip_kp * (hip_target_pos - hip_now_pos) + hip_kd * (
        #     hip_target_vel - hip_now_vel
        # )

        # leg
        self.leg_indices = [1, 4]
        leg_fixed_length = 0 * torch.ones(
            self.num_envs, 2, dtype=torch.float, requires_grad=False
        )
        leg_now_pos = self.dof_pos[:, self.leg_indices]
        leg_target_pos = leg_fixed_length * torch.ones_like(leg_now_pos)
        leg_out = self.leg_pid.compute(leg_target_pos, leg_now_pos)
        action[:, self.leg_indices] = leg_out

        # wheel
        # print(self.base_ang_vel[0,1],self.base_euler_xyz[0,1],)
        self.wheel_indices = [2, 5]

        # 角度环
        # print(self.base_euler_xyz[:, 1].size())
        
        wheel_pitch_out = self.wheel_pitch_pid.compute(pitch_angle, 0)
        # 速度环
        vx = command[:, 0]
        wz = command[:, 2]
        wheel_vel_target = torch.zeros_like(wheel_pitch_out)
        # print(wheel_pitch_out.size())
        # print(pitch_angle.size())
        # print(wheel_vel_target.size())
        wheel_vel_target[:, 0] = (vx - wz * 0.26) / 0.15
        wheel_vel_target[:, 1] = (vx + wz * 0.26) / 0.15
        wheel_vel_out = self.wheel_vel_pid.compute(
            wheel_vel_target, self.base_lin_vel_world[:, [0, 2]]
        )
        print(wheel_vel_out)
        # print(self.base_lin_vel_world[0, [0, 2]])
        action[:, self.wheel_indices] = wheel_pitch_out#+wheel_vel_out

        action_tensor = torch.from_numpy(action)
        # print(action_tensor)
        rt_flag = self.gym.set_dof_actuation_force_tensor(
            self.sim, gymtorch.unwrap_tensor(action_tensor)
        )
        # print(rt_flag)
        ...

    def cont_turn_dir(self):
        self.cont += 1
        if self.cont > int(2 / self.sim_params.dt):
            self.cont = 0
            self.dir *= -1
            print("turn direction: ", self.dir)

    def matrix_to_quat(self, matrix):
        m = matrix[:3, :3]
        tr = m[0, 0] + m[1, 1] + m[2, 2]
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            w = 0.25 * S
            x = (m[2, 1] - m[1, 2]) / S
            y = (m[0, 2] - m[2, 0]) / S
            z = (m[1, 0] - m[0, 1]) / S
        elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
            S = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            w = (m[2, 1] - m[1, 2]) / S
            x = 0.25 * S
            y = (m[0, 1] + m[1, 0]) / S
            z = (m[0, 2] + m[2, 0]) / S
        elif m[1, 1] > m[2, 2]:
            S = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            w = (m[0, 2] - m[2, 0]) / S
            x = (m[0, 1] + m[1, 0]) / S
            y = 0.25 * S
            z = (m[1, 2] + m[2, 1]) / S
        else:
            S = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            w = (m[1, 0] - m[0, 1]) / S
            x = (m[0, 2] + m[2, 0]) / S
            y = (m[1, 2] + m[2, 1]) / S
            z = 0.25 * S
        return np.array([x, y, z, w])


print("Working directory: %s" % os.getcwd())

a = balance_controller()
# a.tf_isaac_init()
# while 0:
viewer_refresh_rate = 1.0 / 30.0
cnt = 0
while not a.query_viewer_has_closed():
    a.evt_handle()
    a.step_physics()

    a.refresh_tensor()
    a.PID_controller()
    if debug:
        time.sleep(0.001)
    # cnt+=1
    # if cnt>10:
    # break
    a.cont_turn_dir()
    a.update_viewer()

a.destroy()
