from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import math
import numpy as np
import torch
import random
import time

from pathlib import Path

def quat_axis(q, axis=0):
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)


def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


# set random seed
np.random.seed(42)

torch.set_printoptions(precision=4, sci_mode=False)

# acquire gym interface
gym = gymapi.acquire_gym()

# parse arguments

# Add custom arguments
custom_parameters = [
    {
        "name": "--controller",
        "type": str,
        "default": "ik",
        "help": "Controller to use for Franka. Options are {ik, osc}",
    },
    {
        "name": "--num_envs",
        "type": int,
        "default": 4,
        "help": "Number of environments to create",
    },
]
args = gymutil.parse_arguments(
    description="Franka Jacobian Inverse Kinematics (IK) + Operational Space Control (OSC) Example",
    custom_parameters=custom_parameters,
)

# Grab controller
controller = args.controller
assert controller in {
    "ik",
    "osc",
}, f"Invalid controller specified -- options are (ik, osc). Got: {controller}"

# set torch device
device = args.sim_device if args.use_gpu_pipeline else "cpu"

# configure sim
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
sim_params.dt = 1.0 / 1000.0
sim_params.substeps = 2
sim_params.use_gpu_pipeline = args.use_gpu_pipeline
print("sim params.use_gpu_pipeline: %s" % sim_params.use_gpu_pipeline)
print("args.use_gpu: %s" % args.use_gpu)
if args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 8
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.rest_offset = 0.0
    sim_params.physx.contact_offset = 0.001
    sim_params.physx.friction_offset_threshold = 0.001
    sim_params.physx.friction_correlation_distance = 0.0005
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
else:
    raise Exception("This example can only be used with PhysX")

# create sim
sim = gym.create_sim(
    args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params
)
if sim is None:
    raise Exception("Failed to create sim")

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise Exception("Failed to create viewer")

script_path = Path(__file__).resolve()
script_dir = (script_path.parent).parent.parent
asset_root = str(script_dir / "resources")
asset_files = "robots/hector_description/urdf/ph_0320_01_rl.urdf"
asset_names = "ph_0320_01_rl"

# load balance asset
asset_options = gymapi.AssetOptions()
asset_options.armature = 0.01
asset_options.fix_base_link = False
asset_options.disable_gravity = False
asset_options.flip_visual_attachments = False
balance_asset = gym.load_asset(sim, asset_root, asset_files, asset_options)

# configure franka dofs
balance_dof_props = gym.get_asset_dof_properties(balance_asset)
balance_lower_limits = balance_dof_props["lower"]
balance_upper_limits = balance_dof_props["upper"]
balance_ranges = balance_upper_limits - balance_lower_limits
balance_mids = 0.3 * (balance_upper_limits + balance_lower_limits)

# use position drive for all dofs
if controller == "ik":
    balance_dof_props["driveMode"][:].fill(gymapi.DOF_MODE_POS)
    balance_dof_props["stiffness"][:].fill(400.0)
    balance_dof_props["damping"][:].fill(40.0)
else:  # osc
    balance_dof_props["driveMode"][:].fill(gymapi.DOF_MODE_EFFORT)
    balance_dof_props["stiffness"][:].fill(0.0)
    balance_dof_props["damping"][:].fill(0.0)

# default dof states and position targets
balance_num_dofs = gym.get_asset_dof_count(balance_asset)
default_dof_pos = np.zeros(balance_num_dofs, dtype=np.float32)
default_dof_pos[:] = balance_mids[:]

default_dof_state = np.zeros(balance_num_dofs, gymapi.DofState.dtype)
default_dof_state["pos"] = default_dof_pos

# send to torch
default_dof_pos_tensor = to_torch(default_dof_pos, device=device)

# configure env grid
num_envs = args.num_envs
num_per_row = int(math.sqrt(num_envs))
spacing = 1.0
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)
print("Creating %d environments" % num_envs)

balance_pose = gymapi.Transform()
balance_pose.p = gymapi.Vec3(0, 0, 0.245)

envs = []
box_idxs = []
hand_idxs = []
init_pos_list = []
init_rot_list = []

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # add franka
    balance_handle = gym.create_actor(env, balance_asset, balance_pose, asset_names, i, 2)

    # set dof properties
    gym.set_actor_dof_properties(env, balance_handle, balance_dof_props)

    # set initial dof states
    gym.set_actor_dof_states(env, balance_handle, default_dof_state, gymapi.STATE_ALL)

    # set initial position targets
    gym.set_actor_dof_position_targets(env, balance_handle, default_dof_pos)

# point camera at middle env
cam_pos = gymapi.Vec3(2, 0, 0.5)
cam_target = gymapi.Vec3(-1, 0, 0)
middle_env = envs[num_envs // 2 + num_per_row // 2]
gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)

# ==== prepare tensors =====
# from now on, we will use the tensor API that can run on CPU or GPU
gym.prepare_sim(sim)

# initial hand position and orientation tensors
# init_pos = torch.Tensor(init_pos_list).view(num_envs, 3).to(device)
# init_rot = torch.Tensor(init_rot_list).view(num_envs, 4).to(device)

# hand orientation for grasping
down_q = (
    torch.stack(num_envs * [torch.tensor([1.0, 0.0, 0.0, 0.0])])
    .to(device)
    .view((num_envs, 4))
)


# downard axis
down_dir = torch.Tensor([0, 0, -1]).to(device).view(1, 3)

# get jacobian tensor
# for fixed-base franka, tensor has shape (num envs, 10, 6, 9)
_jacobian = gym.acquire_jacobian_tensor(sim, asset_names)
jacobian = gymtorch.wrap_tensor(_jacobian)

# get mass matrix tensor
_massmatrix = gym.acquire_mass_matrix_tensor(sim, asset_names)
mm = gymtorch.wrap_tensor(_massmatrix)

# get rigid body state tensor
_rb_states = gym.acquire_rigid_body_state_tensor(sim)
rb_states = gymtorch.wrap_tensor(_rb_states)

# get dof state tensor
dof_num = 6
_dof_states = gym.acquire_dof_state_tensor(sim)
dof_states = gymtorch.wrap_tensor(_dof_states)
dof_pos = dof_states[:, 0].view(num_envs, dof_num, 1).squeeze(-1)
dof_vel = dof_states[:, 1].view(num_envs, dof_num, 1).squeeze(-1)

# Set action tensors
pos_action = torch.zeros_like(dof_pos).squeeze(-1)
effort_action = torch.zeros_like(pos_action)

# simulation loop
while not gym.query_viewer_has_closed(viewer):

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # refresh tensors
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_jacobian_tensors(sim)
    gym.refresh_mass_matrix_tensors(sim)



    print(jacobian.device)



    gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_action))
    gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(effort_action))

    # update viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)
    gym.sync_frame_time(sim)

# cleanup
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
