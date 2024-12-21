"""
Configuration file for "fixed arm" (FA) Pai02 environment
with potential-based rewards implemented
"""

import torch
from gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotRunnerCfg


class Ph_1_ControllerCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096 * 2
        num_actuators = 6
        episode_length_s = 6  # 100

    class terrain(LeggedRobotCfg.terrain):
        curriculum = False
        mesh_type = "plane"  # 'plane' 'heightfield' 'trimesh'
        measure_heights = False  # True, False
        measured_points_x_range = [-0.8, 0.8]
        measured_points_x_num_sample = 33
        measured_points_y_range = [-0.8, 0.8]
        measured_points_y_num_sample = 33
        selected = True  # True, False
        terrain_kwargs = {"type": "stepping_stones"}
        # terrain_kwargs = {'type': 'random_uniform'}
        # terrain_kwargs = {'type': 'gap'}
        # difficulty = 0.35 # For gap terrain
        # platform_size = 5.5 # For gap terrain
        difficulty = 5.0  # For rough terrain
        terrain_length = 18.0  # For rough terrain
        terrain_width = 18.0  # For rough terrain
        # terrain types: [pyramid_sloped, random_uniform, stairs down, stairs up, discrete obstacles, stepping_stones, gap, pit]
        terrain_proportions = [0.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0]

    class init_state(LeggedRobotCfg.init_state):
        reset_mode = "reset_to_basic"  # 'reset_to_basic'
        # reset_mode = 'reset_to_range' # 'reset_to_basic'
        pos = [0., 0., 0.245]        # x,y,z [m]
        # pos = [0.0, 0.0, 0.245]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]

        # ranges for [x, y, z, roll, pitch, yaw]
        root_pos_range = [
            [0.0, 0.0],  # x
            [0.0, 0.0],  # y
            [0.245, 0.266],  # z
            [-torch.pi / 12, torch.pi / 12],  # roll
            [-torch.pi / 12, torch.pi / 12],  # pitch
            [-1e-5, 1e-5],  # yaw
            # [-torch.pi/10, torch.pi/10],  # roll
            # [-torch.pi/10, torch.pi/10],  # pitch
            # [-torch.pi/10, torch.pi/10]   # yaw
        ]

        # ranges for [v_x, v_y, v_z, w_x, w_y, w_z]
        root_vel_range = [
            [-0.1, 0.1],  # x
            [-1e-8, 1e-8],  # y
            [-1e-8, 1e-8],  # z
            [-0.02, 0.02],  # roll
            [-0.02, 0.02],  # pitch
            [-0.02, 0.02],  # yaw
        ]

        default_joint_angles = {
            "l_mt_joint": 0.0,
            "l_p_joint": 0.0,
            "l_wheel_joint": 0.0,
            "r_mt_joint": 0.0,
            "r_p_joint": 0.0,
            "r_wheel_joint": 0.0,
        }

        dof_pos_range = {
            "l_mt_joint": [-0.7, 0.7],
            "l_p_joint": [-0.18, 0.08],
            "l_wheel_joint": [-1e14, 1e14],
            "r_mt_joint": [-0.7, 0.7],
            "r_p_joint": [-0.18, 0.08],
            "r_wheel_joint": [-1e14, 1e14],
        }

        dof_vel_range = {
            "l_mt_joint": [-0.05, 0.05],
            "l_p_joint": [-0.01, 0.01],
            "l_wheel_joint": [-0.05, 0.05],
            "r_mt_joint": [-0.05, 0.05],
            "r_p_joint": [-0.01, 0.01],
            "r_wheel_joint": [-0.05, 0.05],
        }

        dof_ctr_type = {
            "l_mt_joint": "pos",
            "l_p_joint": "pos",
            "l_wheel_joint": "vel",
            "r_mt_joint": "pos",
            "r_p_joint": "pos",
            "r_wheel_joint": "vel",
        }

    class control(LeggedRobotCfg.control):
        # stiffness and damping for joints
        stiffness = {
            "l_mt_joint": 20.0,
            "l_p_joint": 4000.0,
            "l_wheel_joint": 0.0,
            "r_mt_joint": 20.0,
            "r_p_joint": 4000.0,
            "r_wheel_joint": 0.0,
        }
        damping = {
            "l_mt_joint": 1.0,
            "l_p_joint": 80.0,
            "l_wheel_joint": 1.0,
            "r_mt_joint": 1.0,
            "r_p_joint": 80.0,
            "r_wheel_joint": 1.0,
        }

        actuation_scale = 1.0
        exp_avg_decay = None
        decimation = 10

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 1.0
        num_commands = 3
        resampling_time = 10.0  # 5.

        succeed_step_radius = 0.03
        succeed_step_angle = 10
        apex_height_percentage = 0.15

        sample_angle_offset = 20
        sample_radius_offset = 0.05

        dstep_length = 0.15
        dstep_width = 0.1

        class ranges(LeggedRobotCfg.commands.ranges):
            # TRAINING STEP COMMAND RANGES #
            name = "ph_command_range"
            sample_period = [10, 11]  # [20, 21] # equal to gait frequency
            dstep_width = [0.1, 0.1]  # [0.2, 0.4] # min max [m]

            lin_vel_x = [-1.0, 1.0]  # [-3.0, 3.0] # min max [m/s]
            lin_vel_y = 0.0  # 1.5   # min max [m/s]
            yaw_vel = 2.0  # min max [rad/s]
            small_commands_to_zero = 0.1

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True  # True, False
        friction_range = [0.5, 1.25]

        randomize_base_mass = True  # True, False
        added_mass_range = [-1.0, 1.0]

        push_robots = True
        push_interval_s = 2.5
        max_push_vel_xy = 2.0

        # Add DR for rotor inertia and angular damping

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/hector_description/urdf/ph_0320_01_rl.urdf"
        keypoints = ["base"]
        end_effectors = ["r_wheel", "l_wheel"]
        foot_name = "wheel"
        terminate_after_contacts_on = [
            "base_link",
            "r_mt_link",
            "r_p_link",
            # 'r_wheel_link',
            "l_mt_link",
            "l_p_link",
            # 'l_wheel_link',
        ]

        disable_gravity = False
        disable_actuations = False
        disable_motors = False

        # (1: disable, 0: enable...bitwise filter)
        self_collisions = 1
        flip_visual_attachments = False
        collapse_fixed_joints = True  # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        replace_cylinder_with_capsule = True  # replace collision cylinders with capsules, leads to faster/more stable simulation
        fix_base_link = False  # fixe the base of the robot

        # Check GymDofDriveModeFlags
        # (0: none, 1: pos tgt, 2: vel target, 3: effort)
        default_dof_drive_mode = 3

        angular_damping = 0.1
        rotor_inertia = [
            0.0001188,  # RIGHT LEG
            0.0001188,
            0.0001188,
            0.0001188,  # LEFT LEG
            0.0001188,
            0.0001188,
        ]
        apply_humanoid_jacobian = False  # True, False

    class rewards(LeggedRobotCfg.rewards):
        base_height_target = 0.245
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.8
        max_contact_force = 600.0

        curriculum = False
        only_positive_rewards = False
        tracking_sigma = 0.25

        class weights(LeggedRobotCfg.rewards.weights):
            # * Regularization rewards * #
            actuation_rate = 1e-2
            actuation_rate2 = 1e-3
            torques = 1e-4
            dof_vel = 1e-3
            lin_vel_z = 1e-1
            ang_vel_xy = 1e-1
            dof_pos_limits = 10
            torque_limits = 1e-1
            dof_vel_limits = 1e-1
            # * Floating base rewards * #
            base_height = 1.0
            base_axis_xy_orientation = 8.0
            tracking_lin_vel_world = 4.0

            # * shape rewards * #
            two_hip_same_theta = 2.6
            two_hip_zero_theta = 0.1
            two_leg_same_length = 0.6
            two_leg_zero_length = 2.6

            wheel_contact_ground = 1
            # 轮电机的功率限制

        class termination_weights(LeggedRobotCfg.rewards.termination_weights):
            termination = 1.0

    class scaling(LeggedRobotCfg.scaling):
        base_height = 1.0
        base_lin_vel = 1.0  # .5
        base_ang_vel = 1.0  # 2.
        projected_gravity = 1.0
        foot_states_right = 1.0
        foot_states_left = 1.0
        dof_pos = 1.0
        dof_vel = 1.0  # .1
        dof_pos_target = dof_pos  # scale by range of motion

        # Action scales
        commands = 1.0
        clip_actions = 10.0


class Ph_1_ControllerRunnerCfg(LeggedRobotRunnerCfg):
    do_wandb = True
    seed = -1

    class policy(LeggedRobotRunnerCfg.policy):
        init_noise_std = 1.0
        actor_hidden_dims = [128, 64, 64]
        critic_hidden_dims = [128, 64, 64]
        # (elu, relu, selu, crelu, lrelu, tanh, sigmoid)
        activation = "elu"
        normalize_obs = True  # True, False
        actor_obs_history_len = 50  # 10+1 = 11
        actor_obs = [
            "commands",
            "base_ang_vel",
            "projected_gravity",
            "phase_sin",
            "phase_cos",
            "hip_pos",
            "hip_vel",
            "leg_length",
            "leg_vel",
            "wheel_vel",
        ]
        critic_obs_history_len = actor_obs_history_len
        critic_obs = [
            "commands",
            "base_ang_vel",
            "projected_gravity",
            "base_height",
            "base_lin_vel_world",
            "base_lin_vel",
            "phase_sin",
            "phase_cos",
            "hip_pos",
            "hip_vel",
            "leg_length",
            "leg_vel",
            "wheel_vel",
        ]

        actions = ["dof_pos_target"]

        class noise:
            base_height = 0.05
            base_lin_vel = 0.05
            base_lin_vel_world = 0.05
            base_heading = 0.01
            base_ang_vel = 0.05
            projected_gravity = 0.05
            foot_states_right = 0.01
            foot_states_left = 0.01
            step_commands_right = 0.05
            step_commands_left = 0.05
            commands = 0.1
            dof_pos = 0.05
            dof_vel = 0.5
            foot_contact = 0.1
            hip_pos = 0.05
            hip_vel = 0.5
            leg_length = 0.05
            leg_vel = 0.5
            wheel_vel = 0.5

    class algorithm(LeggedRobotRunnerCfg.algorithm):
        class PPO:
            # algorithm training hyperparameters
            value_loss_coef = 1.0
            use_clipped_value_loss = True
            clip_param = 0.2
            entropy_coef = 1.0e-3
            num_learning_epochs = 2
            num_mini_batches = 4  # minibatch size = num_envs*nsteps/nminibatches
            learning_rate = 1.0e-5
            schedule = "adaptive"  # could be adaptive, fixed
            gamma = 0.995
            lam = 0.95
            desired_kl = 0.01
            max_grad_norm = 1.0

    class runner(LeggedRobotRunnerCfg.runner):
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 20
        max_iterations = 5000
        run_name = "sf"
        experiment_name = "ph_Controller"
        save_interval = 50
        plot_input_gradients = False
        plot_parameter_gradients = False
