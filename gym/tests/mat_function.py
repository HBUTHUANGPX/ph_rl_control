from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *



def control_ik(dpose, damping, j_eef, num_envs):
    # print("==============control_ik=================")
    # solve damped least squares
    j_eef_T = torch.transpose(j_eef, 1, 2)
    # print(j_eef_T)
    lmbda = torch.eye(6, device=dpose.device) * (damping**2)
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, 6)
    # u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose)
    # j_eef @ j_eef_T 计算雅可比矩阵与其转置的乘积，形状为 (num_envs, 6, 6)。
    # j_eef @ j_eef_T + lmbda 在雅可比矩阵乘积上加上阻尼矩阵，形状为 (num_envs, 6, 6)。
    # torch.inverse(j_eef @ j_eef_T + lmbda) 计算上述矩阵的逆矩阵，形状为 (num_envs, 6, 6)。
    # j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) 计算雅可比矩阵转置与逆矩阵的乘积，形状为 (num_envs, 7, 6)。
    # j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose 计算上述结果与 dpose 的乘积，得到关节角度变化 u，形状为 (num_envs, 7, 1)。
    # u.view(num_envs, 7) 将结果调整为形状 (num_envs, 7)，表示每个环境中的关节角度变化。
    return u


# OSC params
kp = 150.0
kd = 2.0 * np.sqrt(kp)
kp_null = 10.0
kd_null = 2.0 * np.sqrt(kp_null)


def control_osc(dpose, default_dof_pos_tensor, mm, j_eef, dof_pos, dof_vel,end_vel):
    mm_inv = torch.inverse(mm)  # 计算质量矩阵的逆
    m_eef_inv = (
        j_eef @ mm_inv @ torch.transpose(j_eef, 1, 2)
    )  # 计算末端执行器的质量矩阵
    m_eef = torch.inverse(m_eef_inv)

    # end_vel = torch.matmul(j_eef, dof_vel).squeeze(-1)
    # print(feet_vel[0])
    u = (  # 计算控制输入 u
        torch.transpose(j_eef, 1, 2)
        @ m_eef
        @ (kp * dpose - kd * end_vel.unsqueeze(-1))
    )

    # Nullspace control torques `u_null` 防止关节配置发生大的变化
    # 它们被添加到 OSC 的 nullspace 中，以保持末端执行器的方向不变
    # roboticsproceedings.org/rss07/p31.pdf
    j_eef_inv = m_eef @ j_eef @ mm_inv
    u_null = kd_null * -dof_vel + kp_null * (
        (default_dof_pos_tensor.view(1, -1, 1) - dof_pos + np.pi) % (2 * np.pi) - np.pi
    )
    # u_null = u_null[:, :7]
    u_null = mm @ u_null
    u += (  # 将 Nullspace 控制力矩添加到控制输入 u 中
        torch.eye(6, device="cpu").unsqueeze(0)
        - torch.transpose(j_eef, 1, 2) @ j_eef_inv
    ) @ u_null
    return u.squeeze(-1)

def quaternion_to_rotation_matrix(q):
    q = q / torch.norm(q, dim=1, keepdim=True)
    qw, qx, qy, qz = q[:, 3], q[:, 0], q[:, 1], q[:, 2]
    R = torch.stack(
        [
            1 - 2 * qy**2 - 2 * qz**2,
            2 * qx * qy - 2 * qz * qw,
            2 * qx * qz + 2 * qy * qw,
            2 * qx * qy + 2 * qz * qw,
            1 - 2 * qx**2 - 2 * qz**2,
            2 * qy * qz - 2 * qx * qw,
            2 * qx * qz - 2 * qy * qw,
            2 * qy * qz + 2 * qx * qw,
            1 - 2 * qx**2 - 2 * qy**2,
        ],
        dim=-1,
    ).reshape(-1, 3, 3)
    return R

def create_transformation_matrix(rotation_matrix, position):
    n = rotation_matrix.shape[0]
    transformation_matrix = torch.eye(4).repeat(n, 1, 1)
    transformation_matrix[:, :3, :3] = rotation_matrix
    transformation_matrix[:, :3, 3] = position
    return transformation_matrix

def compute_dpose(goal_transform, current_transform):
    """
    计算从当前变换到目标变换的位姿误差。
    goal_transform: 形状为 (n, 4, 4) 的目标变换矩阵张量
    current_transform: 形状为 (n, 4, 4) 的当前变换矩阵张量
    返回: 形状为 (n, 6, 1) 的位姿误差张量
    """
    # 计算相对变换矩阵
    relative_transform = torch.inverse(current_transform) @ goal_transform

    # 提取位置误差
    pos_err = relative_transform[:, :3, 3]

    # 提取旋转误差
    rotation_matrix_err = relative_transform[:, :3, :3]
    orn_err = torch.zeros(
        (relative_transform.shape[0], 3), device=relative_transform.device
    )
    orn_err[:, 0] = rotation_matrix_err[:, 2, 1] - rotation_matrix_err[:, 1, 2]
    orn_err[:, 1] = rotation_matrix_err[:, 0, 2] - rotation_matrix_err[:, 2, 0]
    orn_err[:, 2] = rotation_matrix_err[:, 1, 0] - rotation_matrix_err[:, 0, 1]
    orn_err = orn_err / 2.0

    # 合并位置误差和旋转误差
    dpose = torch.cat([pos_err, orn_err], dim=-1).unsqueeze(-1)
    # print("dpose.size: ",dpose.size())
    return dpose

def create_transformation_matrices(tensor):
    """
    将位置和四元数转换为变换矩阵。
    tensor: 形状为 (n, 7) 的张量，前三列为位置，后四列为四元数
    返回: 形状为 (n, 4, 4) 的变换矩阵张量
    """
    positions = tensor[:, :3]
    quaternions = tensor[:, 3:]

    # 归一化四元数
    quaternions = quaternions / torch.norm(quaternions, dim=1, keepdim=True)
    qw, qx, qy, qz = (
        quaternions[:, 3],
        quaternions[:, 0],
        quaternions[:, 1],
        quaternions[:, 2],
    )

    # 构建旋转矩阵
    rotation_matrices = torch.stack(
        [
            1 - 2 * qy**2 - 2 * qz**2,
            2 * qx * qy - 2 * qz * qw,
            2 * qx * qz + 2 * qy * qw,
            2 * qx * qy + 2 * qz * qw,
            1 - 2 * qx**2 - 2 * qz**2,
            2 * qy * qz - 2 * qx * qw,
            2 * qx * qz - 2 * qy * qw,
            2 * qy * qz + 2 * qx * qw,
            1 - 2 * qx**2 - 2 * qy**2,
        ],
        dim=-1,
    ).reshape(-1, 3, 3)

    # 构建变换矩阵
    n = rotation_matrices.shape[0]
    transformation_matrices = torch.eye(4).repeat(n, 1, 1)
    transformation_matrices[:, :3, :3] = rotation_matrices
    transformation_matrices[:, :3, 3] = positions

    return transformation_matrices

def get_tf_mat(rb_states, base_idxs, feet_l_idxs):
    return torch.inverse(
        create_transformation_matrices(rb_states[base_idxs, :7])
    ) @ create_transformation_matrices(rb_states[feet_l_idxs, :7])

def invert_tf_mat(transform_matrices):
    """
    计算在feet坐标系下，baselink到feet的旋转平移矩阵

    参数:
    transform_matrices (torch.Tensor): 形状为 (n, 4, 4) 的旋转平移矩阵

    返回:
    torch.Tensor: 形状为 (n, 4, 4) 的逆旋转平移矩阵
    """
    n = transform_matrices.size(0)

    # 提取旋转矩阵和平移向量
    R = transform_matrices[:, :3, :3]  # 形状 (n, 3, 3)
    p = transform_matrices[:, :3, 3]  # 形状 (n, 3)

    # 计算逆旋转矩阵
    R_inv = R.transpose(1, 2)  # 形状 (n, 3, 3)

    # 计算新的平移向量
    p_inv = -torch.bmm(R_inv, p.unsqueeze(2)).squeeze(2)  # 形状 (n, 3)

    # 构建新的旋转平移矩阵
    transform_matrices_inv = torch.eye(4).repeat(n, 1, 1)  # 形状 (n, 4, 4)
    transform_matrices_inv[:, :3, :3] = R_inv
    transform_matrices_inv[:, :3, 3] = p_inv

    return transform_matrices_inv

def invert_jacobian(jacobian_bf, rotation_matrices):
    """
    计算从feet到baselink的雅可比矩阵

    参数:
    jacobian_bf (torch.Tensor): 形状为 (n, 6, m) 的从baselink到feet的雅可比矩阵
    rotation_matrices (torch.Tensor): 形状为 (n, 3, 3) 的旋转矩阵

    返回:
    torch.Tensor: 形状为 (n, 6, m) 的从feet到baselink的雅可比矩阵
    """
    n, _, _ = jacobian_bf.shape
    # 提取旋转矩阵
    R = rotation_matrices  # 形状 (n, 3, 3)
    # 计算逆旋转矩阵
    R_inv = R.transpose(1, 2)  # 形状 (n, 3, 3)
    # 构建转换矩阵
    T = torch.zeros((n, 6, 6), dtype=jacobian_bf.dtype, device=jacobian_bf.device)
    T[:, :3, :3] = R_inv
    T[:, 3:, 3:] = R_inv
    # 计算新的雅可比矩阵
    jacobian_fb = torch.bmm(T, jacobian_bf)
    return jacobian_fb

def invert_mass_matrix(mass_matrix_bf, rotation_matrices):
    """
    计算从feet到baselink的质量矩阵

    参数:
    mass_matrix_bf (torch.Tensor): 形状为 (n, 6, 6) 的从baselink到feet的质量矩阵
    rotation_matrices (torch.Tensor): 形状为 (n, 3, 3) 的旋转矩阵

    返回:
    torch.Tensor: 形状为 (n, 6, 6) 的从feet到baselink的质量矩阵
    """
    n = mass_matrix_bf.size(0)
    # 提取旋转矩阵
    R = rotation_matrices  # 形状 (n, 3, 3)
    # 计算逆旋转矩阵
    R_inv = R.transpose(1, 2)  # 形状 (n, 3, 3)
    # 构建转换矩阵
    T = torch.zeros((n, 6, 6), dtype=mass_matrix_bf.dtype, device=mass_matrix_bf.device)
    T[:, :3, :3] = R_inv
    T[:, 3:, 3:] = R_inv
    T_inv = torch.zeros(
        (n, 6, 6), dtype=mass_matrix_bf.dtype, device=mass_matrix_bf.device
    )
    T_inv[:, :3, :3] = R
    T_inv[:, 3:, 3:] = R
    # 计算新的质量矩阵
    mass_matrix_fb = torch.bmm(T, torch.bmm(mass_matrix_bf, T_inv))
    return mass_matrix_fb

def transform_velocity_to_feet_frame(v_feet_in_baselink, rotation_matrices):
    """
    将在baselink坐标系下的feet速度转换为在feet坐标系下的baselink速度

    参数:
    v_feet_in_baselink (torch.Tensor): 形状为 (n, 6) 的在baselink坐标系下的feet速度
    rotation_matrices (torch.Tensor): 形状为 (n, 3, 3) 的旋转矩阵

    返回:
    torch.Tensor: 形状为 (n, 6) 的在feet坐标系下的baselink速度
    """
    # 提取线速度和角速度
    v_lin_feet_in_baselink = v_feet_in_baselink[:, :3]  # 形状 (n, 3)
    v_ang_feet_in_baselink = v_feet_in_baselink[:, 3:]  # 形状 (n, 3)
    # 计算逆旋转矩阵
    R_inv = rotation_matrices.transpose(1, 2)  # 形状 (n, 3, 3)
    # 转换线速度
    v_lin_baselink_in_feet = -torch.bmm(R_inv, v_lin_feet_in_baselink.unsqueeze(2)).squeeze(2)  # 形状 (n, 3)
    # 转换角速度
    v_ang_baselink_in_feet = -torch.bmm(R_inv, v_ang_feet_in_baselink.unsqueeze(2)).squeeze(2)  # 形状 (n, 3)
    # 合并线速度和角速度
    print(R_inv[0])
    print(rotation_matrices[0])
    print(v_lin_feet_in_baselink[0],v_lin_baselink_in_feet[0])
    v_baselink_in_feet = torch.cat((v_lin_baselink_in_feet, v_ang_baselink_in_feet), dim=1)  # 形状 (n, 6)
    return v_baselink_in_feet
