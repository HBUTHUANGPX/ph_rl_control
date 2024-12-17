import torch

class PIDController:
    def __init__(self, Kp, Ki, Kd, dt, device='cpu'):
        """
        初始化 PID 控制器。

        参数：
        - Kp (torch.Tensor): 比例增益，形状为 (n_robots, n_controls)
        - Ki (torch.Tensor): 积分增益，形状为 (n_robots, n_controls)
        - Kd (torch.Tensor): 微分增益，形状为 (n_robots, n_controls)
        - dt (float): 时间步长
        - device (str): 计算设备，例如 'cpu' 或 'cuda'
        """
        self.Kp = Kp.to(device)
        self.Ki = Ki.to(device)
        self.Kd = Kd.to(device)
        self.dt = dt
        self.device = device

        # 初始化积分和上一次误差
        self.integral = torch.zeros_like(self.Kp, device=device)
        self.prev_error = torch.zeros_like(self.Kp, device=device)

    def reset(self):
        """重置积分项和上一次误差。"""
        self.integral.zero_()
        self.prev_error.zero_()

    def compute(self, setpoint, measurement):
        """
        计算 PID 控制输出。

        参数：
        - setpoint (torch.Tensor): 期望值，形状为 (n_robots, n_controls)
        - measurement (torch.Tensor): 测量值，形状为 (n_robots, n_controls)

        返回：
        - control_output (torch.Tensor): 控制输出，形状为 (n_robots, n_controls)
        """
        # 计算误差
        error = setpoint - measurement

        # 计算积分项
        self.integral += error * self.dt

        # 计算微分项
        derivative = (error - self.prev_error) / self.dt

        # 更新上一次误差
        self.prev_error = error

        # 计算控制输出
        control_output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative

        return control_output
