import torch
import torch.nn.functional as F
import math

class KANLinear(torch.nn.Module):
    def __init__(
            self,
            in_features,  # 输入特征的数量
            out_features,  # 输出特征的数量
            grid_size=5,  # 网格的大小 original is 5
            spline_order=4,  # 样条的阶数
            scale_noise=0.1,  # 噪声的缩放因子
            scale_base=1.0,  # 基础权重的缩放因子
            scale_spline=1.0,  # 样条权重的缩放因子
            enable_standalone_scale_spline=True,  # 是否启用独立的样条缩放因子
            base_activation=torch.nn.SiLU,  # 基础激活函数#Silu
            grid_eps=0.02,  # 网格的 epsilon 值
            grid_range=[-1, 1],  # 网格的范围
    ):
        super(KANLinear, self).__init__()  # 调用父类的构造函数
        self.in_features = in_features  # 保存输入特征数
        self.out_features = out_features  # 保存输出特征数
        self.grid_size = grid_size  # 保存网格大小
        self.spline_order = spline_order  # 保存样条阶数

        # 计算单个网格的高度
        h = (grid_range[1] - grid_range[0]) / grid_size
        # 创建网格，扩展到输入特征的数量
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)  # 扩展到 (in_features, grid_size + 2 * spline_order + 1)
            .contiguous()
        )
        self.register_buffer("grid", grid)  # 注册网格为缓冲区，以便在模型保存和加载时保持

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))  # 基础权重参数
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)  # 样条权重参数
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)  # 独立样条缩放因子
            )

        # 保存缩放因子和标志
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()  # 初始化基础激活函数
        self.grid_eps = grid_eps  # 保存网格 epsilon 值

        self.reset_parameters()  # 重置参数

    def reset_parameters(self):
        # Kaiming 初始化基础权重
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():  # 不计算梯度
            # 生成噪声并缩放
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            # 用噪声和曲线计算样条权重
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order: -self.spline_order],  # 使用网格计算曲线
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # Kaiming 初始化样条缩放因子
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        计算给定输入张量的 B 样条基函数。

        Args:
            x (torch.Tensor): 形状为 (batch_size, in_features) 的输入张量。

        Returns:
            torch.Tensor: 形状为 (batch_size, in_features, grid_size + spline_order) 的 B 样条基函数张量。
        """
        assert x.dim() == 2 and x.size(1) == self.in_features  # 确保输入张量的维度正确

        grid: torch.Tensor = (
            self.grid  # (in_features, grid_size + 2 * spline_order + 1)
        )
        x = x.unsqueeze(-1)  # 将 x 扩展为 (batch_size, in_features, 1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)  # 计算基函数的初始值
        for k in range(1, self.spline_order + 1):
            # 根据 B 样条定义递归计算基函数
            bases = (
                (x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)]) * bases[:, :, :-1
                ]
            ) + (
                (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:(-k)]) * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )  # 确保输出维度正确
        return bases.contiguous()  # 返回连续的张量

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        计算插值给定点的曲线系数。

        Args:
            x (torch.Tensor): 形状为 (batch_size, in_features) 的输入张量。
            y (torch.Tensor): 形状为 (batch_size, in_features, out_features) 的输出张量。

        Returns:
            torch.Tensor: 形状为 (out_features, in_features, grid_size + spline_order) 的系数张量。
        """
        assert x.dim() == 2 and x.size(1) == self.in_features  # 确保输入维度正确
        assert y.size() == (x.size(0), self.in_features, self.out_features)  # 确保输出维度正确

        A = self.b_splines(x).transpose(0, 1)  # 转置以得到 (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # 转置以得到 (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(A, B).solution  # 线性最小二乘法求解
        result = solution.permute(2, 0, 1)  # 转置得到 (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )  # 确保输出维度正确
        return result.contiguous()  # 返回连续的张量

    @property
    def scaled_spline_weight(self):
        # 根据是否启用独立缩放因子调整样条权重
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        # 前向传播方法
        assert x.size(-1) == self.in_features  # 确保输入特征数匹配
        original_shape = x.shape  # 保存原始形状
        x = x.contiguous().view(-1, self.in_features)  # 将输入展平以适应线性层

        # 计算基础输出
        base_output = F.linear(self.base_activation(x), self.base_weight)
        # 计算样条输出
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),  # 展平样条基函数的输出
            self.scaled_spline_weight.view(self.out_features, -1),  # 展平样条权重
        )
        output = base_output + spline_output  # 合并基础输出和样条输出

        output = output.view(*original_shape[:-1], self.out_features)  # 恢复输出的形状
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        """
        更新网格以适应输入数据的分布。

        Args:
            x (torch.Tensor): 形状为 (batch_size, in_features) 的输入张量。
            margin (float): 网格更新时的边距。
        """
        assert x.dim() == 2 and x.size(1) == self.in_features  # 确保输入维度正确
        batch = x.size(0)  # 获取批大小

        splines = self.b_splines(x)  # 计算样条基函数
        splines = splines.permute(1, 0, 2)  # 转置以得到 (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # 获取原始样条权重
        orig_coeff = orig_coeff.permute(1, 2, 0)  # 转置以得到 (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # 计算未缩减的样条输出
        unreduced_spline_output = unreduced_spline_output.permute(1, 0, 2)  # 转置为 (batch, in, out)

        # 对每个通道单独排序以收集数据分布
        x_sorted = torch.sort(x, dim=0)[0]  # 排序输入数据
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]  # 自适应网格

        # 计算均匀步长
        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )  # 生成均匀网格

        # 根据均匀网格和自适应网格生成最终网格
        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)  # 更新网格
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))  # 更新样条权重

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        计算正则化损失。

        模拟原始的 L1 正则化，因为原始实现需要计算绝对值和熵，
        但这些是在 F.linear 函数后面隐藏的中间张量中，导致内存效率低下。

        L1 正则化现在计算为样条权重的平均绝对值。
        """
        l1_fake = self.spline_weight.abs().mean(-1)  # 计算样条权重的平均绝对值
        regularization_loss_activation = l1_fake.sum()  # L1 正则化损失
        p = l1_fake / regularization_loss_activation  # 计算概率分布
        regularization_loss_entropy = -torch.sum(p * p.log())  # 计算熵损失
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy  # 返回总正则化损失
        )

