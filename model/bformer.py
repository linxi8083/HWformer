'''
    Dynamic_conv + transformer + ssim + Denosing
    动态卷积 + Transformer + 结构相似性 + 去噪
'''

import sys
sys.path.append('../')

from model_common import common
from model_common.bformer_transformerlayer import *


def make_model(args):
    """创建模型实例的工厂函数"""
    return DTSD(args), 1


class DTSD(nn.Module):
    """DTSD模型类 - 动态卷积Transformer去噪模型"""

    def __init__(self, args, conv=common.default_conv):
        """初始化函数
        Args:
            args: 配置参数
            conv: 卷积类型，默认为普通卷积
        """
        super(DTSD, self).__init__()

        self.scale_idx = 0  # 尺度索引

        self.args = args

        # 模型参数设置
        n_feats = args.n_feats  # 特征通道数
        kernel_size = 3  # 卷积核大小
        act = nn.ReLU(True)  # 激活函数

        # 均值调整模块 - 用于数据预处理和后处理
        self.sub_mean = common.MeanShift(args.rgb_range)  # sub = 减，减去均值
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)  # add = 加，加上均值

        # 头部网络 - 特征提取
        self.head = nn.Sequential(
            conv(args.n_colors, n_feats, kernel_size),  # conv1: 输入通道到特征通道的转换
            common.ResBlock(conv, n_feats, 5, act=act),  # conv2: 残差块，扩张因子为5
            common.ResBlock(conv, n_feats, 5, act=act),  # conv3: 残差块，扩张因子为5
        )

        # Transformer相关参数设置
        self.window_size = 48  # 窗口大小
        self.img_size = 96     # 图像尺寸
        self.num_layers = 8    # Transformer层数
        self.num_heads = 9     # 注意力头数
        self.patch_dim = 3     # 补丁维度

        # 随机深度衰减规则（Stochastic Depth）
        drop_path_rate = 0.1
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.num_layers)]

        # 定义各层的注意力模式
        layers = [
            'horizontal', 'vertical', 'no_shift',    # 水平、垂直、无偏移
            'horizontal', 'vertical', 'no_shift',    # 重复模式
            'horizontal', 'vertical'                 # 最后两层
        ]

        # 确保层数与模式数量匹配
        assert self.num_layers == len(layers)

        # 主体网络 - 使用三种不同的Transformer层
        # 第一个CBformer层：高维度注意力
        self.body1 = CBformerLayer2(dim=n_feats, img_size=self.img_size, win_size=self.img_size,
                                    num_heads=self.num_heads * 4, patch_dim=self.patch_dim * 2)

        # 第二个WBformer层序列：带动态MLP的窗口注意力
        self.body2 = nn.Sequential(
            WBformerLayer_with_DMlp(dim=n_feats, img_size=self.img_size, num_heads=self.num_heads,
                                    patch_dim=self.patch_dim, window_size=self.window_size, layers=layers, drop_path=dpr)
        )

        # 第三个CBformer层：高维度注意力
        self.body3 = CBformerLayer2(dim=n_feats, img_size=self.img_size, win_size=self.img_size,
                                    num_heads=self.num_heads * 4, patch_dim=self.patch_dim * 2)

        # 尾部网络 - 输出重构
        self.tail = conv(n_feats, args.n_colors, kernel_size)

    def forward(self, x):
        """前向传播
        Args:
            x: 输入图像
        Returns:
            去噪后的残差结果
        """
        y = x  # 保存原始输入

        # 1. 头部特征提取
        x = self.head(x)

        # 2. 主体Transformer处理
        x = self.body1(x)  # 第一个Transformer块
        x = self.body2(x)  # 第二个Transformer块序列
        x = self.body3(x)  # 第三个Transformer块

        # 3. 尾部输出重构
        out = self.tail(x)

        # 4. 残差学习：输出噪声图，最终结果为 y - out
        # print("============")
        return y - out