
from torch import nn
import torch

# 均方根正则化
class RMSNorm(nn.Module):
    def __init__(self, dim:int, eps:float = 1e-12):
        super().__init__()
        self.eps = eps

        # 初始化RMSNort的参数（放缩因子）
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * (torch.rsqrt(x.pow(2).mean(-1, keepdim=True)) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def test_RMSNorm():
    rms = RMSNorm(4)
    x = torch.randn(1,8, 4)
    output = rms(x)
    print(output.shape)
    print("="*20)
    print(x)
    print("="*20)

if __name__ == "__main__":
    test_RMSNorm()
