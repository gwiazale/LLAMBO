"""
Build PyTorch model from NAS-Bench-201 architecture string.
Used to run zero-shot proxies and to train the selected architecture.
Arch string format: |op~0|+|op~0|op~1|+|op~0|op~1|op~2|
Edges: (0->1), (0->2), (1->2), (0->3), (1->3), (2->3) with ops in that order.
"""

import random
import torch
from torch import nn

# Same op names as in NAS-Bench-201 arch strings
OPS = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]


def arch_str_to_ops(arch_str):
    """Parse arch string to list of 6 op names (one per edge)."""
    parts = [p.strip() for p in arch_str.split("|") if p.strip() and "~" in p]
    if len(parts) != 6:
        raise ValueError("Expected 6 edges in arch string, got %s" % len(parts))
    return [p.split("~")[0] for p in parts]


def index_to_arch_str(index):
    """Map index in [0, 15624] to NAS-Bench-201 arch string (5^6 = 15625)."""
    if not 0 <= index < 15625:
        raise ValueError("index must be in [0, 15624]")
    edges = []
    for _ in range(6):
        index, r = divmod(index, 5)
        edges.append(OPS[r])
    # Edge order: 0->1, 0->2, 1->2, 0->3, 1->3, 2->3
    return "|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|".format(
        edges[0], edges[1], edges[2], edges[3], edges[4], edges[5]
    )


def arch_str_to_index(arch_str):
    """Map arch string to index in [0, 15624]."""
    ops = arch_str_to_ops(arch_str)
    idx = 0
    for i, op in enumerate(ops):
        if op not in OPS:
            raise ValueError("Unknown op: %s" % op)
        idx += OPS.index(op) * (5 ** i)
    return idx


def mutate_arch_string_simple(arch_str):
    """
    Mutate one operation in the NAS-Bench-201 cell string.
    Format: |op~0|+|op~0|op~1|+|op~0|op~1|op~2| (6 edges).
    """
    parts = [p.strip() for p in arch_str.split("|") if p.strip() and "~" in p]
    if len(parts) != 6:
        return None
    idx = random.randint(0, 5)
    op_node = parts[idx]
    op, node = op_node.split("~")
    new_op = random.choice([o for o in OPS if o != op])
    parts[idx] = "{}~{}".format(new_op, node)
    return "|{}|+|{}|{}|+|{}|{}|{}|".format(
        parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]
    )


class ReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride=1):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(C_out),
        )

    def forward(self, x):
        return self.op(x)


class NASBench201Cell(nn.Module):
    """Single cell: 4 nodes, 6 edges. Node 0 = input; 1,2,3 computed from previous nodes.
    In reduce cells (stride=2), edges from node 0 (0->1, 0->2, 0->3) use stride 2."""

    def __init__(self, arch_str, C_in, C_out, stride=1):
        super().__init__()
        self.ops = arch_str_to_ops(arch_str)
        self._ops = nn.ModuleList()
        # Edge 0: 0->1
        self._ops.append(self._make_op(self.ops[0], C_in, C_out, stride))
        # Edge 1: 0->2
        self._ops.append(self._make_op(self.ops[1], C_in, C_out, stride))
        # Edge 2: 1->2
        self._ops.append(self._make_op(self.ops[2], C_out, C_out, 1))
        # Edge 3: 0->3
        self._ops.append(self._make_op(self.ops[3], C_in, C_out, stride))
        # Edge 4: 1->3
        self._ops.append(self._make_op(self.ops[4], C_out, C_out, 1))
        # Edge 5: 2->3
        self._ops.append(self._make_op(self.ops[5], C_out, C_out, 1))
        self.C_out = C_out

    def _make_op(self, op_name, C_in, C_out, stride):
        if op_name == "none":
            return Zero()
        if op_name == "skip_connect":
            return nn.Identity() if C_in == C_out and stride == 1 else FactorizedReduce(C_in, C_out)
        if op_name == "nor_conv_1x1":
            return ReLUConvBN(C_in, C_out, 1, stride)
        if op_name == "nor_conv_3x3":
            return ReLUConvBN(C_in, C_out, 3, stride)
        if op_name == "avg_pool_3x3":
            return nn.Sequential(
                nn.AvgPool2d(3, stride=stride, padding=1),
                nn.Conv2d(C_in, C_out, 1, bias=False),
                nn.BatchNorm2d(C_out),
            )
        raise ValueError("Unknown op: %s" % op_name)

    def forward(self, x):
        x0 = x
        x1 = self._ops[0](x0)
        x2 = self._ops[1](x0) + self._ops[2](x1)
        x3 = self._ops[3](x0) + self._ops[4](x1) + self._ops[5](x2)
        return x3


class Zero(nn.Module):
    def forward(self, x):
        return x.mul(0.0)


class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, 1, stride=2, bias=False),
            nn.BatchNorm2d(C_out),
        )

    def forward(self, x):
        return self.conv(x)


class NASBench201TinyNet(nn.Module):
    """
    Tiny net for CIFAR: stem + 3 stages (16, 32, 64 channels), 3 cells per stage.
    Same cell topology used in all stages. First cell of stage 2 and 3 use stride 2.
    """

    def __init__(self, arch_str, num_classes=10):
        super().__init__()
        self.arch_str = arch_str
        self.num_classes = num_classes
        C_base = 16
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_base, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(C_base),
        )
        self.cells = nn.ModuleList()
        # Stage 1: 3 cells, 16 channels, stride 1
        self.cells.append(NASBench201Cell(arch_str, C_base, C_base, 1))
        self.cells.append(NASBench201Cell(arch_str, C_base, C_base, 1))
        self.cells.append(NASBench201Cell(arch_str, C_base, C_base, 1))
        # Stage 2: 3 cells, 32 channels, first stride 2
        self.cells.append(NASBench201Cell(arch_str, C_base, C_base * 2, 2))
        self.cells.append(NASBench201Cell(arch_str, C_base * 2, C_base * 2, 1))
        self.cells.append(NASBench201Cell(arch_str, C_base * 2, C_base * 2, 1))
        # Stage 3: 3 cells, 64 channels, first stride 2
        self.cells.append(NASBench201Cell(arch_str, C_base * 2, C_base * 4, 2))
        self.cells.append(NASBench201Cell(arch_str, C_base * 4, C_base * 4, 1))
        self.cells.append(NASBench201Cell(arch_str, C_base * 4, C_base * 4, 1))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(C_base * 4, num_classes)

    def forward(self, x):
        x = self.stem(x)
        for cell in self.cells:
            x = cell(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def forward_pre_GAP(self, x):
        """Features before global average pooling (for Zen-NAS and similar)."""
        x = self.stem(x)
        for cell in self.cells:
            x = cell(x)
        return x

    def get_model_size(self):
        """Number of parameters (for Params proxy)."""
        return sum(p.numel() for p in self.parameters())

    def get_FLOPs(self, resolution=32):
        """Approximate FLOPs for one forward at given resolution (for Flops proxy)."""
        try:
            import thop
            x = torch.zeros(1, 3, resolution, resolution)
            flops, _ = thop.profile(self, inputs=(x,), verbose=False)
            return flops
        except Exception:
            return 0.0


def build_nasbench201(arch_str, num_classes=10):
    """Build NASBench201TinyNet from arch string."""
    return NASBench201TinyNet(arch_str, num_classes=num_classes)
