"""Differentiable oriented-box separation using both vehicles' full extents."""
import torch
import torch.nn.functional as F


def pairwise_box_separation(positions, yaws, extents):
    """[B,B,S,T] signed separation: <=0 means SAT box overlap/contact.

    This is the maximum separating-axis gap, not Euclidean polygon distance.
    Axes come from both boxes, so crossing/turning widths are included.
    """
    delta = positions[:, None] - positions[None, :]
    a = yaws[:, None, ..., 0]
    b = yaws[None, :, ..., 0]
    ea = extents[:, None, None, None, :]
    eb = extents[None, :, None, None, :]
    gaps = []
    for angle in (a, a + torch.pi / 2, b, b + torch.pi / 2):
        distance = torch.abs(delta[..., 0] * torch.cos(angle) + delta[..., 1] * torch.sin(angle))
        ra = (ea[..., 0] * torch.abs(torch.cos(a-angle)) + ea[..., 1] * torch.abs(torch.sin(a-angle))) / 2
        rb = (eb[..., 0] * torch.abs(torch.cos(b-angle)) + eb[..., 1] * torch.abs(torch.sin(b-angle))) / 2
        gaps.append(distance - ra - rb)
    return torch.stack(gaps).amax(dim=0)


def body_gap_avoidance_loss(positions, yaws, extents, margin_m=.25):
    gap = pairwise_box_separation(positions, yaws, extents)
    # One-metre transition provides a nonzero approach gradient before contact;
    # unlike a shell-shaped density, deep overlap does not reduce the penalty.
    return F.softplus(margin_m - gap).square()
