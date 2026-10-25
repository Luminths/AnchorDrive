"""One CUDA kernel for the existing causal control projection, without CPU copies."""
import torch
import triton
import triton.language as tl


@triton.jit
def _project_scan(Reference, Initial, Previous, Output, Rows,
                  T: tl.constexpr, DT: tl.constexpr, AMIN: tl.constexpr,
                  AMAX: tl.constexpr, JMAX: tl.constexpr, WMAX: tl.constexpr,
                  WAMAX: tl.constexpr, LATMAX: tl.constexpr, VMAX: tl.constexpr,
                  KMAX: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = row < Rows
    speed = tl.load(Initial + row * 4 + 2, mask=valid, other=0.)
    acceleration = tl.load(Previous + row * 2, mask=valid, other=0.)
    omega = tl.load(Previous + row * 2 + 1, mask=valid, other=0.)
    for t in range(T):
        # Match PyTorch's tensor / Python-scalar path: compute the reciprocal
        # in Python precision, then multiply as float32 inside the kernel.
        lower = tl.maximum(AMIN, (-speed) * (1. / (DT + (-AMIN) / JMAX)))
        upper = tl.minimum(AMAX, (VMAX - speed) * (1. / (DT + AMAX / JMAX)))
        lower = tl.maximum(lower, acceleration - JMAX * DT)
        upper = tl.minimum(upper, acceleration + JMAX * DT)
        ref_acceleration = tl.load(Reference + (row * T + t) * 2, mask=valid, other=0.)
        acceleration = tl.maximum(lower, tl.minimum(upper, ref_acceleration))
        speed = speed + acceleration * DT
        bound = tl.minimum(WMAX, tl.div_rn(LATMAX, tl.maximum(speed, .1)))
        bound = tl.minimum(bound, tl.div_rn(LATMAX, tl.maximum(speed + AMAX * WMAX / WAMAX, .1)))
        bound = tl.minimum(bound, tl.maximum(speed, 0.) * KMAX)
        lower = tl.maximum(-bound, omega - WAMAX * DT)
        upper = tl.minimum(bound, omega + WAMAX * DT)
        ref_omega = tl.load(Reference + (row * T + t) * 2 + 1, mask=valid, other=0.)
        omega = tl.maximum(lower, tl.minimum(upper, ref_omega))
        tl.store(Output + (row * T + t) * 2, acceleration, mask=valid)
        tl.store(Output + (row * T + t) * 2 + 1, omega, mask=valid)


def project_controls_cuda(reference, initial_states, previous_controls, dt, limits):
    if reference.ndim != 3 or reference.shape[-1] != 2 or initial_states.shape != (reference.shape[0], 4):
        raise ValueError('Expected flattened [vehicle*sample,time,2] controls and [vehicle*sample,4] states')
    if reference.dtype != torch.float32 or initial_states.dtype != torch.float32 or previous_controls.dtype != torch.float32:
        raise ValueError('Fused control projection requires float32 tensors')
    reference = reference.contiguous()
    initial_states = initial_states.contiguous()
    previous_controls = previous_controls.contiguous()
    output = torch.empty_like(reference)
    _project_scan[(triton.cdiv(reference.shape[0], 32),)](
        reference, initial_states, previous_controls, output, reference.shape[0], reference.shape[1],
        dt, limits['acceleration_min'], limits['acceleration_max'], limits['jerk_max'],
        limits['yaw_rate_max'], limits['yaw_acceleration_max'], limits['lateral_acceleration_max'],
        limits['speed_max'], limits['curvature_max'], 32, num_warps=1, enable_fp_fusion=False)
    return output
