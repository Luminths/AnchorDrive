"""Optional constant attack weights before the independent temporal fade."""
import math


def attacker_weight_override(weights, names, attackers, loss_names, override):
    """Override only designated attackers; preserve helper/BG loss rows.

    These are soft optimization weights, not action bounds or acceptance rules.
    Per-prediction collision decay remains the caller's separate time scale.
    """
    if override is None:
        return weights, None
    if weights is None or weights.shape != (len(names), len(loss_names)):
        raise ValueError('Attack guidance requires actor-by-loss weights')
    if set(override) != set(loss_names):
        raise ValueError('Specify each enabled loss exactly once for attack weights')
    values = [float(override[name]) for name in loss_names]
    if any(not math.isfinite(value) or value < 0 for value in values) or sum(values) <= 0:
        raise ValueError('Attack weights must be finite, nonnegative and nonzero')
    adjusted = weights.clone()
    original = {}
    for index, actor in enumerate(names):
        if actor not in attackers:
            continue
        original[actor] = weights[index].detach().cpu().tolist()
        adjusted[index] = weights.new_tensor(values)
    return adjusted, dict(method='constant_attack_weights_before_temporal_fade',
                         loss_names=list(loss_names), values=values,
                         original_wrapper_weights_by_actor=original)
