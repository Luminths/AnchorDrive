"""Select a recorded baseline's native agents without changing their poses."""


def select_baseline_agents(agents, names):
    names=list(names)
    if 'ego' not in names or len(set(names))!=len(names):
        raise ValueError('Baseline native actor IDs must be unique and include ego')
    available={a.name:a for a in agents}
    missing=set(names)-set(available)
    if missing:
        raise ValueError('Baseline native actors unavailable at initial frame: '+','.join(sorted(missing)))
    return [available[name] for name in names]
