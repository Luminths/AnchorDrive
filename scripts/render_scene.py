"""Render an exported rollout to MP4 with SafeSim's nuScenes visualizer."""
import argparse
from pathlib import Path
from types import SimpleNamespace


def render_scene(source, output, cache, view_m=90.):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import dill
    import h5py
    import imageio.v2 as imageio
    import numpy as np
    from scripts.visualize import NuscRenderer, draw_scene_data

    with h5py.File(source, 'r') as handle:
        name = next(iter(handle))
        group = handle[name]
        data = {key: value[:] for key, value in group.items()}
        dt = float(group.attrs.get('dt', 0.1))
    ids = [value.decode() if isinstance(value, bytes) else str(value) for value in data['agent_id']]
    ego = ids.index('ego')
    attackers = np.flatnonzero(data['controlled'])
    if not len(attackers):
        raise ValueError('No controlled vehicle in the exported rollout')
    adv = int(attackers[0])
    if data['yaw'].ndim == 3:
        data['yaw'] = data['yaw'].squeeze(-1)
    metadata_path = cache/'nusc_trainval'/name/f'scene_metadata_dt{dt:.2f}.dill'
    with metadata_path.open('rb') as handle:
        metadata = dill.load(handle)
    dataset = SimpleNamespace(cache_path=cache, augmentations=[],
                              num_scenes=lambda: 1, get_scene=lambda index: metadata)
    renderer = NuscRenderer(dataset, raster_size=round(view_m*2), resolution=2)
    center = np.mean(data['centroid'][[ego, *attackers]].reshape(-1, 2), axis=0)
    patch = renderer.render(center, 0, name)
    renderer.render = lambda **kwargs: patch
    output.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output, fps=1./dt, codec='libx264', macro_block_size=4) as writer:
        for step in range(data['centroid'].shape[1]):
            fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
            try:
                fig.subplots_adjust(left=0, right=1, bottom=0, top=.94)
                draw_scene_data(ax, name, data, step, renderer, draw_trajectory=True,
                                traj_len=20, ras_pos=center, ctrl_agent_id=adv,
                                ego_agent_id=ego, show_scene_labels=False,
                                show_llm_plan=False, show_adv_proposals=False)
                fig.suptitle(f'{name} | t = {data["time_s"][step]:.1f} s')
                fig.canvas.draw()
                writer.append_data(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
            finally:
                plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=Path.home()/'.unified_data_cache')
    parser.add_argument('--view-m', type=float, default=90.)
    args = parser.parse_args()
    if args.view_m <= 0:
        parser.error('--view-m must be positive')
    if args.output.exists():
        parser.error('Choose a new output video path')
    render_scene(args.scene_dir/'data.hdf5', args.output, args.cache.expanduser(), args.view_m)


if __name__ == '__main__':
    main()
