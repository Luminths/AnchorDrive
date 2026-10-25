"""Run all 150 scenes, retaining independent logs and continuing after errors."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SCENES = tuple(range(150))


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/full150'))
    parser.add_argument('--workers', type=int, default=1, help='Maximum concurrent scene processes')
    parser.add_argument('--devices', default='cuda:0', help='Comma-separated diffusion devices')
    parser.add_argument('--resume', action='store_true', help='Skip completed scenes; retry errors in new attempt directories')
    args = parser.parse_args()
    devices = [d.strip() for d in args.devices.split(',') if d.strip()]
    if args.workers < 1 or not devices:
        parser.error('Provide a positive worker count and at least one device')
    if not os.environ.get('OPENAI_API_KEY', '').strip():
        parser.error('Set OPENAI_API_KEY before running the 150-scene pipeline')
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        parser.error('Output is not empty; use --resume or choose a new directory')
    output.mkdir(parents=True, exist_ok=True)
    slots = queue.Queue()
    for i in range(args.workers): slots.put(devices[i % len(devices)])
    rows = {}
    pending = []
    for scene in SCENES:
        marker = output/f'scene_{scene:03d}'/'completion.json'
        if args.resume and marker.exists():
            previous = json.loads(marker.read_text())
            result = Path(previous.get('run_directory',''))/'stage2'/f'execute_{scene:03d}'/f'scene_{scene:03d}'/'result.json'
            if previous.get('status') == 'completed' and result.is_file():
                rows[scene] = previous
                continue
        pending.append(scene)
    def work(scene):
        device = slots.get()
        try:
            directory = output/f'scene_{scene:03d}'
            directory.mkdir(exist_ok=True)
            attempt = 1
            while (directory/f'attempt_{attempt:02d}').exists(): attempt += 1
            target = directory/f'attempt_{attempt:02d}'
            log = directory/f'attempt_{attempt:02d}.log'
            # Reserve the directory before launch, so interrupted attempts stay intact.
            target.mkdir()
            command = [sys.executable,'-m','scripts.run_pipeline','--scenes',str(scene),
                       '--output',str(target),'--device',device]
            env = dict(os.environ)
            for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: env[key] = '1'
            started = time.monotonic()
            with log.open('w') as stream:
                process = subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            row = dict(scene=scene,status='completed' if process.returncode == 0 else 'error',
                       returncode=process.returncode,run_directory=str(target),log=str(log),device=device,
                       elapsed_s=round(time.monotonic()-started,3))
            save(directory/'completion.json',row)
            return row
        finally:
            slots.put(device)
    def progress():
        save(output/'progress.json',dict(total=150,finished=len(rows),
            completed=sum(r['status']=='completed' for r in rows.values()),
            errors=sum(r['status']=='error' for r in rows.values()),scenes=[rows[i] for i in sorted(rows)]))
    progress()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(work,s):s for s in pending}
        for future in as_completed(jobs):
            scene = jobs[future]
            try: rows[scene] = future.result()
            except Exception as exc: rows[scene] = dict(scene=scene,status='error',error=str(exc))
            progress()
            print(f"[{len(rows)}/150] scene {scene:03d}: {rows[scene]['status']}",flush=True)
    return int(any(r['status'] == 'error' for r in rows.values()))


if __name__ == '__main__':
    raise SystemExit(main())
