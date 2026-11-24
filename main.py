import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import platform
import time
from types import SimpleNamespace
import numpy as np
import numba
import torch
from src.adsp import Settings, estimate, generate_observations
from src.at_original import train_baseline
from src.plots import plot_figures

ROOT = Path(__file__).resolve().parent
METHODS = {'ising': ('original', 'at', 'adsp'), 'potts': ('at', 'adsp')}


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def specifications(config):
    runs = []
    for model, methods in METHODS.items():
        for index, truth in enumerate(config['betas']):
            for method in methods:
                seeds = config['ising_adsp_seeds'] if (model, method) == ('ising', 'adsp') else config['seeds']
                for seed in seeds:
                    values = config['adsp' if method == 'adsp' else 'baseline'][model]
                    if values['steps'] != 1000 or len(seeds) != 5:
                        raise ValueError('The figures require 1000 updates and 5 seeds per condition')
                    runs.append({'key': f'{model}_{method}_{truth:g}_{seed}', 'model': model,
                        'method': method, 'beta_truth': truth, 'seed': seed,
                        'data_seed': config['data_seed_offsets'][model]+seed*10+index if method == 'adsp' else seed,
                        'settings': {**values, 'seed': seed}})
    return runs


def experiment_spec(config):
    return {'configuration': config,
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'torch': str(torch.__version__), 'numba': numba.__version__},
        'source_sha256': {name: sha256(ROOT/name) for name in ('main.py', 'src/adsp.py', 'src/at_original.py')},
        'statistics': {'trajectory': 'mean across 5 seeds at each step',
            'band': 'mean +/- 1 sample SD, ddof=1', 'last_100': 'steps 901..1000 inclusive',
            'endpoint': 'state after update 1000'}}


def run_one(run):
    torch.set_num_threads(1)
    started = time.perf_counter()
    if run['method'] == 'adsp':
        cfg = Settings(**run['settings'])
        data = generate_observations(run['model'], run['beta_truth'], cfg, run['data_seed'])
        history, details = estimate(data, cfg, run['model'])
        details['data_sha256'] = hashlib.sha256(data.tobytes()).hexdigest()
    else:
        cfg = SimpleNamespace(**run['settings'])
        records = train_baseline(run['model'], run['beta_truth'], cfg, run['method'], progress=False)
        history = np.array([float(np.float32(cfg.beta_init))]+[r['beta'] for r in records])
        details = {'theta_parameters': 9}
    record = {**run, **details, 'origin': 'training', 'seconds': time.perf_counter()-started,
        'outer_cd_steps': cfg.steps, 'beta_updates': cfg.steps, 'theta_updates': cfg.steps,
        'record_points': len(history)}
    return run['key'], history, record


def save_results(output, spec, trajectories, records):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    temporary = output/'trajectories.tmp.npz'
    np.savez_compressed(temporary, **{key: trajectories[key] for key in sorted(trajectories)})
    temporary.replace(output/'trajectories.npz')
    manifest = {'experiment': spec, 'trajectories_sha256': sha256(output/'trajectories.npz'),
                'runs': {key: records[key] for key in sorted(records)}}
    temporary = output/'manifest.tmp.json'
    temporary.write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    temporary.replace(output/'manifest.json')


def load_results(output, spec):
    path = output/'manifest.json'
    if not path.exists():
        if (output/'trajectories.npz').exists():
            raise ValueError(f'Missing manifest: {output}')
        return {}, {}
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if manifest['experiment'] != spec:
        raise ValueError('Configuration or source changed; choose a new --output directory')
    if sha256(output/'trajectories.npz') != manifest['trajectories_sha256']:
        raise ValueError('Trajectory file checksum mismatch')
    with np.load(output/'trajectories.npz', allow_pickle=False) as saved:
        values = {key: saved[key] for key in saved.files}
    return values, manifest['runs']


def check_results(runs, trajectories, records):
    by_key = {run['key']: run for run in runs}
    if set(trajectories)-set(by_key) or set(records) != set(trajectories):
        raise ValueError('Saved runs do not match the experiment')
    for key, values in trajectories.items():
        run, record = by_key[key], records[key]
        if values.shape != (1001,) or not np.isfinite(values).all():
            raise ValueError(f'Incomplete or nonfinite trajectory: {key}')
        if not np.isclose(values[0], run['settings']['beta_init'], rtol=0, atol=1e-7):
            raise ValueError(f'Incorrect initial state: {key}')
        if any(record[field] != run[field] for field in run):
            raise ValueError(f'Run configuration mismatch: {key}')
        if any(record[field] != 1000 for field in ('outer_cd_steps', 'beta_updates', 'theta_updates')):
            raise ValueError(f'Incorrect update counts: {key}')


def main():
    parser = argparse.ArgumentParser(description='Run Ising/Potts experiments and generate two learning-trajectory PDFs.')
    parser.add_argument('--plots-only', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT/'output')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    output = args.output.resolve()
    config = json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
    spec, runs = experiment_spec(config), specifications(config)
    try:
        trajectories, records = load_results(output, spec)
        check_results(runs, trajectories, records)
        missing = [run for run in runs if run['key'] not in trajectories]
        if args.plots_only and missing:
            raise ValueError('Missing trajectories:\n'+'\n'.join(run['key'] for run in missing))
        print(f'{len(trajectories)} saved runs; {len(missing)} runs to execute.', flush=True)
        if missing:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                for future in as_completed([pool.submit(run_one, run) for run in missing]):
                    key, values, record = future.result()
                    trajectories[key], records[key] = values, record
                    save_results(output, spec, trajectories, records)
                    print(f'[{len(trajectories)}/{len(runs)}] {key}: step 1000 = {values[-1]:.6f}', flush=True)
        check_results(runs, trajectories, records)
        for path in plot_figures(output, config, runs, trajectories):
            print(path, flush=True)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f'{error}\n')


if __name__ == '__main__':
    main()
