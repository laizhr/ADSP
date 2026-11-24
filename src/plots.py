from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

COLORS = {'original': '#C44E52', 'at': '#4477AA', 'adsp': '#228833'}
LABELS = {'original': 'Original', 'at': 'AT', 'adsp': 'ADSP'}
CHECKPOINTS = (0, 200, 400, 600, 800, 1000)


def plot_figures(output, config, runs, trajectories):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.titlesize': 11, 'axes.labelsize': 11, 'xtick.labelsize': 9,
        'ytick.labelsize': 9, 'pdf.fonttype': 42, 'path.simplify': False,
        'axes.spines.top': False, 'axes.spines.right': False})
    grouped = {}
    for run in runs:
        group = (run['model'], run['beta_truth'], run['method'])
        grouped.setdefault(group, []).append(trajectories[run['key']])
    grouped = {key: np.stack(values) for key, values in grouped.items()}
    files = []
    for model in ('ising', 'potts'):
        fig, axes = plt.subplots(4, 2, figsize=(13.5, 18), sharex=True, sharey='row')
        fig.subplots_adjust(left=.075, right=.975, top=.915, bottom=.145, hspace=.88, wspace=.21)
        comparisons = (('original', 'at'), ('original', 'adsp')) if model == 'ising' else (('at',), ('adsp',))
        for row, truth in enumerate(config['betas']):
            bounds = [truth]
            for col, methods in enumerate(comparisons):
                ax = axes[row, col]
                endpoint_labels = []
                for method in methods:
                    values = grouped[(model, truth, method)]
                    mean = values.mean(0)
                    sd = values.std(0, ddof=1)
                    color, label = COLORS[method], LABELS[method]
                    ax.plot(np.arange(1001), mean, color=color, lw=1.2, label=label, zorder=3)
                    ax.fill_between(np.arange(1001), mean-sd, mean+sd, color=color, alpha=.17, linewidth=0)
                    ax.axhline(values[:, 901:1001].mean(), color=color, ls=':', lw=1.25,
                               label=f'{label}: Mean of Last 100')
                    ax.scatter(CHECKPOINTS, mean[list(CHECKPOINTS)], s=20, edgecolor=color,
                               facecolor='white', linewidth=.9, zorder=4, clip_on=False)
                    ax.scatter([1000], [mean[1000]], s=32, marker='D', color=color, zorder=5, clip_on=False)
                    endpoint_labels.append(f'{label} = {mean[1000]:.4f}')
                    bounds.extend([values.min(), values.max(), (mean-sd).min(), (mean+sd).max()])
                ax.axhline(truth, color='#262626', ls='--', lw=1.15, label='Ground truth')
                comparison = ' vs '.join(LABELS[m] for m in methods)
                if model == 'potts': comparison = 'PGD+' + comparison
                title = 'Ising' if model == 'ising' else '3-state Potts'
                ax.set_title(f'{title} | Ground-truth '+rf'$\beta={truth:.2f}$'+f' | {comparison}', pad=29)
                ax.text(0, 1.025, 'Step 1000: '+';  '.join(endpoint_labels), transform=ax.transAxes,
                        fontsize=9, color='#202020', va='bottom')
                ax.set(xlim=(0, 1000), xticks=CHECKPOINTS, xlabel='Contrastive Divergence Step', ylabel=r'$\tilde{\beta}$')
                ax.tick_params(labelbottom=True, labelleft=True)
                ax.grid(alpha=.13, linewidth=.7)
                handles, labels = ax.get_legend_handles_labels()
                order = [labels.index(LABELS[m]) for m in methods]+[labels.index('Ground truth')]
                order += [labels.index(f'{LABELS[m]}: Mean of Last 100') for m in methods]
                ax.legend([handles[i] for i in order], [labels[i] for i in order], loc='upper left',
                          bbox_to_anchor=(0, -.31), ncol=2, fontsize=8.4, frameon=False,
                          borderaxespad=0, columnspacing=1.5, handlelength=2.5)
            low, high = min(bounds), max(bounds)
            margin = max((high-low)*.07, .01)
            axes[row, 0].set_ylim(low-margin, high+margin)
        title = 'Ising learning trajectories' if model == 'ising' else '3-state Potts learning trajectories'
        fig.suptitle(title, fontsize=17, y=.98)
        fig.text(.075, .034,
            'Solid lines: mean across 5 seeds. Shading: mean ± 1 sample SD (ddof=1).\n'
            'Dotted lines: Mean of Last 100 (steps 901–1000). Dashed black: Ground truth.\n'
            'Circles: steps 0, 200, 400, 600, 800, 1000. Diamonds and annotations: Step 1000.',
            fontsize=9.3, linespacing=1.65, color='#333333')
        name = 'figure_a17_ising.pdf' if model == 'ising' else 'figure_table1_potts.pdf'
        path = Path(output)/name
        fig.savefig(path, facecolor='white')
        plt.close(fig)
        files.append(path.resolve())
    return files
