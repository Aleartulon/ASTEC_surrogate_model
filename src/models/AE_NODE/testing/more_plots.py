"""Compare several trained models on the same plot.

This is the multi-model counterpart of ``combine_metrics_in_one_plot`` in
``support_functions.py``.  That function took the ``plotting_data.pkl`` of a
*single* model and drew its three metrics on top of each other.  Here we instead
fix *one* metric (the reconstruction std error by default) and overlay the
*several* models, one colour per model, labelled by their number of latent
dimensions.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Human readable name of every metric stored inside plotting_data.pkl
# ----------------------------------------------------------------------
METRIC_DISPLAY = {
    'RMSE_divided_by_mean': r'RMSE$_{mean}(vr)$',
    'RMSE_divided_by_max':  r'RMSE$_{max}(vr)$',
    'RMSE_divided_by_std':  r'RMSE$_{std}(vr)$',
}


def _load_models(model_paths: dict) -> dict:
    """Load every ``plotting_data.pkl``.

    ``model_paths`` maps a model label (e.g. the latent dimension) to either the
    ``plotting_data.pkl`` file itself or the directory that contains
    ``combined_plots/plotting_data.pkl``.
    """
    data = {}
    for label, path in model_paths.items():
        pkl = path if path.endswith('.pkl') else os.path.join(path, 'combined_plots', 'plotting_data.pkl')
        with open(pkl, 'rb') as f:
            data[label] = pickle.load(f)
    return data


def _reference_labels(models_data: dict, variable: str, metric: str) -> list:
    """Union of the variable labels across every model, preserving order.

    Different models can carry a slightly different set of variables (e.g. one
    extra ``m magma`` face), so we cannot align the models by index.  We start
    from the ordering of the model that has the most labels and append any label
    that only shows up in another model.
    """
    base = max(models_data.values(),
               key=lambda d: len(d[variable][metric]['labels']))
    ref = list(base[variable][metric]['labels'])
    for d in models_data.values():
        for lab in d[variable][metric]['labels']:
            if lab not in ref:
                ref.append(lab)
    return ref


def _var_title(variable: str) -> str:
    if variable == 'cr_f':
        return r'$s_{cr},s_f$'
    elif variable == 'g':
        return r'$s_g, s_{B_1}, s_{B_2}$'
    else:
        return rf'$s_{{{variable}}}$'


def _plot_variable(ax, models_data: dict, variable: str, metric: str,
                   model_labels: list, colors: list, text_fontsize: int = 13,
                   label_prefix: str = 'latent '):
    """Draw one metric of one variable group, overlaying every model on ``ax``.

    Returns the (handles, labels) so a shared figure legend can be built.
    """
    ref_labels = _reference_labels(models_data, variable, metric)
    x_pos = np.arange(len(ref_labels))

    n = len(model_labels)
    # small horizontal dodge so the error bars of different models do not overlap
    offsets = np.linspace(-0.18, 0.18, n) if n > 1 else [0.0]

    handles, labels = [], []
    for i, ml in enumerate(model_labels):
        d = models_data[ml][variable][metric]
        lut = {lab: (v, u) for lab, v, u in zip(d['labels'], d['values'], d['uncertainties'])}
        ys = [lut[lab][0] if lab in lut else np.nan for lab in ref_labels]
        us = [lut[lab][1] if lab in lut else np.nan for lab in ref_labels]
        ep = ax.errorbar(x_pos + offsets[i], ys, yerr=us,
                         fmt='o', capsize=4,
                         color=colors[i % len(colors)],
                         label=f'{label_prefix}{ml}', alpha=0.85)
        handles.append(ep)
        labels.append(f'{label_prefix}{ml}')

    ax.set_ylabel('Error', fontsize=16)
    ax.hlines(0.5, xmin=0, xmax=max(len(ref_labels) - 1, 1),
              colors='green', linestyles='dashed')
    ax.set_yscale('log')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([])
    ax.tick_params(axis='y', labelsize=13)
    for pos, label_text in zip(x_pos, ref_labels):
        ax.text(pos, -0.02, label_text,
                ha='center', va='top',
                transform=ax.get_xaxis_transform(),
                fontsize=text_fontsize, rotation=45)

    return handles, labels


def combine_models_in_one_plot(model_paths: dict, where_to_save_data: str,
                               string_after_saving: str,
                               metric: str = 'RMSE_divided_by_std',
                               label_prefix: str = 'latent ',
                               suptitle: str = None):
    """Overlay several models for a single metric.

    Parameters
    ----------
    model_paths : dict
        Maps a model label (used in the legend, e.g. the number of latent
        dimensions) to its ``plotting_data.pkl`` file or its parent folder.
        The insertion order of the dict fixes the legend / colour order.
    where_to_save_data : str
        Base folder; a ``combined_plots`` sub-folder is created inside it.
    string_after_saving : str
        Suffix appended to every saved file name.
    metric : str, optional
        Which metric to plot, defaults to ``'RMSE_divided_by_std'``.  Any key of
        the ``plotting_data.pkl`` metric dictionaries is accepted
        (``RMSE_divided_by_mean``, ``RMSE_divided_by_max``,
        ``RMSE_divided_by_std``).
    label_prefix : str, optional
        Prepended to every model label in the legend.  The default keeps the
        original latent-dimension behaviour; pass ``''`` when the labels are
        already full names (e.g. 'Baseline', 'AE-NODE').
    suptitle : str, optional
        Title of the combined mosaic; defaults to the latent-dimension one.
    """
    models_data = _load_models(model_paths)
    model_labels = list(model_paths.keys())
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e', '#8c564b']
    metric_disp = METRIC_DISPLAY.get(metric, metric)

    os.makedirs(f'{where_to_save_data}/combined_plots/', exist_ok=True)

    # ---- one figure per variable group ----
    for variable in ['g', 'v', 'p', 'cr_f']:
        fig, ax = plt.subplots(1, 1, tight_layout=True)
        _plot_variable(ax, models_data, variable, metric, model_labels, colors, label_prefix=label_prefix)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)
        ax.set_title(rf'{_var_title(variable)},  {metric_disp}', fontsize=16)
        plt.savefig(f'{where_to_save_data}/combined_plots/{variable}_{metric}_{string_after_saving}.png',
                    dpi=300, bbox_inches='tight')
        plt.close(fig)

    # ---- combined mosaic: row 1 -> cr_f, p | row 2 -> g, v ----
    fig, axs_dict = plt.subplot_mosaic(
        [['cr_f', 'p'],
         ['g',    'v']],
        figsize=(14, 10),
        tight_layout=True
    )
    legend_handles, legend_labels = [], []
    for idx_var, variable in enumerate(['cr_f', 'p', 'g', 'v']):
        handles, labels = _plot_variable(axs_dict[variable], models_data,
                                         variable, metric, model_labels, colors, label_prefix=label_prefix)
        axs_dict[variable].set_title(_var_title(variable), fontsize=16)
        if idx_var == 0:
            legend_handles, legend_labels = handles, labels

    if suptitle is None:
        suptitle = rf'Reconstruction error {metric_disp} vs latent dimension'
    fig.suptitle(suptitle, fontsize=18)
    fig.legend(legend_handles, legend_labels,
               loc='lower center',
               ncol=len(model_labels),
               fontsize=16,
               bbox_to_anchor=(0.5, -0.04))

    plt.savefig(f'{where_to_save_data}/combined_plots/all_models_{metric}_{string_after_saving}.png',
                dpi=300, bbox_inches='tight')
    plt.close(fig)

    return 0


if __name__ == '__main__':
    base = '/scratch/ROM_datasets_ale/ASTEC/saved_logs/SBO/AE_NODE/Models'
    tail = 'Images/AutoEncoding/global_errors_reconstruction_fields/combined_plots/plotting_data.pkl'

    # model label (number of latent dimensions) -> plotting_data.pkl
    model_paths = {
        6:  f'{base}/NO_SMOOTHING_6_gamma_0.99_reset_lr_0.00045/{tail}',
        12: f'{base}/No_smoothing_latent_12/{tail}',
        24: f'{base}/No_smoothing_latent_24/{tail}',
        48: f'{base}/No_smoothing_latent_48/{tail}',
    }

    where_to_save = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'latent_dimension_comparison')
    combine_models_in_one_plot(model_paths, where_to_save,
                               string_after_saving='latent_comparison',
                               metric='RMSE_divided_by_std')
    print(f'Saved plots in {where_to_save}/combined_plots/')
