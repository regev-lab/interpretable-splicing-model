import matplotlib.pyplot as plt
from joblib import load
import scipy.cluster.hierarchy as sch
import pandas as pd

import figutils
import sequence_logo
from quad_model import *



def get_link_midpoint(
    link_function, midpoint=0.5, epsilon=1e-5, lb=-100, ub=100, max_iters=50
):
    """Assumes monotonicity and smoothness of link function"""

    iters = 0
    while iters < max_iters:
        xx = np.linspace(lb, ub, 1000)
        yy = link_function(xx[:, None]).numpy().flatten()

        if min(np.abs(yy - midpoint)) < epsilon:
            return xx[np.abs(yy - midpoint) < epsilon][0]
        lb_idx = np.where((yy - midpoint) < 0)[0][-1]
        ub_idx = np.where((yy - midpoint) > 0)[0][0]

        lb = xx[lb_idx]
        ub = xx[ub_idx]

        iters += 1
    raise RuntimeError(f"Max iterations ({max_iters}) reached without solution...")



def collapse_filters(
    test_act_incl,
    test_act_skip,
    iM,
    sM,
    iM_struct,
    sM_struct,
    num_seq_filters,
):
    incl_seq_filter_names = [
        "incl_seq_" + "_".join(map(str, iM[key])) for key in iM.keys()
    ]
    incl_struct_filter_names = [
        "incl_struct_" + "_".join(map(str, iM_struct[key])) for key in iM_struct.keys()
    ]
    skip_seq_filter_names = [
        "skip_seq_" + "_".join(map(str, sM[key])) for key in sM.keys()
    ]
    skip_struct_filter_names = [
        "skip_struct_" + "_".join(map(str, sM_struct[key])) for key in sM_struct.keys()
    ]

    nf_i = len(incl_seq_filter_names) + len(incl_struct_filter_names)
    nf_s = len(skip_seq_filter_names) + len(skip_struct_filter_names)

    test_act_incl_collapsed = np.zeros((test_act_incl.shape[0], nf_i))
    test_act_skip_collapsed = np.zeros((test_act_skip.shape[0], nf_s))

    ctr = 0
    for incl_idx, incl_key in enumerate(iM.keys()):
        test_act_incl_collapsed[:, incl_idx] = test_act_incl[:, iM[incl_key]].sum(
            axis=1
        )
        ctr += 1

    for incl_idx, incl_key in enumerate(iM_struct.keys()):
        test_act_incl_collapsed[:, ctr + incl_idx] = test_act_incl[
            :, num_seq_filters + iM_struct[incl_key]
        ].sum(axis=1)

    ctr = 0
    for skip_idx, skip_key in enumerate(sM.keys()):
        test_act_skip_collapsed[:, skip_idx] = test_act_skip[:, sM[skip_key]].sum(
            axis=1
        )
        ctr += 1

    for skip_idx, skip_key in enumerate(sM_struct.keys()):
        test_act_skip_collapsed[:, ctr + skip_idx] = test_act_skip[
            :, num_seq_filters + sM_struct[skip_key]
        ].sum(axis=1)

    return pd.DataFrame(
        test_act_incl_collapsed,
        columns=incl_seq_filter_names + incl_struct_filter_names,
    ), pd.DataFrame(
        test_act_skip_collapsed,
        columns=skip_seq_filter_names + skip_struct_filter_names,
    )
    
    
def create_force_data(
    test_act_incl,
    test_act_skip,
    seq_filters_grouping,
    struct_filters_grouping,
    num_seq_filters,
    sum_positions=True,
    link_midpoint=None
):
    iM = seq_filters_grouping["incl_membership_dict"]
    sM = seq_filters_grouping["skip_membership_dict"]

    iM_struct = struct_filters_grouping["incl_membership_dict"]
    sM_struct = struct_filters_grouping["skip_membership_dict"]

    iT, sT = collapse_filters(
        test_act_incl,
        test_act_skip,
        iM,
        sM,
        iM_struct,
        sM_struct,
        num_seq_filters,
    )

    if sum_positions:
        A, B = iT.sum(axis=0), sT.sum(axis=0)

        if link_midpoint is not None:
            if link_midpoint < 0:
                A['incl_bias'] = np.abs(link_midpoint)
            else:
                B['skip_bias'] = np.abs(link_midpoint)

        return A, B

    A, B = collapse_positions(iT), collapse_positions(sT)

    if link_midpoint is not None:
        if link_midpoint < 0:
            A['incl_bias'] = np.abs(link_midpoint)
        else:
            B['skip_bias'] = np.abs(link_midpoint)
    

    return A, B




def merge_small_forces(forces, threshold=1):
    if (forces < threshold).sum() == 0:
        return forces
    merged_key = '___'.join(forces[forces < threshold].index)
    merged_force = forces[forces < threshold].sum()
    
    out = forces[forces >= threshold].copy()
    out[merged_key] = merged_force
    return out






####################
## INITIALIZE
####################

plt.style.use('clean.mplstyle')

# Load data and models

DATA_DIR = "../../sequence_analysis_utils/force_plots_es7/data/"
xTr = load(DATA_DIR+f'xTr_ES7_HeLa_ABC.pkl.gz')
yTr = load(DATA_DIR+f'yTr_ES7_HeLa_ABC.pkl.gz')
xTe = load(DATA_DIR+f'xTe_ES7_HeLa_ABC.pkl.gz')
yTe = load(DATA_DIR+f'yTe_ES7_HeLa_ABC.pkl.gz')

model_fname = f'custom_adjacency_regularizer_20210731_124_step3.h5'
model = tf.keras.models.load_model(model_fname)

num_seq_filters = model.get_layer('qc_incl').kernel.shape[2]
num_struct_filters = model.get_layer('c_incl_struct').kernel.shape[2]

position_bias_size = model.get_layer('position_bias_incl').kernel.shape[0]
struct_filter_width = model.get_layer("c_incl_struct").kernel.shape[0]
input_length = model.input[0].shape[1]

# Group sequence filters

def get_membership_dict(ind):
    out = {}
    
    for i, group_i in enumerate(ind):
        if group_i not in out:
            out[group_i] = []
        out[group_i].append(i)
    return out

def get_fig_num_rows_cols(membership_dict):
    fig_rows = max([len(e) for e in list(membership_dict.values())])
    fig_cols = len(membership_dict.keys())
    return fig_rows, fig_cols 

structure_out_model = Model(inputs=model.inputs, outputs=[
    model.get_layer('activation_2').output,
    model.get_layer('activation_3').output
])

incl_act, skip_act = structure_out_model.predict(xTr, verbose=1, batch_size=1024)
incl_act_seq = incl_act[:, :, :num_seq_filters]
skip_act_seq = skip_act[:, :, :num_seq_filters]

incl_inds = sch.fcluster(sch.linkage(incl_act_seq.sum(axis=1).T,
                               metric='correlation',
                               method='complete'),
                   t=0.9,
                   criterion='distance')

skip_inds = sch.fcluster(sch.linkage(skip_act_seq.sum(axis=1).T,
                               metric='correlation',
                               method='complete'),
                   t=0.9,
                   criterion='distance')


incl_membership_dict = get_membership_dict(incl_inds)
skip_membership_dict = get_membership_dict(skip_inds)

def get_representative_dict(membership_dict, activations):
    scores = activations.sum(axis=1).mean(axis=0)
    out = dict()
    
    for cluster_id in membership_dict:
        filter_ids = membership_dict[cluster_id]
        
        top_filter = np.argmax(scores[filter_ids])
        out[cluster_id] = filter_ids[top_filter]
    return out


incl_representative_dict = get_representative_dict(incl_membership_dict, incl_act_seq)
skip_representative_dict = get_representative_dict(skip_membership_dict, skip_act_seq)

# Manually modify representatives
skip_representative_dict[4] = 10
incl_representative_dict[3] = 5

# Group sequence filters
seq_filters_grouping = dict(incl_membership_dict=incl_membership_dict,
     skip_membership_dict=skip_membership_dict,
     incl_representative_dict=incl_representative_dict,
     skip_representative_dict=skip_representative_dict)

# Group structure filters

incl_membership_struct_dict = {1: np.array([0, 1, 2, 3, 4, 5, 6, 7])}
skip_membership_struct_dict = {1: np.array([1]), 2: np.array(
    [0, 2, 3]), 3: np.array([5, 6, 7]), 4: np.array([4])}
incl_representative_struct_dict = {1: 0}
skip_representative_struct_dict = {1: 1, 2: 0, 3: 5}

struct_filters_grouping = dict(incl_membership_dict=incl_membership_struct_dict,
     skip_membership_dict=skip_membership_struct_dict,
     incl_representative_dict=incl_representative_struct_dict,
     skip_representative_dict=skip_representative_struct_dict)

incl_color = '#669aff'
skip_color = '#ff6666'
light_incl_color = '#C5D6FB'
light_skip_color = '#F6C3C2'

incl_membership_scores = {
    key: np.quantile(incl_act_seq[:, :, incl_membership_dict[key]].sum(axis=(1, 2)), 0.9)
    for key in incl_membership_dict.keys()
}
skip_membership_scores = {
    key: np.quantile(skip_act_seq[:, :, skip_membership_dict[key]].sum(axis=(1, 2)), 0.9)
    for key in skip_membership_dict.keys()
}

incl_plot_order = sorted(incl_membership_scores, key=lambda x: -incl_membership_scores[x])
skip_plot_order = sorted(skip_membership_scores, key=lambda x: -skip_membership_scores[x])

top_k = 4
num_extra_filters = 2

incl_filter_lookup = {}
for key, value in seq_filters_grouping['incl_membership_dict'].items():
    incl_filter_lookup[f'incl_seq_' + '_'.join(map(str, sorted(value)))] = incl_plot_order.index(key) + 1
for key, value in struct_filters_grouping['incl_membership_dict'].items():
    incl_filter_lookup[f'incl_struct_' + '_'.join(map(str, sorted(value)))] = incl_plot_order.index(key) + 1
    
skip_filter_lookup = {}
for key, value in seq_filters_grouping['skip_membership_dict'].items():
    skip_filter_lookup[f'skip_seq_' + '_'.join(map(str, sorted(value)))] = skip_plot_order.index(key) + 1
for key, value in struct_filters_grouping['skip_membership_dict'].items():
    skip_filter_lookup[f'skip_struct_' + '_'.join(map(str, sorted(value)))] = skip_plot_order.index(key) + 1
    
# Manually modify symbols for G-poor, structure, inclusion bias, and others
skip_filter_lookup['skip_struct_0_2_3'] = 'S'
skip_filter_lookup['skip_struct_1'] = 'P'
skip_filter_lookup['skip_struct_5_6_7'] = '.'
skip_filter_lookup['skip_struct_4'] = ' '

incl_filter_lookup['incl_bias'] = 'B'
skip_filter_lookup['skip_bias'] = 'B'

# Manually relabel skipping filters
for idx in range(len(skip_plot_order)):
    skip_key = skip_plot_order[idx]
    skip_filter_num = skip_representative_dict[skip_key]
    skip_filter_group = [b for a,b in list(seq_filters_grouping['skip_membership_dict'].items()) if skip_filter_num in b][0]
    key = f'skip_seq_' + '_'.join(map(str, sorted(skip_filter_group)))
    
    skip_filter_lookup[key] = idx + 1 + top_k + num_extra_filters
    
    
    
    
def get_model_midpoint(model, midpoint=0.5):    
    """ Compute the midpoint using the model's link function. This is the negation of the basal strength. I.e., positive value corresponds to a skipping basal strength. 
    """
    link_input = Input(shape=(1,))
    w = model.get_layer('energy_seq_struct').w.numpy()
    b = model.get_layer('energy_seq_struct').b.numpy()
    link_output = model.get_layer('output_activation')(model.get_layer('gen_func')(w*link_input + b))
    link_function = Model(inputs=link_input, outputs=link_output)
    return get_link_midpoint(link_function, midpoint)

# The main function for drawing a force plot
def draw_force_plot(sequences, # sequences should be flanked by 7 intronic nucleotide on each side; for our dataset, that gives 76+7+7=90
                    annotations,
                    highlight_forces = [],
                    incl_color=incl_color, skip_color=skip_color, light_incl_color=light_incl_color, light_skip_color=light_skip_color,
                    figsize=(40/2, 10/2), force_y_range=(0, 90), delta_force_y_range=(-15, 25),
                    ys=[0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                        0.7, 0.8, 0.9, 0.95, 0.98, 0.986],
                    draw_numbers=False,
                    numbers_min_bar_height=4,
                    label_rotation=0, label_alignment='center', delta_bar_width=2, 
                    width_ratios=[2, 2], # for horizontal plots
                    height_ratios=[1, 1], # for vertical plots
                    vertical=False, sharex=False, axislinewidth=1,
                    vertical_adjustement=0,
                    parent_figure = None, # if force plot should be a subfigure, pass the subfigure here
                    custom_model = model # in case exons are non-standard size or a basal shift is required, a custom model can be provided here; layers should be following the same names as the main model
                    ):
    #assert data_incl_act.shape == data_skip_act.shape
    assert(len(sequences) == len(annotations))
        
    link_midpoint = get_model_midpoint(custom_model)
    

    # compute activations for given sequences
    activations_model = Model(inputs=custom_model.inputs, outputs=[
        custom_model.get_layer('activation_2').output,
        custom_model.get_layer('activation_3').output
    ])
    data_incl_act, data_skip_act = activations_model.predict(figutils.create_input_data(sequences))
    
    N = data_incl_act.shape[0] 
    
    if parent_figure:
        if vertical:
            axarr = parent_figure.subplots(2, 1, gridspec_kw={'height_ratios': height_ratios}, sharex=sharex)
        else:
            axarr = parent_figure.subplots(1, 2, gridspec_kw={'width_ratios': width_ratios})
            
    else:
        if vertical:
            fig, axarr = plt.subplots(2, 1, figsize=figsize, dpi=150, gridspec_kw={
                'height_ratios': height_ratios}, sharex=sharex)
        else:
            fig, axarr = plt.subplots(1, 2, figsize=figsize, dpi=150, gridspec_kw={
                                      'width_ratios': width_ratios})

    for idx in range(N):
        incl_act = data_incl_act[idx]
        skip_act = data_skip_act[idx]

        incl_forces, skip_forces = create_force_data(incl_act, skip_act,
                                                           seq_filters_grouping, struct_filters_grouping,
                                                           num_seq_filters, sum_positions=True, link_midpoint=link_midpoint)

        incl_forces = merge_small_forces(incl_forces, threshold=0)
        incl_forces = incl_forces.sort_values(ascending=False, key=lambda x: (incl_forces.keys()=="incl_bias")*1000+incl_forces.values)
        skip_forces = merge_small_forces(skip_forces, threshold=0)
        skip_forces = skip_forces.sort_values(ascending=False, key=lambda x: (skip_forces.keys()=="skip_bias")*1000+skip_forces.values)
        
        total_i = 0
        total_s = 0
        for (f_i_name, f_i), (f_s_name, f_s) in zip(incl_forces.items(), skip_forces.items()):
            if f_i_name in highlight_forces:
                f_i_color = incl_color
            else:
                f_i_color = light_incl_color
            if f_s_name in highlight_forces:
                f_s_color = skip_color
            else:
                f_s_color = light_skip_color

            axarr[0].bar([3*idx], [f_i], bottom=[total_i], color=f_i_color,
                         linewidth=1, edgecolor='#6b6b6b', width=1, zorder=2)
            total_i += f_i
            axarr[0].bar([3*idx+1], [f_s], bottom=[total_s],
                         color=f_s_color, linewidth=1, edgecolor='#6b6b6b', width=1, zorder=2)
            total_s += f_s

            # draw numbers
            if draw_numbers:
                labels_i = [incl_filter_lookup[e]
                            for e in f_i_name.split("___")]
                labels_s = [skip_filter_lookup[e]
                            for e in f_s_name.split("___")]

                if f_i > numbers_min_bar_height:
                    axarr[0].text(3*idx, total_i - f_i/2 - vertical_adjustement,
                                  labels_i[0], ha='center', va='center')
                if f_s > numbers_min_bar_height:
                    axarr[0].text(3*idx + 1, total_s - f_s/2 - vertical_adjustement,
                                  labels_s[0], ha='center', va='center')

        delta_force = incl_forces.sum() - skip_forces.sum()
        axarr[1].bar([3*idx + 0.5], [delta_force], color='#dad7cd' if delta_force <
                     0 else '#dad7cd', linewidth=1, edgecolor='#6b6b6b', width=delta_bar_width, zorder=2)

    axarr[0].set_ylim(*force_y_range)
    axarr[0].set_yticks(20 * np.unique(np.arange(force_y_range[1] +
                        1 if force_y_range[1] % 20 == 0 else force_y_range[1]) // 20))

    axarr[1].set_ylim(*delta_force_y_range)

    axarr[0].grid(axis='y', which='both', zorder=0)
    axarr[1].grid(axis='y', which='both', zorder=0)
    xmin, xmax = axarr[1].get_xlim()
    axarr[1].hlines(0, xmin, xmax, color='k', zorder=3, linewidth=1)
    axarr[1].set_xlim(xmin, xmax)

    axarr[0].set_ylabel('Strength (a.u.)', fontsize=14)
    axarr[1].set_ylabel('$\Delta$ Strength (a.u.)', fontsize=14)
    axarr[0].spines['right'].set_visible(True)

    axarr[0].set_xticks([3*i + 0.5 for i, _ in enumerate(annotations)])
    axarr[0].set_xticklabels([annotations[i] for i, _ in enumerate(
        annotations)], rotation=label_rotation, ha=label_alignment)

    axarr[1].set_xticks([3*i + 0.5 for i, _ in enumerate(annotations)])
    axarr[1].set_xticklabels([annotations[i] for i, _ in enumerate(
        annotations)], rotation=label_rotation, ha=label_alignment)

    xs = [get_model_midpoint(custom_model, midpoint=m)-link_midpoint for m in ys]
    ys = [np.round(e, 2) for e in ys]

    ax2 = axarr[1].twinx()
    axarr[1].spines['right'].set_visible(True)

    ax2.set_yticks(xs)
    ax2.set_yticklabels(ys)
    ax2.set_ylim(*delta_force_y_range)
    ax2.set_ylabel('Predicted PSI', fontsize=14)
    
    for ax in axarr:
        ax.tick_params(axis='both', labelsize=12)
    axarr[0].tick_params(axis='x', length=0)

    if (parent_figure != None):
        parent_figure.align_ylabels(axarr)
    else:
        fig.align_ylabels(axarr)
        return fig
