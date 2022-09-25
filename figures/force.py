import matplotlib.pyplot as plt
from matplotlib import patches as pat
from matplotlib import transforms
from matplotlib.collections import PatchCollection
import numpy as np
import pandas as pd
import re
from typing import List, Tuple, Any
from collections.abc import Iterable


def collapse_seq_filters(
    test_act_incl, test_act_skip, iM, sM, num_seq_filters, num_struct_filters
):
    incl_filter_names = [
        "incl_seq_" + "_".join(map(str, iM[key])) for key in iM.keys()
    ] + [f"incl_struct_{i}" for i in range(num_struct_filters)]
    skip_filter_names = [
        "skip_seq_" + "_".join(map(str, sM[key])) for key in sM.keys()
    ] + [f"skip_struct_{i}" for i in range(num_struct_filters)]

    nf_i = len(incl_filter_names) - num_struct_filters
    nf_s = len(skip_filter_names) - num_struct_filters

    test_act_incl_collapsed = np.zeros(
        (test_act_incl.shape[0], nf_i + num_struct_filters)
    )
    test_act_skip_collapsed = np.zeros(
        (test_act_skip.shape[0], nf_s + num_struct_filters)
    )

    for incl_idx, incl_key in enumerate(iM.keys()):
        test_act_incl_collapsed[:, incl_idx] = test_act_incl[:, iM[incl_key]].sum(
            axis=1
        )

    for skip_idx, skip_key in enumerate(sM.keys()):
        test_act_skip_collapsed[:, skip_idx] = test_act_skip[:, sM[skip_key]].sum(
            axis=1
        )

    test_act_incl_collapsed[:, nf_i:] = test_act_incl[:, num_seq_filters:]
    test_act_skip_collapsed[:, nf_s:] = test_act_skip[:, num_seq_filters:]

    return pd.DataFrame(
        test_act_incl_collapsed, columns=incl_filter_names
    ), pd.DataFrame(test_act_skip_collapsed, columns=skip_filter_names)


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


def plot_forces_sum_position(
    ax,
    test_act_incl,
    test_act_skip,
    seq_filters_grouping,
    link_function,
    num_seq_filters,
    num_struct_filters,
):
    iM = seq_filters_grouping["incl_membership_dict"]
    iR = seq_filters_grouping["incl_representative_dict"]
    sM = seq_filters_grouping["skip_membership_dict"]
    sR = seq_filters_grouping["skip_representative_dict"]

    iT, sT = collapse_seq_filters(
        test_act_incl, test_act_skip, iM, sM, num_seq_filters, num_struct_filters
    )

    incl_values = iT.sum(axis=0)
    skip_values = sT.sum(axis=0)

    height = 5
    gap = 0.0

    start_point = test_act_incl.sum() - test_act_skip.sum()
    current_x = start_point
    current_y = height

    up = True
    label_height_up = 5
    label_height_down = 2
    for label, val in skip_values.sort_values(ascending=False).iteritems():
        ax.add_patch(
            pat.Rectangle(
                (current_x, current_y),
                val,
                height,
                fc="#cc4444",
                ec="#000000",
                linewidth=2,
                angle=0,
            )
        )

        if np.abs(val) > 5:
            ax.annotate(
                f"{label}",
                xy=(current_x + val / 2, current_y + height),
                xytext=(
                    current_x + val / 2,
                    current_y + (label_height_up if up else label_height_down) * height,
                ),
                arrowprops=dict(
                    facecolor="black", shrink=0.05, width=1, headwidth=1, headlength=1
                ),
                verticalalignment="center",
                horizontalalignment="center",
                color="#000000",
            )
            up = not up

        current_x += val + gap

    current_y = height
    current_x = start_point
    up = True
    for label, val in incl_values.sort_values(ascending=False).iteritems():
        ax.add_patch(
            pat.Rectangle(
                (current_x - val, current_y),
                val,
                height,
                fc="#4444cc",
                ec="#000000",
                linewidth=2,
                angle=0,
            )
        )
        current_x -= val + gap
        if np.abs(val) > 5:
            ax.annotate(
                f"{label}",
                xy=(current_x + val / 2, current_y + height),
                xytext=(
                    current_x + val / 2,
                    current_y + (label_height_up if up else label_height_down) * height,
                ),
                arrowprops=dict(
                    facecolor="black", shrink=0.05, width=1, headwidth=1, headlength=1
                ),
                verticalalignment="center",
                horizontalalignment="center",
                color="#000000",
            )
            up = not up

    ax.axis("scaled")


def plot_net_forces_sum_position(
    ax,
    test_act_incl,
    test_act_skip,
    seq_filters_grouping,
    link_function,
    num_seq_filters,
    num_struct_filters,
    midpoint="auto",
):
    iM = seq_filters_grouping["incl_membership_dict"]
    sM = seq_filters_grouping["skip_membership_dict"]

    iT, sT = collapse_seq_filters(
        test_act_incl, test_act_skip, iM, sM, num_seq_filters, num_struct_filters
    )

    incl_values = iT.sum(axis=0)
    skip_values = sT.sum(axis=0)

    height = 2
    gap = 0.0

    if midpoint == "auto":
        start_point = get_link_midpoint(link_function)
    else:
        assert isinstance(midpoint, int) or isinstance(
            midpoint, float
        ), f"midpoint {midpoint} must be an int or float"
        start_point = midpoint
    current_x = start_point
    current_y = height

    up = True
    label_height_up = 5
    label_height_down = 3
    for label, val in incl_values.sort_values(ascending=False).iteritems():
        ax.add_patch(
            pat.Rectangle(
                (current_x, current_y),
                val,
                height,
                fc="#4444cc",
                ec="#000000",
                linewidth=2,
                angle=0,
            )
        )

        if np.abs(val) > 5:
            ax.annotate(
                f"{label}",
                xy=(current_x + val / 2, current_y + height),
                xytext=(
                    current_x + val / 2,
                    current_y + (label_height_up if up else label_height_down) * height,
                ),
                arrowprops=dict(
                    facecolor="black", shrink=0.05, width=1, headwidth=1, headlength=1
                ),
                verticalalignment="center",
                horizontalalignment="center",
                color="#000000",
            )
            up = not up

        current_x += val + gap

    current_y = 0
    # current_x = start_point
    up = True
    for label, val in skip_values.sort_values(ascending=False).iteritems():
        ax.add_patch(
            pat.Rectangle(
                (current_x - val, current_y),
                val,
                height,
                fc="#cc4444",
                ec="#000000",
                linewidth=2,
                angle=0,
            )
        )
        current_x -= val + gap
        if np.abs(val) > 5:
            ax.annotate(
                f"{label}",
                xy=(current_x + val / 2, current_y),
                xytext=(
                    current_x + val / 2,
                    current_y - (label_height_up if up else label_height_down) * height,
                ),
                arrowprops=dict(
                    facecolor="black", shrink=0.05, width=1, headwidth=1, headlength=1
                ),
                verticalalignment="center",
                horizontalalignment="center",
                color="#000000",
            )
            up = not up

    ax.axis("scaled")


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


def collapse_positions(A):
    out = dict()

    for row in A.index:
        for col in A.columns:
            out[f"{col}__pos_{row}"] = A.loc[row, col]
    return pd.Series(out)


def plot_forces_shap_style(
    ax,
    incl_values,
    skip_values,
    link_function,
    height=5,
    gap=0.0,
    chevron_angle=120,
    incl_color="#4444cc",
    skip_color_struct="#cc4444",
    skip_color="#cc4444",
    draw_prediction=True,
    hatch_structure=True,
    hatch_pattern="...",
    structure_label_regex=re.compile("struct"),
    fontsize=12,
    linewidth=2,
    pred_box_slope_height=2,
    pred_box_width=7,
    pred_box_height=4,
    pred_box_linewidth=2,
):
    start_point = incl_values.values.sum() - skip_values.values.sum()
    current_x = start_point
    for idx, (label, val) in enumerate(
        skip_values.sort_values(ascending=False).iteritems()
    ):
        hatch = (
            hatch_pattern
            if re.match(structure_label_regex, label) and hatch_structure
            else ""
        )
        ax.add_patch(
            pat.Polygon(
                get_chevron_coords(
                    current_x,
                    0,
                    height,
                    val,
                    chevron_angle,
                    draw_left=idx != 0,
                    draw_right=True,
                    direction="left",
                ),
                fc=skip_color_struct
                if re.match(structure_label_regex, label)
                else skip_color,
                ec="#000000",
                linewidth=linewidth,
                hatch=hatch,
            )
        )
        current_x += val + gap

    current_x = start_point
    for idx, (label, val) in enumerate(
        incl_values.sort_values(ascending=False).iteritems()
    ):
        hatch = (
            hatch_pattern
            if re.match(structure_label_regex, label) and hatch_structure
            else ""
        )
        ax.add_patch(
            pat.Polygon(
                get_chevron_coords(
                    current_x - val,
                    0,
                    height,
                    val,
                    chevron_angle,
                    draw_left=True,
                    draw_right=idx != 0,
                    direction="right",
                ),
                fc=incl_color,
                ec="#000000",
                linewidth=linewidth,
                hatch=hatch,
            )
        )
        current_x -= val + gap

    ax.axis("scaled")

    if draw_prediction:
        pred_height_gap = 0.5
        pred_box_height = pred_box_height
        pred_box_slope_height = pred_box_slope_height
        pred_box_width = pred_box_width
        ax.add_patch(
            pat.Polygon(
                [
                    (start_point, pred_height_gap + height),
                    (
                        start_point - pred_box_width / 2,
                        pred_height_gap + height + pred_box_slope_height,
                    ),
                    (
                        start_point - pred_box_width / 2,
                        pred_height_gap
                        + height
                        + pred_box_slope_height
                        + pred_box_height,
                    ),
                    (
                        start_point + pred_box_width / 2,
                        pred_height_gap
                        + height
                        + pred_box_slope_height
                        + pred_box_height,
                    ),
                    (
                        start_point + pred_box_width / 2,
                        pred_height_gap + height + pred_box_slope_height,
                    ),
                ],
                fc="#ffffff",
                ec="#000000",
                linewidth=pred_box_linewidth,
            )
        )
        pred = link_function(np.array([[start_point]])).numpy()[0, 0]
        ax.text(
            start_point,
            height + pred_height_gap + pred_box_slope_height + pred_box_height / 2,
            f"{pred:.2f}",
            ha="center",
            va="center",
            fontsize=fontsize,
        )


def get_chevron_coords(
    left_x,
    lower_y,
    height,
    width,
    chevron_angle=120,
    draw_left=True,
    draw_right=True,
    direction="right",
):
    upper_y = lower_y + height
    shifter = 1 if direction == "right" else -1
    inner_x = left_x + shifter * (
        height / 2 / np.tan(np.pi / 180 * (chevron_angle / 2))
    )
    outer_x = inner_x + width

    sector1 = [(left_x, lower_y)]
    sector2 = [(inner_x, lower_y + height / 2)] if draw_left else []
    sector3 = [(left_x, upper_y), (left_x + width, upper_y)]
    sector4 = [(outer_x, lower_y + height / 2)] if draw_right else []
    sector5 = [(left_x + width, lower_y)]

    points = np.array(sector1 + sector2 + sector3 + sector4 + sector5)

    x_offset = left_x - points.min(axis=0)[0]
    points[:, 0] += x_offset

    return points


def max_x_patch(p: pat.Patch) -> float:
    return p.get_path().vertices.max(axis=0)[0]


def get_bounds(patches: List[pat.Patch]) -> Tuple[Any, ...]:
    all_vertices: np.ndarray = np.concatenate(
        [p.get_path().vertices for p in patches], axis=0
    )

    return tuple(all_vertices.min(axis=0)) + tuple(all_vertices.max(axis=0))


def fill_pattern_in_bbox(
    left_x, lower_y, width, height, pattern_fn, pattern_width=None, pattern_spacing=0
):
    total_width = 0
    current_x = left_x
    patches = []

    while total_width <= width:
        chev = pattern_fn(current_x, lower_y)
        # add chevron to the list
        patches.append(chev)

        # set total_width
        total_width = max_x_patch(chev) - left_x

        # update current_x with pattern width and pattern spacing
        if pattern_width is None:
            _, a, _, b = get_bounds([chev])
            current_x += pattern_spacing + (b - a)
        else:
            current_x += pattern_spacing + pattern_width

    # return patches if list is empty
    if len(patches) == 0:
        return patches

    # trim excess
    if (max_x_patch(patches[-1]) - left_x) > width:
        patches = patches[:-1]

    return patches


def center_pattern_in_bbox(
    left_x, lower_y, width, height, pattern_fn, pattern_width=None, pattern_spacing=0
):
    patches = fill_pattern_in_bbox(
        left_x,
        lower_y,
        width,
        height,
        pattern_fn=pattern_fn,
        pattern_width=pattern_width,
        pattern_spacing=pattern_spacing,
    )
    if len(patches) == 0:
        return patches
    minx, miny, maxx, maxy = get_bounds(patches)

    # compute left x offset
    x_offset = (width - (maxx - minx)) / 2

    # compute lower y offset
    y_offset = (height - (maxy - miny)) / 2

    centered_patches = fill_pattern_in_bbox(
        left_x + x_offset,
        lower_y + y_offset,
        width,
        height,
        pattern_fn=pattern_fn,
        pattern_width=pattern_width,
        pattern_spacing=pattern_spacing,
    )

    return centered_patches


def draw_force(
    ax,
    fig,
    left_x,
    lower_y,
    width,
    height,
    labels=None,
    label_spacing=2,
    label_linewidth=2,
    label_fontsize=15,
    label_textcolor="k",
    alpha=1.0,
    draw_chevrons=True,
    chevron_width=2 / 3,
    chevron_spacing=2,
    chevron_height_fraction=0.6,
    chevron_angle=30,
    ec="#000000",
    linewidth=1.5,
    zorder=1,
    left_color="#e63946",
    right_color="#4895ef",
    direction="right",
    transform=None,
):
    assert direction in {"left", "right"}
    assert width > 0 and height > 0

    def right_chevron_fn(
        current_x,
        lower_y,
        chevron_height=chevron_height_fraction * height,
        chevron_width=chevron_width,
        chevron_angle=chevron_angle,
    ):
        return pat.Polygon(
            get_chevron_coords(
                current_x,
                lower_y,
                height=chevron_height,
                width=chevron_width,
                chevron_angle=chevron_angle,
            ),
            fc="k",
            ec="w",
            alpha=0.15,
            linewidth=0.0,
        )

    def left_chevron_fn(
        current_x,
        lower_y,
        chevron_height=chevron_height_fraction * height,
        chevron_width=chevron_width,
        chevron_angle=chevron_angle,
    ):
        return pat.Polygon(
            get_chevron_coords(
                current_x,
                lower_y,
                height=chevron_height,
                width=chevron_width,
                chevron_angle=chevron_angle,
                direction="left",
            ),
            fc="k",
            ec="w",
            alpha=0.15,
            linewidth=0.0,
        )

    # get pattern function based on direction
    pattern_fn = left_chevron_fn if direction == "left" else right_chevron_fn
    fc = left_color if direction == "left" else right_color

    # render rectangle
    rectangle = PatchCollection(
        [
            pat.Rectangle(
                xy=(left_x, lower_y),
                width=width,
                height=height,
                fc=fc,
                ec=ec,
                linewidth=linewidth,
                alpha=alpha,
            )
        ],
        zorder=zorder,
        match_original=True,
        transform=transform,
    )

    # render chevron
    chevrons = PatchCollection(
        center_pattern_in_bbox(
            left_x + 0.5,
            lower_y,
            width - 1,
            height,
            pattern_fn=pattern_fn,
            pattern_spacing=chevron_spacing,
        ),
        zorder=zorder + 1,
        match_original=True,
        transform=transform,
    )

    # render and draw label(s)
    ## record size of cluster label
    text_patch = ax.text(
        0,
        0,
        f"10",
        color="k",
        zorder=zorder + 2,
        # transform=ax.transData,
        transform=transform,
        weight="bold",
        fontweight="normal",
        ha="center",
        va="center",
        alpha=0,
        fontsize=label_fontsize,
        bbox={
            "boxstyle": "circle",
            "facecolor": fc,
            "linewidth": label_linewidth,
            "edgecolor": "k",
            "alpha": 0,
        },
    )
    # bbox = ax.transData.inverted().transform_bbox(
    #     text_patch.get_window_extent(renderer=fig.canvas.get_renderer())
    # )
    bbox = transform.inverted().transform_bbox(
        text_patch.get_window_extent(renderer=fig.canvas.get_renderer())
    )
    label_width = bbox.width

    ## draw labels
    if labels is not None:
        a = left_x
        b = left_x + width
        k = len(labels)
        c = (a + b) / 2 - (k * label_width + (k - 1) * label_spacing) / 2
        if c > a + 1: # + label_width:
            for i, label in enumerate(labels):
                x_pos = c + label_width / 2 + i * (label_spacing + label_width)
                t = ax.text(
                    x_pos,
                    lower_y + height / 2,
                    f"{label}",
                    color=label_textcolor,
                    zorder=zorder + 2,
                    # transform=ax.transData,
                    transform=transform,
                    weight="bold",
                    fontweight="normal",
                    ha="center",
                    va="center",
                    fontsize=label_fontsize,
                    bbox={
                        "boxstyle": "circle",
                        "facecolor": fc,
                        "linewidth": label_linewidth,
                        "edgecolor": "k",
                        "alpha": alpha
                    },
                )

    # draw force
    ax.add_collection(rectangle)
    # draw chevrons only if draw_chevrons is true
    if draw_chevrons and width > 1:
        ax.add_collection(chevrons)

    return ax


def draw_force_plot(
    ax,
    fig,
    incl_forces,
    skip_forces,
    incl_filter_lookup,
    skip_filter_lookup,
    highlight_forces=None,
    draw_delta=False,
    ymin=None,
    zorder=2,
    alpha=1.0,
    current_y=0,
    transform=None,
    bar_height=0.5,
    label_fontsize=18,
    label_linewidth=1,
    chevron_width=2 / 3,
    chevron_spacing=2,
    chevron_height_fraction=0.6,
    chevron_angle=30,
    left_color="#ff6666",
    light_left_color="#F6C3C2",
    right_color="#669aff",
    light_right_color="#C5D6FB",
    ec="#6b6b6b"
):
    """1. compute fill area bounds
    2. compute chevron width, spacing between chevrons
    3. compute max number of chevrons (with spacing) you can fit in the area bounds
    4. create a compound object of chevrons and spacing
    5. center compound object within area bounds
    6. plot compound object"""

    # Draw forces
    ## inclusion
    current_x = 0
    for f_i_name, f_i in incl_forces.items():
        if highlight_forces is not None and isinstance(highlight_forces, Iterable):
            if f_i_name in highlight_forces:
                rc = right_color
            else:
                rc = light_right_color
        else:
            rc = right_color
        
        labels = [incl_filter_lookup[e] for e in f_i_name.split("___")]
        draw_force(
            ax,
            fig,
            current_x,
            current_y,
            f_i,
            bar_height,
            labels=labels,
            label_fontsize=label_fontsize,
            label_linewidth=label_linewidth,
            alpha=alpha,
            zorder=zorder,
            direction="right",
            chevron_width=chevron_width,
            chevron_spacing=chevron_spacing,
            chevron_height_fraction=chevron_height_fraction,
            chevron_angle=chevron_angle,
            transform=transform,
            left_color=left_color,
            right_color=rc,
            ec=ec
        )
        current_x += f_i

    ## skipping
    for f_s_name, f_s in skip_forces.items():
        if highlight_forces is not None and isinstance(highlight_forces, Iterable):
            if f_s_name in highlight_forces:
                lc = left_color
            else:
                lc = light_left_color
        else:
            lc = left_color

        labels = [skip_filter_lookup[e] for e in f_s_name.split("___")]
        draw_force(
            ax,
            fig,
            current_x - f_s,
            current_y - bar_height,
            f_s,
            bar_height,
            labels=labels,
            label_fontsize=label_fontsize,
            label_linewidth=label_linewidth,
            alpha=alpha,
            zorder=zorder,
            direction="left",
            chevron_width=chevron_width,
            chevron_spacing=chevron_spacing,
            chevron_height_fraction=chevron_height_fraction,
            chevron_angle=chevron_angle,
            transform=transform,
            left_color=lc,
            right_color=right_color,
            ec=ec
        )
        current_x -= f_s

    # draw delta forces
    if draw_delta:
        assert ymin is not None, "Must provide ymin if draw_delta=True"
        # Draw delta forces
        delta_force = incl_forces.sum() - skip_forces.sum()
        ax.plot(
            [delta_force, delta_force],
            [ymin, current_y - bar_height],
            color="r",
            linestyle=":",
            linewidth=2,
            transform=transform,
        )


def draw_axes(
    ax,
    xmin,
    xmax,
    ymin,
    ymax,
    link_function,
    transform=None,
    x_tick_interval=10,
    zorder=0,
):
    ax.spines["left"].set_visible(False)
    ax.set_yticks([])
    ax.spines["bottom"].set_visible(False)
    ax.set_xticks([])

    # plot central line
    # ax.plot([0, 0], [ymin, ymax], color="k", linewidth=3, transform=transform, zorder=zorder)
    link_midpoint = get_link_midpoint(link_function)
    ax.plot([link_midpoint, link_midpoint], [ymin, ymax], color="k", linewidth=3, transform=transform, zorder=zorder)

    # draw x-axis line
    ax.plot([xmin, xmax], [ymin, ymin], color="k", linewidth=3, transform=transform, zorder=zorder)
    for x in np.arange(xmin, xmax, x_tick_interval):
        # delta force ticks
        ax.plot([x, x], [ymin, ymin + 0.1], color="k", transform=transform, zorder=zorder)
        # delta force grid lines
        ax.plot(
            [x, x], [ymin, ymax], color="k", alpha=0.1, zorder=zorder, transform=transform
        )
        # delta force minor grid lines
        ax.plot(
            [x + x_tick_interval / 2, x + x_tick_interval / 2],
            [ymin, ymax],
            color="k",
            alpha=0.1,
            zorder=zorder,
            linewidth=1,
            transform=transform,
        )
        # if x != 0:
        # delta force annotations
        ax.text(
            x,
            ymin + 0.25,
            f"{x}",
            color="k",
            zorder=zorder + 1,
            # transform=ax.transData,
            transform=transform,
            weight="bold",
            fontweight="normal",
            ha="center",
            va="center",
            fontsize=15,
        )

    # draw PSI axis
    ys = sorted(list(np.linspace(0.01, 0.988, 11)) + [0.93, 0.95, 0.97, 0.98])
    xs = [get_link_midpoint(link_function, midpoint=m) for m in ys]

    for x, y in zip(xs, ys):
        # psi ticks
        ax.plot([x, x], [ymin, ymin - 0.1], color="k", transform=transform)
        # psi tick annotations
        if y < 0.9:
            ax.text(
                x,
                ymin - 0.25,
                f"{np.round(y, decimals=1)}",
                color="k",
                zorder=zorder + 1,
                transform=transform,
                weight="bold",
                fontweight="normal",
                ha="center",
                va="center",
                fontsize=15,
            )
        else:
            ax.text(
                x,
                ymin - 0.25,
                f"{np.round(y, decimals=2)}",
                color="k",
                zorder=zorder + 1,
                transform=transform,
                weight="bold",
                fontweight="normal",
                ha="center",
                va="center",
                fontsize=15,
            )

    # draw axis labels
    ax.text(
        xmin - 2,
        ymin - 0.25,
        f"PSI",
        color="k",
        zorder=zorder + 1,
        transform=transform,
        weight="bold",
        fontweight="bold",
        ha="right",
        va="center",
        fontsize=15,
    )
    ax.text(
        xmin - 2,
        ymin + 0.25,
        f"$\Delta$ Force",
        color="k",
        zorder=zorder + 1,
        transform=transform,
        weight="bold",
        fontweight="bold",
        ha="right",
        va="center",
        fontsize=15,
    )


def get_fixed_height_transform(ax, fig):
    return transforms.blended_transform_factory(
        ax.transData, fig.dpi_scale_trans
    ) + transforms.ScaledTranslation(
        0,
        0,
        scale_trans=transforms.blended_transform_factory(
            fig.dpi_scale_trans, ax.transData
        ),
    )

def merge_small_forces(forces, threshold=1):
    if (forces < threshold).sum() == 0:
        return forces
    merged_key = '___'.join(forces[forces < threshold].index)
    merged_force = forces[forces < threshold].sum()
    
    out = forces[forces >= threshold].copy()
    out[merged_key] = merged_force
    return out

def draw_left_justified_force_plot(
    axarr,
    fig,
    incl_forces,
    skip_forces,
    incl_filter_lookup,
    skip_filter_lookup,
    highlight_forces=None,
    draw_delta=False,
    ymin=None,
    zorder=2,
    alpha=1.0,
    current_y=0,
    transform=None,
    bar_height=0.5,
    label_fontsize=18,
    label_linewidth=1,
    draw_chevrons=False,
    chevron_width=2 / 3,
    chevron_spacing=2,
    chevron_height_fraction=0.6,
    chevron_angle=30,
    left_color="#ff6666",
    light_left_color="#F6C3C2",
    right_color="#669aff",
    light_right_color="#C5D6FB",
    ec="#6b6b6b"
):
    pass