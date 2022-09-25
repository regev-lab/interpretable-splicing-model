import logomaker
import pandas as pd
import numpy as np
from tqdm.auto import tqdm


def plot_logo(
    df,
    threshold,
    ax,
    color_map={"A": "#00dc33", "C": "#1c1cd5", "G": "#f2a93c", "U": "#ff1525"},
    nts=["A", "C", "G", "U"],
):

    current_df = df.query(f"activation > {threshold}")
    if len(current_df) == 0:
        return
    data = compute_heights(compute_freqs(list(current_df.input), nts=nts), nts=nts)

    logomaker.Logo(data, color_scheme=color_map, vpad=0.1, width=0.8, ax=ax)


def compute_freqs(kmer_list, nts=["A", "C", "G", "U"]):
    n_pos = len(kmer_list[0])
    out = pd.DataFrame(np.zeros((n_pos, len(nts))), columns=nts)

    iterator = (
        tqdm(kmer_list, desc="Iterating over k-mer list", leave=False)
        if len(kmer_list) > 1000
        else kmer_list
    )
    for kmer in iterator:
        for idx, nt in enumerate(kmer):
            out[nt][idx] += 1

    return out / out.sum(axis=1).values.reshape(-1, 1)


EPSILON = 0.00000001  # to avoid log(0)


def compute_info(freqs):
    N = freqs.shape[1]

    I = np.log(N) / np.log(2) + (freqs * np.log(freqs + EPSILON) / np.log(2)).sum(
        axis=1
    )
    return I


def compute_heights(freqs, nts=["A", "C", "G", "U"]):
    I = compute_info(freqs).values.reshape(-1, 1)

    df = freqs * I
    df.columns = nts
    return df


def sequence_logo_heights(df, nts=["A", "C", "G", "U"]):
    return compute_heights(compute_freqs(list(df.input), nts=nts), nts=nts)


def draw_floating_logo(
    heights_df,
    rectangle_width,
    rectangle_height,
    rectangle_x,
    rectangle_y,
    ax,
    color_lookup={"A": "#00dc33", "C": "#1c1cd5", "G": "#f2a93c", "U": "#ff1525"},
    zorder=1,
):
    char_width = rectangle_width / heights_df.shape[0]
    heights_scale = rectangle_height / heights_df.sum(axis=1).max()

    for p in heights_df.index:
        char_heights_p = heights_df.iloc[p].sort_values(ascending=True)
        cum_height = 0

        x_loc = rectangle_x
        for c, v in char_heights_p.iteritems():
            glyph = logomaker.Glyph(
                char_width * (p + 1 / 2) + rectangle_x,
                c,
                ax=ax,
                floor=rectangle_y + cum_height,
                ceiling=rectangle_y + cum_height + heights_scale * v,
                width=char_width,
                color=color_lookup[c],
                flip=False,
                zorder=zorder,
                font_name="sans",
                alpha=1.0,
                vpad=0.0,
            )
            cum_height += heights_scale * v


def compute_EDLogo_scores(
    kmer_list, nts=["A", "C", "G", "U"], normed=True, max_kind="per_nt"
):
    assert max_kind in {"per_nt", "global"}
    P = compute_freqs(kmer_list, nts=nts)
    baseline_freqs = np.ones(4) / 4

    X = len(kmer_list) * P
    Y = len(kmer_list) * baseline_freqs

    alpha = np.log(X / Y).copy() / np.log(2)
    alpha_corner_case = np.log((X + 0.5) / (Y + 0.5)).copy() / np.log(2)
    alpha_low = alpha_corner_case.copy() - 0.5
    alpha_high = alpha_corner_case.copy() + 0.5

    alpha.values[(X == 0).values] = alpha_low.values[(X == 0).values]
    alpha.values[(X == len(kmer_list))] = alpha_high.values[(X == len(kmer_list))]

    R = alpha.subtract(alpha.median(axis=1), axis=0)

    if normed:
        if max_kind == "per_nt":
            return R / np.maximum(1, np.abs(R).max())
        elif max_kind == "global":
            return R / np.abs(R).values.max()
    return R


def plot_EDLogo(
    df,
    threshold,
    ax,
    color_map={"A": "#00dc33", "C": "#1c1cd5", "G": "#f2a93c", "U": "#ff1525"},
    nts=["A", "C", "G", "U"],
    normed=True,
    max_kind="global",
):

    current_df = df.query(f"activation > {threshold}")
    if len(current_df) == 0:
        return
    scores = compute_EDLogo_scores(
        current_df["input"].values, nts=nts, normed=normed, max_kind=max_kind
    )

    logomaker.Logo(
        scores,
        color_scheme=color_map,
        # vpad=0.1,
        # width=0.8,
        ax=ax,
        font_name="Arial Rounded MT Bold",
    )
