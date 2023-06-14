import os
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import tensorflow as tf
from matplotlib import cm
from tqdm.auto import tqdm
import pandas as pd
import RNAutils


def ei_vec(i, len):  # give a one-hot encoding
    result = [0 for i in range(len)]
    result[i] = 1
    return result


def str_to_vector(str, template):
    #   return [ei_vec(template.index(nt),len(template)) for nt in str]
    mapping = dict(zip(template, range(len(template))))
    seq = [mapping[i] for i in str]
    return np.eye(len(template))[seq]


def nts_to_vector(nts, rna=False):
    if rna:
        return str_to_vector(nts, "ACGU")
    return str_to_vector(nts, "ACGT")


def human_format(num):
    num = float("{:.3g}".format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return "{}{}".format(
        "{:f}".format(num).rstrip("0").rstrip("."), [
            "", "K", "M", "B", "T"][magnitude]
    )


def hamming(s1, s2):
    """Calculate the Hamming distance between two bit strings"""
    assert len(s1) == len(s2)
    if s1 == s2:
        return 0  # optimization in case strings are equal
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def revcomp(str):
    complement = {
        "A": "T",
        "C": "G",
        "G": "C",
        "T": "A",
        "a": "t",
        "c": "g",
        "g": "c",
        "t": "a",
    }
    return "".join(complement.get(base, base) for base in reversed(str))


def get_qualities(str):
    return [ord(str[i]) - 33 for i in range(len(str))]


def contains_Esp3I_site(str):
    return ("CGTCTC" in str) or ("GAGACG" in str)


# Reads a line from file, and updates tqdm
def tqdm_readline(file, pbar):
    line = file.readline()
    pbar.update(len(line))
    return line


# Reads both FASTQ file, and applies callback on each read
# Returns number of reads
def process_paired_fastq_file(filename1, filename2, callback):
    file_size = os.path.getsize(filename1)
    with tqdm(total=file_size) as pbar:

        file1 = open(filename1, "r")
        file2 = open(filename2, "r")

        total_reads = 0

        while True:
            temp = tqdm_readline(file1, pbar).strip()  # header
            if temp == "":
                break  # end of file
            read_1 = tqdm_readline(file1, pbar).strip()
            tqdm_readline(file1, pbar)  # header
            read_1_q = tqdm_readline(file1, pbar).strip()

            file2.readline()  # header
            read_2 = file2.readline().strip()
            file2.readline()  # header
            read_2_q = file2.readline().strip()

            callback(read_1, read_2, read_1_q, read_2_q)

            total_reads += 1

    return total_reads


def scatter_with_kde(x, y, ax, alpha=0.1):
    assert len(x) == len(y)
    if len(x) < 10000:
        kde = scipy.stats.gaussian_kde([x, y])
    else:  # subsample
        indices_to_pick = np.random.choice(range(len(x)), size=10000)
        kde = scipy.stats.gaussian_kde(
            [x[indices_to_pick], y[indices_to_pick]])
    zz = kde([x, y])
    cc = cm.jet((zz - zz.min()) / (zz.max() - zz.min()))
    ax.scatter(x, y, alpha=alpha, s=1, facecolors=cc)


PRE_SEQUENCE = "TCTGCCTATGTCTTTCTCTGCCATCCAGGTT"
POST_SEQUENCE = "CAGGTCTGACTATGGGACCCTTGATGTTTT"


def add_flanking(nts, flanking_len):
    return PRE_SEQUENCE[-flanking_len:] + nts + POST_SEQUENCE[:flanking_len]


BARCODE_PRE_SEQUENCE = "CACAAGTATCACTAAGCTCGCTCTAGA"
BARCODE_POST_SEQUENCE = "ATAGGGCCCGTTTAAACCCGCTGAT"


def add_barcode_flanking(nts, flanking_len):
    return (
        BARCODE_PRE_SEQUENCE[-flanking_len:]
        + nts
        + BARCODE_POST_SEQUENCE[:flanking_len]
    )


def find_parentheses(s):
    """Find and return the location of the matching parentheses pairs in s.

    Given a string, s, return a dictionary of start: end pairs giving the
    indexes of the matching parentheses in s. Suitable exceptions are
    raised if s contains unbalanced parentheses.

    """

    # The indexes of the open parentheses are stored in a stack, implemented
    # as a list

    stack = []
    parentheses_locs = {}
    for i, c in enumerate(s):
        if c == "(":
            stack.append(i)
        elif c == ")":
            try:
                parentheses_locs[stack.pop()] = i
            except IndexError:
                raise IndexError(
                    "Too many close parentheses at index {}".format(i))
    if stack:
        raise IndexError(
            "No matching close parenthesis to open parenthesis "
            "at index {}".format(stack.pop())
        )
    return parentheses_locs


# compute_bijection("(((....)))....(...)")
# array([ 9,  8,  7,  3,  4,  5,  6,  2,  1,  0, 10, 11, 12, 13, 18, 15, 16,
#       17, 14])
def compute_bijection(s):
    parens = find_parentheses(s)
    ret = np.arange(len(s))
    for x in parens:
        ret[x] = parens[x]
        ret[parens[x]] = x
    return ret


def compute_wobble_indicator(sequence, structure):
    # Compute an indicator vector of all the wobble base pairs (G-U or U-G)
    assert len(sequence) == len(structure)
    assert set(sequence).issubset(
        {"A", "C", "G", "T"}
    ), "Unknown character found in sequence"
    bij = compute_bijection(structure)
    return [
        (1 if {sequence[i], sequence[bij[i]]} == {"G", "T"} else 0)
        for i in range(len(sequence))
    ]


def folding_to_vector(nts):
    # return str_to_vector(nts, ".,|{}()")
    return str_to_vector(nts, ".()")


def safelog(x, tol=1e-9):
    return np.log(np.maximum(x, tol))


def bin_kl(y_true, y_pred):
    y_0 = (1 - y_true) * (safelog(1 - y_true) - safelog(1 - y_pred))
    y_1 = y_true * (safelog(y_true) - safelog(y_pred))

    return y_0 + y_1


def flatten_dict(d):
    keys = []
    values = []
    for key in d:
        for value in d[key]:
            keys.append(key)
            values.append(value)
    return keys, values


def nt_freqs(arr):
    out = np.zeros((4, arr.shape[-1]))

    for j in range(arr.shape[-1]):
        for i in range(4):
            out[i, j] = (arr[:, j] == i).sum()
    return out/out.sum(axis=0)


def apply_to_elems(d, fn, preprocess_fn, merge_fn):
    out = dict()
    keys, values = flatten_dict(d)
    values = preprocess_fn(values)
    mapped_values = fn(values)

    for key, val in zip(keys, mapped_values):
        if key not in out:
            out[key] = []
        out[key].append(val)

    for key in out.keys():
        out[key] = merge_fn(out[key])
    return out


def insert_motif_in_middle_of_sequence(seq_nt, test_motif):
    seq_nt_copy = list(seq_nt)
    start_index = int(len(seq_nt) / 2 - len(test_motif) / 2)
    for j in range(len(test_motif)):
        seq_nt_copy[start_index + j] = test_motif[j]
    seq_nt_copy = "".join(seq_nt_copy)
    return seq_nt_copy


def insert_motif_in_middle_of_sequences(seq_nts, test_motif):
    out = {}

    for seq_nt in tqdm(seq_nts, desc=f"Creating {test_motif} landing pads"):
        out[seq_nt] = [insert_motif_in_middle_of_sequence(seq_nt, test_motif)]

    return out


def landing_pads_to_sw_exons(fourteen_mers, test_motif, prefix="", suffix=""):
    out = []

    n = len(fourteen_mers)
    for i in range(n):
        tmp = [str(e) for e in fourteen_mers]
        tmp[i] = insert_motif_in_middle_of_sequence(tmp[i], test_motif)
        out.append(prefix + "".join(tmp) + suffix)
    return out


def all_seqs(length):
    nts_list = [["A", "C", "G", "U"] for i in range(length)]
    return ["".join(x) for x in product(*nts_list)]


def compute_activations_simple_conv(layer, window_size=6, kind="simple"):
    all_kmers = all_seqs(window_size)
    all_seqs_oh = np.array(
        [nts_to_vector(s_i, rna=True) for s_i in all_seqs(window_size)],
        dtype=np.float32,
    )

    activations = tf.squeeze(layer(all_seqs_oh), axis=1).numpy()

    dfs_by_filter = dict()

    for j in range(activations.shape[1]):
        dfs_by_filter[j] = pd.DataFrame(
            {"activation": activations[:, j], "input": all_kmers}
        ).sort_values("activation", ascending=False)

    return dfs_by_filter


def compute_activations(seqs, qc_incl, qc_skip):
    incl_act = qc_incl(seqs).numpy()
    skip_act = qc_skip(seqs).numpy()
    raw_activations = np.concatenate([incl_act, skip_act], axis=2)
    return raw_activations


def compute_sw_activations(sw_seqs_nt, qc_incl, qc_skip):
    key_subkeys = []
    vectors = []

    for key in tqdm(sw_seqs_nt.keys()):
        for subkey in sw_seqs_nt[key]:
            key_subkeys.append((key, subkey))
            vectors.append(nts_to_vector(subkey, rna=True))

    vectors = np.stack(vectors)
    activations = compute_activations(vectors, qc_incl, qc_skip)

    out = dict()
    for (key, subkey), act in zip(key_subkeys, activations):
        if key not in out:
            out[key] = dict()
        out[key][subkey] = act

    return out


def extract_str_patches(lst, n):
    out = []
    for elem in lst:
        tmp = []
        n_elem = len(elem)
        assert n_elem >= n, f"{elem} is of length < n ({n})"
        for i in range(n_elem - n + 1):
            tmp.append(elem[i: i + n])
        out.append(tmp)
    return out


def one_hot(seqs):
    out = []

    for row in seqs:
        out_i = []
        for row_j in row:
            e_ij = np.zeros(4)
            e_ij[int(row_j)] = 1
            out_i.append(e_ij)
        out.append(out_i)
    return np.array(out)


def sample_seqs(seq_freqs, n_samples=1):
    out = np.zeros((n_samples, seq_freqs.shape[1]))

    for i in tqdm(range(n_samples)):
        for j in range(seq_freqs.shape[1]):
            out[i, j] = np.random.choice(np.arange(4), p=seq_freqs[:, j])

    return out


def oh_2_str(x, fn=None, kind="seq"):
    if fn is None:
        if kind == "seq":
            fn = np.vectorize(lambda x: {0: "A", 1: "C", 2: "G", 3: "U"}[x])
        elif kind == "struct":
            fn = np.vectorize(lambda x: {0: ".", 1: "(", 2: ")"}[x])
        else:
            raise NotImplementedError(
                f"Function {kind} has not been implemented.")

    x_am = x.argmax(axis=-1)
    if len(x_am.shape) > 1:
        return np.array(["".join(fn(e)) for e in x_am])
    return "".join(fn(x_am))


def rna_fold_structs(seq_nts, maxBPspan=0):
    struct_mfes = RNAutils.RNAfold(
        seq_nts,
        maxBPspan=maxBPspan,  # maxBPspan 0 means don't pass in maxBPpan
        RNAfold_bin="RNAfold",
    )
    structs = [e[0] for e in struct_mfes]
    mfes = np.array([e[1] for e in struct_mfes])
    return structs, mfes


def compute_structure(seq_nts, maxBPspan=0):
    structs, mfes = rna_fold_structs(seq_nts, maxBPspan=maxBPspan)
    # one-hot-encode structure
    struct_oh = np.array([folding_to_vector(x) for x in structs])

    return struct_oh, structs, mfes


def compute_seq_oh(seq_nts):
    return np.array([nts_to_vector(x) for x in [seq.replace('U', 'T') for seq in seq_nts]])


def compute_wobbles(seq_nts, structs):
    return np.array([
        np.expand_dims(compute_wobble_indicator(
            x.replace('U', 'T'), y), axis=-1)
        for (x, y) in zip(seq_nts, structs)
    ])


def create_input_data(seq_nts, maxBPspan=0):
    # get sequence one-hot-encodings
    seq_oh = compute_seq_oh(seq_nts)

    # get structure one-hot-encodings and mfe
    struct_oh, structs, _ = compute_structure(seq_nts, maxBPspan=maxBPspan)

    # compute wobble pairs
    wobbles = compute_wobbles(seq_nts, structs)

    return seq_oh, struct_oh, wobbles


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
