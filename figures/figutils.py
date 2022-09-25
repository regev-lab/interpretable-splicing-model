import numpy as np
from tqdm.auto import tqdm
from itertools import product
import tensorflow as tf
import pandas as pd
import RNAutils
import matplotlib.pyplot as plt
import scipy.stats
from matplotlib import cm

def subsample_points(x, y, max_points=30000):
    assert(len(x)==len(y))
    if (len(x)>max_points):
        indices_to_pick = np.random.choice(range(len(x)), size=max_points)
        x = x[indices_to_pick]
        y = y[indices_to_pick]
    return x, y
    
def scatter_with_kde(x,y,ax,alpha=0.2, max_points=30000):
    x, y = subsample_points(x, y, max_points)

    # Calculate the point density
    xy = np.vstack([x,y])
    z = scipy.stats.gaussian_kde(xy)(xy)

    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
        
    ax.scatter(x,y, alpha=alpha, s=10,  c=np.log(z), cmap=plt.cm.jet, linewidth=0.)


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


def extract_str_patches(lst, n):
    out = []
    for elem in lst:
        tmp = []
        n_elem = len(elem)
        assert n_elem >= n, f"{elem} is of length < n ({n})"
        for i in range(n_elem - n + 1):
            tmp.append(elem[i : i + n])
        out.append(tmp)
    return out


def softplus(x):
    return np.log(1 + np.exp(x))

def compute_activations_simple_conv(layer,
                                    window_size=6,
                                    kind='simple'):
    all_kmers = all_seqs(window_size)
    all_seqs_oh = np.array(
        [nts_to_vector(s_i, rna=True) for s_i in all_seqs(window_size)],
        dtype=np.float32)
    
    activations = tf.squeeze(layer(all_seqs_oh), axis=1).numpy()

    dfs_by_filter = dict()
    
    for j in range(activations.shape[1]):
        dfs_by_filter[j] = pd.DataFrame({
        "activation": activations[:, j],
        "input": all_kmers
    }).sort_values('activation', ascending=False)
    
    return dfs_by_filter

def ei_vec(i,len):  # give a one-hot encoding
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

def oh_2_str(x, fn=None, kind='seq'):
    if fn is None:
        if kind == 'seq':
            fn = np.vectorize(lambda x: {0:'A', 1:'C', 2:'G', 3:'U'}[x])
        elif kind == 'struct':
            fn = np.vectorize(lambda x: {0:'.', 1:'(', 2:')'}[x])
        else:
            raise NotImplementedError(f'Function {kind} has not been implemented.')
    
    x_am = x.argmax(axis=-1)
    if len(x_am.shape) > 1:
        return np.array([''.join(fn(e)) for e in x_am])
    return ''.join(fn(x_am))

def rna_fold_structs(seq_nts):
    struct_mfes = RNAutils.RNAfold(
        seq_nts,
        # maxBPspan=30,
        RNAfold_bin='RNAfold')
    structs = [e[0] for e in struct_mfes]
    mfes = np.array([e[1] for e in struct_mfes])
    return structs, mfes

def compute_structure(seq_nts):
    structs, mfes = rna_fold_structs(seq_nts)
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

def create_input_data(seq_nts):
    # get sequence one-hot-encodings
    seq_oh = compute_seq_oh(seq_nts)
    
    # get structure one-hot-encodings and mfe
    struct_oh, structs, _ = compute_structure(seq_nts)
    
    # compute wobble pairs
    wobbles = compute_wobbles(seq_nts, structs)
    
    return seq_oh, struct_oh, wobbles

def find_parentheses(s):
    """ Find and return the location of the matching parentheses pairs in s.

    Given a string, s, return a dictionary of start: end pairs giving the
    indexes of the matching parentheses in s. Suitable exceptions are
    raised if s contains unbalanced parentheses.

    """

    # The indexes of the open parentheses are stored in a stack, implemented
    # as a list

    stack = []
    parentheses_locs = {}
    for i, c in enumerate(s):
        if c == '(':
            stack.append(i)
        elif c == ')':
            try:
                parentheses_locs[stack.pop()] = i
            except IndexError:
                raise IndexError('Too many close parentheses at index {}'
                                                                .format(i))
    if stack:
        raise IndexError('No matching close parenthesis to open parenthesis '
                         'at index {}'.format(stack.pop()))
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
    ## Compute an indicator vector of all the wobble base pairs (G-U or U-G)
    assert(len(sequence)==len(structure))
    bij = compute_bijection(structure)
    return [(1 if {sequence[i], sequence[bij[i]]}=={'G','T'} else 0) for i in range(len(sequence))]    

def folding_to_vector(nts):
   #return str_to_vector(nts, ".,|{}()")
   return str_to_vector(nts, ".()")

PRE_SEQUENCE = "TCTGCCTATGTCTTTCTCTGCCATCCAGGTT"
POST_SEQUENCE = "CAGGTCTGACTATGGGACCCTTGATGTTTT"


def add_flanking(nts, flanking_len):
    return PRE_SEQUENCE[-flanking_len:] + nts + POST_SEQUENCE[:flanking_len]
