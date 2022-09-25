import argparse

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

import utils
from pathlib import Path

SEED = 420
np.random.seed(SEED)


def read_dataset(splicing_analysis_csv_path, filter_cryptic_restriction_site=True):
    barcode_statistics = pd.read_csv(splicing_analysis_csv_path).set_index("barcode")
    barcode_statistics = barcode_statistics[
        barcode_statistics.badly_coupled == False
    ]  # remove badly coupled barcodes

    # Filter barcodes containing restriction site, as those contain artifacts
    if filter_cryptic_restriction_site:
        contains_restriction_site = barcode_statistics.apply(
            lambda x: utils.contains_Esp3I_site(utils.add_flanking(x.exon, 5))
            or utils.contains_Esp3I_site(utils.add_barcode_flanking(x.name, 5)),
            axis=1,
        )
        barcode_statistics = barcode_statistics[~contains_restriction_site]

    return barcode_statistics


def to_input_data(df, flanking_length=10):
    assert flanking_length <= 30 and flanking_length >= 0

    return utils.create_input_data(
        [utils.add_flanking(exon, flanking_length) for exon in df.exon]
    )


def to_target_data(df):
    return np.array(
        df.num_exon_inclusion / (df.num_exon_inclusion + df.num_exon_skipping)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", required=True, type=str, help="Input folder")
    args = parser.parse_args()

    data_files = [
        "BS11504A_S1_splicing_analysis.csv",
        "BS11505A_S2_splicing_analysis.csv",
        "BS11506A_S3_splicing_analysis.csv",
    ]

    data_folder = args.input_folder

    splicing_analysis_csvs = [
        a
        for b in [
            list(Path(data_folder).rglob(f"*{data_file}")) for data_file in data_files
        ]
        for a in b
    ]

    print('Reading datasets... ', end='')
    datasets = [read_dataset(d) for d in splicing_analysis_csvs]
    print('Done.')

    numeric_columns = np.unique(
        [e for d in datasets for e in d.columns.values if "num" in e]
    )
    non_numeric_columns = np.unique(
        [e for d in datasets for e in d.columns.values if e not in numeric_columns]
    )

    d_numeric = sum([d[numeric_columns] for d in datasets])

    dataset = (datasets[0][non_numeric_columns]).join(d_numeric).dropna()

    # add statistics to dataset
    dataset["others"] = (
        dataset.num_unknown_splicing
        + dataset.num_intron_retention
        + dataset.num_bad_reads
        + dataset.num_bad_exon1
    )
    dataset["total"] = (
        dataset.others
        + dataset.num_exon_skipping
        + dataset.num_exon_inclusion
        + dataset.num_splicing_in_exon
    )

    # filter exons with too few reads
    MIN_READS = 60
    dataset = dataset[
        dataset.num_exon_skipping + dataset.num_exon_inclusion >= MIN_READS
    ]

    # Also, we want inclusion and skipping to be at least 80% of the total reads;
    # this gets rid of splice sites inside exon
    dataset = dataset[
        (dataset.num_exon_inclusion + dataset.num_exon_skipping) / dataset.total > 0.8
    ]

    # split dataset
    TEST_SPLIT_FRACTION = 0.2
    dataset_tr, dataset_te = train_test_split(
        dataset,
        test_size=TEST_SPLIT_FRACTION,
        train_size=1 - TEST_SPLIT_FRACTION,
        random_state=SEED,
    )

    # create datasets
    print('Computing structure, one-hot-encoding... ', end='')
    xTr = to_input_data(dataset_tr)
    yTr = to_target_data(dataset_tr)

    xTe = to_input_data(dataset_te)
    yTe = to_target_data(dataset_te)
    print('Done.')

    data_dump_list = [xTr, yTr, xTe, yTe, dataset_tr, dataset_te]

    dataset_names = [
        "xTr",
        "yTr",
        "xTe",
        "yTe",
        "barcode_statistics_train",
        "barcode_statistics_test",
    ]

    print('Dumping preprocessed data to disk... ', end='')
    for D, Dn in tqdm(
        zip(data_dump_list, dataset_names), leave=False, total=len(data_dump_list)
    ):
        dump(D, Path(data_folder) / f"{Dn}_ES7_HeLa_ABC.pkl.gz")
    print('Done.')
