# Reads the raw DNA reads, filters out those that don't look like barcode reads, and sorts by barcode

import sys
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import io
import os
from collections import Counter
from utils import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_folder", required=True, type=str, help="Input folder")
args = parser.parse_args()

###############################################
## Read the plasmid sequencing file into memory
## Once a barcode is identified, we store the corresponding exon in a Counter for later analysis.
## If exon cannot be read, record it as error (we cannot ignore it, since it might correspond to a badly coupled barcode, like CACATACTGAGCATCTAACT in ES7A)


couplings = {}


def collect_barcodes(read_1, read_2, read_1_q, read_2_q):
    global good_reads
    global reads_with_N
    global unidentified_reads
    global couplings
    global lib_num

    # Check read 1 (barcode) proper format
    # This should match the check done in the RNA data analysis
    assert len(read_1) == 54
    if "N" in read_1:
        reads_with_N += 1
        return
    if (read_1[5 + 14 : 5 + 16] != "AT") or (
        hamming(read_1[5 : 5 + 16], "TTTAAACGGGCCCTAT") >= 2
    ):
        unidentified_reads += 1
        return
    if (read_1[41:43] != "TC") or (hamming(read_1[41:], "TCTAGTGAGACGT") >= 2):
        unidentified_reads += 1
        return

    # Barcode identified!
    BARCODE_POSITION = 21
    barcode = revcomp(read_1[BARCODE_POSITION : BARCODE_POSITION + 20])

    good_reads += 1

    # add barcode to coupling dictionary if not already there
    if not barcode in couplings:
        couplings[barcode] = [
            Counter(),
            0,
        ]  # second coordinate counts number of bad reads

    # Check read 2 (exon) proper format
    assert len(read_2) == 106
    # if N is in read 2, it's a bad read. Exit the function
    if "N" in read_2:
        couplings[barcode][1] += 1
        return
    # This also filters out exons with deletions, which are reasonably common; consider improving:
    READ_2_PREFIX = {0: "GCCATCCAGGTT"}[lib_num]
    if read_2[:12] != READ_2_PREFIX:
        couplings[barcode][1] += 1
        return
    if read_2[82:87] != "CAGGT":
        couplings[barcode][1] += 1
        return

    # if we've passed all the bad read exit conditions,
    # add the exon sequence (70 nt) to the couplings[barcode] counter
    exon = read_2[12 : 12 + 70]
    couplings[barcode][0][exon] += 1


ALL_LIBRARY_NAMES = {0: "ES7A"}
ALL_FILE_NAMES = {0: "BS06911A_S22"}
ALL_BASE_DIR_NAMES = {0: os.path.join(args.input_folder, "Sample_BS06911A/")}

all_couplings = []
for lib_num in ALL_LIBRARY_NAMES:
    good_reads = 0
    reads_with_N = 0
    unidentified_reads = 0
    couplings = {}

    BASE_DIR_NAME = ALL_BASE_DIR_NAMES[lib_num]
    FILE_NAME = ALL_FILE_NAMES[lib_num]
    FULL_FILE_NAME = BASE_DIR_NAME + FILE_NAME
    num_reads = process_paired_fastq_file(
        FULL_FILE_NAME + "_R1_001.fastq",
        FULL_FILE_NAME + "_R2_001.fastq",
        collect_barcodes,
    )

    all_couplings.append(couplings)

    print(
        "Done reading file",
        FILE_NAME,
        ":",
        human_format(num_reads),
        "total reads;",
        human_format(unidentified_reads),
        "unidentified reads; ",
        human_format(reads_with_N),
        "reads with N;",
        human_format(good_reads),
        "good reads",
    )

##############################
# Check coupling between exons and barcodes, and write it to file
##############################

# below that threshold, we won't even write the barcode to the coupling file; this is meant to filter read errors in the barcode, as those are unlikely to be seen more than once.
MIN_NUMBER_OF_READS = 2

for lib_num in ALL_LIBRARY_NAMES:
    print("Processing", ALL_FILE_NAMES[lib_num])
    f = open(ALL_BASE_DIR_NAMES[lib_num] + "coupling.txt", "w")
    num_keys_with_enough_reads = 0
    num_uniquely_coupled_keys = 0
    num_too_many_errors_in_exon = 0
    num_with_no_clear_majority = 0
    barcode_coupling = []
    couplings = all_couplings[lib_num]
    for barcode in tqdm(couplings.keys()):
        coupling_data = couplings[barcode][0]
        # total number of reads for that barcode; note that len(coupling_data) gives the number of different *exons* associated with this barcode
        reads_for_barcode = sum(coupling_data.values())
        if reads_for_barcode < MIN_NUMBER_OF_READS:  # too few reads
            continue
        num_keys_with_enough_reads += 1
        sequence_frequencies: Counter = Counter(coupling_data)
        num_reads_most_common = sequence_frequencies.most_common(1)[0][1]
        # second most common exon should not be too common (since random errors should not form clusters)
        if (len(sequence_frequencies) > 1) and (
            sequence_frequencies.most_common(2)[1][1]
            >= max(2, num_reads_most_common / 4)
        ):
            badly_coupled = True
            num_with_no_clear_majority += 1
        # number of bad reads should also not be too high
        elif couplings[barcode][1] >= max(2, num_reads_most_common / 4):
            badly_coupled = True
            num_too_many_errors_in_exon += 1
        else:
            badly_coupled = False
            num_uniquely_coupled_keys += 1
        most_common_full_exon = sequence_frequencies.most_common(1)[0][0]
        # TODO: flanking should use utils file
        most_common_full_exon_with_flanking = "AGGTT" + most_common_full_exon + "CAGGT"
        most_common_full_exon_contains_restriction_site = (
            "CGTCTC" in most_common_full_exon_with_flanking
        ) or ("GAGACG" in most_common_full_exon_with_flanking)
        barcode_coupling.append(
            [
                barcode,
                most_common_full_exon,
                badly_coupled,
                most_common_full_exon_contains_restriction_site,
                reads_for_barcode,
            ]
        )
        print(
            "?"
            if most_common_full_exon_contains_restriction_site
            else ("*" if badly_coupled else "."),
            barcode,
            sequence_frequencies.most_common(2),
            file=f,
        )

    print(
        "Total number of barcodes seen:",
        human_format(len(couplings)),
        "Barcodes with enough reads:",
        human_format(num_keys_with_enough_reads),
        "Uniquely coupled barcodes:",
        human_format(num_uniquely_coupled_keys),
        "Barcodes with too many errors in exon reads:",
        human_format(num_too_many_errors_in_exon),
        "Barcodes with no clear majority exon:",
        human_format(num_with_no_clear_majority),
    )

    f.close()

    df = pd.DataFrame(
        barcode_coupling,
        columns=[
            "barcode",
            "exon",
            "badly_coupled",
            "contains_restriction_site",
            "num_reads",
        ],
    ).set_index("barcode")
    df.to_csv(ALL_BASE_DIR_NAMES[lib_num] + "coupling.csv")
    print(f"Wrote coupling.csv to {ALL_BASE_DIR_NAMES[lib_num]}coupling.csv")
