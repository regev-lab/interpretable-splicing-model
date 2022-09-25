# Take cDNA reads, and produce splicing statistics per barcode

# Read 1: NNNNN[N][N]TTTAAACGGGCCCTATNNNNNNNNNNNNNNNNNNNNTCTAGAGCGAG[CT]
#   Number of Ns (UMI) is random 5-7; barcode is 20N
# Read 2:
#   Diversity: [NN]  (0-2Ns)
#   End of exon 1: AAGTTGGTGGTGAGGCCCTGGGCAG
#   Exon 2: GTTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNCAG
#   Exon 3: CTCCTGGGCAACGTGCTGGTCTGTGTGCTGGCCCATCACTTTGGCAAAGAATTCACCCCACCAGTGCAGGCTGCCTATCAGAAAGTGGTGGCTGGTGTGGCTAATGCCCTGGCCCACAAGTATCACTAAGCTCGCTCTAGA

# Plasmid library:
#   ES7A_Lib1	BS06911A

# %%
import os
import pandas as pd
import numpy as np
from collections import Counter
import random
from utils import *
import argparse
from tqdm.auto import tqdm

# %%
parser = argparse.ArgumentParser()
parser.add_argument("--input_folder", required=True, type=str, help="Input folder")
parser.add_argument("--output_folder", required=True, type=str, help="Output folder")
parser.add_argument(
    "--plasmid_coupling_file_name",
    required=True,
    type=str,
    help="Plasmid coupling filename. Must be a csv file.",
)
args = parser.parse_args()

# %%

NUM_LIBS = 3
ALL_LIBRARY_NAMES = {0: "ES7_HeLa_A", 1: "ES7_HeLa_B", 2: "ES7_HeLa_C"}
ALL_FILE_NAMES = {0: "BS11504A_S1", 1: "BS11505A_S2", 2: "BS11506A_S3"}
ALL_BASE_DIR_NAMES = {
    0: "Sample_BS11504A/",
    1: "Sample_BS11505A/",
    2: "Sample_BS11506A/",
}


PLASMID_COUPLING_FILE_NAME = args.plasmid_coupling_file_name
# "data/Sample_BS07028A/coupling.csv"
INPUT_FOLDER = args.input_folder
OUTPUT_FOLDER = args.output_folder

assert os.path.isdir(INPUT_FOLDER), f"No such folder: {INPUT_FOLDER}"
if not os.path.isdir(OUTPUT_FOLDER):
    print(f"{OUTPUT_FOLDER} not found. Creating...")
    os.makedirs(OUTPUT_FOLDER)

# %%
bad_read_1_reads = 0
unknown_barcode_reads = 0

SUBSAMPLE_RATIO = 1  # Only analyze 1 in SUBSAMPLE_RATIO samples; used to get a quick sample for testing purposes; set to 1 in production


def identify_splicing_pattern(read_1, read_2, read_1_q, read_2_q):
    global bad_read_1_reads
    global unknown_barcode_reads
    global barcode_statistics

    if (SUBSAMPLE_RATIO > 1) and (random.randrange(SUBSAMPLE_RATIO) != 0):
        return

    # Check read 1 (barcode) proper format
    # This should match the check done in the DNA data analysis
    assert len(read_1) == 54
    if "N" in read_1:
        bad_read_1_reads += 1
        return

    # Try to identify length of UMI (5-7nt)
    umi_length = -1
    for i in (
        5,
        6,
        7,
    ):
        if (read_1[i + 14 : i + 16] != "AT") or (
            hamming(read_1[i : i + 16], "TTTAAACGGGCCCTAT") >= 2
        ):
            continue
        if (read_1[i + 36 : i + 38] != "TC") or (
            hamming(read_1[i + 36 :], "TCTAGAGCGAGCT."[: 4 - i]) >= 2
        ):  # this is important to distinguish Lib1 carryover from Lib2 product; Lib1 ends with TCTAGTGAGACGT
            continue
        umi_length = i
        break
    if (
        umi_length == -1
    ):  # we were unable to identify a frame containing the desired sequences
        bad_read_1_reads += 1
        return

    # Barcode identified!
    barcode = revcomp(read_1[umi_length + 16 : umi_length + 16 + 20])
    if not barcode in barcode_statistics.index:  # Barcode not in the plasmid sequencing
        unknown_barcode_reads += 1
        return

    # At this point we identified the barcode and found it in the coupling database, so the output should be recorded in the Dataframe row for that barcode

    # Check read 2 (exon) proper format
    assert len(read_2) == 106
    if "N" in read_2:
        barcode_statistics.at[barcode, "num_bad_reads"] += 1
        return
    EXON_1 = "AAGTTGGTGGTGAGGCCCTGGGCAG"
    read2_frame = -1
    for i in range(
        3
    ):  # try to identify frame of read (0-2nt of Ns in beginning of Read 2)
        if (
            hamming(read_2[i : i + 25], EXON_1) > 2
        ):  # there are often read errors in the beginning of the read. Allow up to 2.
            continue
        read2_frame = i
        break
    if read2_frame == -1:  # we could not identify where exon 1 is
        barcode_statistics.at[barcode, "num_bad_exon1"] += 1
        return

    if (
        read_2[read2_frame + 25 : read2_frame + 35] == "CTCCTGGGCA"
    ):  # this is the beginning of exon 3, so we have exon skipping
        barcode_statistics.at[barcode, "num_exon_skipping"] += 1
        return
    if (
        read_2[read2_frame + 25 : read2_frame + 35] == "GTTGGTATCA"
    ):  # this is the beginning of intron 1, so we have intron retention
        barcode_statistics.at[barcode, "num_intron_retention"] += 1
        return
    if (
        hamming(
            read_2[read2_frame + 25 :],
            "GTT"
            + barcode_statistics.at[barcode, "exon"]
            + "CAG"
            + "CTCCT."[: -1 - read2_frame],
        )
        <= 2
    ):  # we see the full randomized exon and exon 3
        barcode_statistics.at[barcode, "num_exon_inclusion"] += 1
        return
    if (
        read_2[read2_frame + 25 : read2_frame + 25 + 6]
        == "GTT" + barcode_statistics.at[barcode, "exon"][:3]
    ) and (
        "CTCCTGGGCAA" in read_2[read2_frame + 25 + 6 :]
    ):  # we see the beginning of exon 2, but also beginning of exon 3; probably splicing in randomized exon
        barcode_statistics.at[barcode, "num_splicing_in_exon"] += 1
        return
    # otherwise, we were unable to identify the splicing pattern
    barcode_statistics.at[barcode, "num_unknown_splicing"] += 1


# %%

all_barcode_statistics = []
for lib_num in tqdm(range(NUM_LIBS), desc="Iterating libraries"):
    bad_read_1_reads = 0
    unknown_barcode_reads = 0

    barcode_statistics = pd.read_csv(PLASMID_COUPLING_FILE_NAME).set_index("barcode")
    barcode_statistics["num_intron_retention"] = [
        0 for i in range(len(barcode_statistics))
    ]
    barcode_statistics["num_exon_inclusion"] = [
        0 for i in range(len(barcode_statistics))
    ]
    barcode_statistics["num_exon_skipping"] = [
        0 for i in range(len(barcode_statistics))
    ]
    barcode_statistics["num_bad_reads"] = [0 for i in range(len(barcode_statistics))]
    barcode_statistics["num_bad_exon1"] = [0 for i in range(len(barcode_statistics))]
    barcode_statistics["num_splicing_in_exon"] = [
        0 for i in range(len(barcode_statistics))
    ]
    barcode_statistics["num_unknown_splicing"] = [
        0 for i in range(len(barcode_statistics))
    ]

    BASE_DIR_NAME = ALL_BASE_DIR_NAMES[lib_num]
    FILE_NAME = ALL_FILE_NAMES[lib_num]
    FULL_FILE_NAME = BASE_DIR_NAME + FILE_NAME
    num_reads = process_paired_fastq_file(
        os.path.join(INPUT_FOLDER, FULL_FILE_NAME + "_R1_001.fastq"),
        os.path.join(INPUT_FOLDER, FULL_FILE_NAME + "_R2_001.fastq"),
        identify_splicing_pattern,
    )
    print(
        "Done reading file",
        FILE_NAME,
        "(" + ALL_LIBRARY_NAMES[lib_num] + ")",
        ":",
        human_format(num_reads),
        "total reads;",
        human_format(unknown_barcode_reads),
        "reads with unknown barcode",
        human_format(bad_read_1_reads),
        "reads with bad Read 1",
    )
    barcode_statistics.to_csv(
        os.path.join(OUTPUT_FOLDER, FILE_NAME + "_splicing_analysis.csv")
    )
    all_barcode_statistics.append(barcode_statistics)
