import pandas as pd
import figutils as utils
import os

### READ ALL INPUT DATASETS

NUM_LIBS = 14
ALL_LIBRARY_NAMES = {
    11: "ES7_HeLa_A",
    12: "ES7_HeLa_B",
    13: "ES7_HeLa_C",
}
ALL_FILE_NAMES = {
    11: "BS11504A_S1",
    12: "BS11505A_S2",
    13: "BS11506A_S3",
}

def read_all_datasets_no_filtering(DATA_FOLDER):
    all_barcode_statistics = []
    for lib_num in ALL_LIBRARY_NAMES:
        FILE_NAME = ALL_FILE_NAMES[lib_num]
        FULL_FILE_NAME = os.path.join(DATA_FOLDER, FILE_NAME)

        barcode_statistics = pd.read_csv(
            FULL_FILE_NAME + "_splicing_analysis.csv"
        ).set_index("barcode")

        all_barcode_statistics.append(barcode_statistics)

    return all_barcode_statistics
