export DATA_FOLDER=./fasta_files

python data_preprocessing/compute_coupling.py --input_folder $DATA_FOLDER
python data_preprocessing/compute_splicing_outcomes.py \
    --input_folder $DATA_FOLDER \
    --output_folder $DATA_FOLDER \
    --plasmid_coupling_file_name $DATA_FOLDER/Sample_BS06911A/coupling.csv
python data_preprocessing/generate_training_data.py --input_folder $DATA_FOLDER

