export DATA_FOLDER=./data
export MODEL_FOLDER=./output

python model_training/train_model.py --index 0 --data_folder $DATA_FOLDER --model_folder $MODEL_FOLDER --results_folder $MODEL_FOLDER --epochs_per_batch_step 10

