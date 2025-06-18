source /gpfs/home/acad/ulg-intelsig/jheld/anaconda3/etc/profile.d/conda.sh

echo "Start training : Reply = True"

conda activate VARS

python main.py --pooling_type "attention" --start_frame 63 --end_frame 87 --fps 17 --path "path/to/dataset" --pre_model "slowfast" --path_to_model_weights "best_model.pth"