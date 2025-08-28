dataset_folder="/nfshomes/asarkar6/aditya/PRISM/validation/"
output_dir="/nfshomes/asarkar6/aditya/gen_images/"
checkpoint_dir="/nfshomes/asarkar6/trinity/model_weights/metaquery_training1/"

OMP_NUM_THREADS=12 torchrun --nproc-per-node=2 --master-port=29501 /nfshomes/asarkar6/aditya/PRISM/metaquery/infer.py \
 --dataset_folder=$dataset_folder \
 --output_dir=$output_dir \
 --checkpoint_path=$checkpoint_dir \
 --inference_type=1