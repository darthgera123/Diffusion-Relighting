## ControlNet Training
accelerate launch train_controlnet.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --resolution=512 \
 --learning_rate=1e-5 \
 --train_batch_size=4 \
--cache_dir=/scratch/inf0/user/pgera/FlashingLights/SingleRelight/cache \
--relit_path /scratch/inf0/user/pgera/FlashingLights/SingleRelight/sunrise_pullover/pose_01 \
--env_path /scratch/inf0/user/pgera/FlashingLights/SingleRelight/envmaps/med_exr