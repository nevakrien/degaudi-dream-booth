# degaudi-dream-booth
working on taking code ment for gaudi and making it cross platform pytorch
https://github.com/huggingface/optimum-habana/blob/main/examples/stable-diffusion/training/train_dreambooth.py

apperently its already based on cross platform code at https://github.com/huggingface/peft/blob/608a90ded9985ee1c5912d738082bb1fd618902b/examples/stable_diffusion/train_dreambooth.py 

so I just stole that and tried to adapt the runing script

no requirments file because honestly I stopped beliving in reproducing python scripts after half a year. 
and I belive that putting one in here will make your life harder.
just install whatever is missing (potentially want to install some specific version of jax as noted in https://stackoverflow.com/questions/78302031/stable-diffusion-attributeerror-module-jax-random-has-no-attribute-keyarray)

export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="dog"
export CLASS_DIR="path-to-class-images"
export OUTPUT_DIR="out"

python main.py\
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --class_data_dir=$CLASS_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks dog" \
  --class_prompt="a photo of dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --num_class_images=200 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=800 \
  --mixed_precision=bf16 \
  full
