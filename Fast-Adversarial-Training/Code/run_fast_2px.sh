# +
DATA=../../../BrainScore/fast_adversarial_transformers/ImageNet/Dataset/imagenet
#NAME_CONFIG=ViT-Large-Patch-16-224-FAT-ROTINV
#NAME_CONFIG=ViT-Small-Patch-32-224-FAT-ROTINV
#CONFIG=../Configs/${NAME_CONFIG}/configs_fast_2px.yml
#python -u train_adv.py --data $DATA --config $CONFIG

NAME_CONFIG=crossvit-15-dagger-240-FAT-ROTINV
CONFIG=../Configs/${NAME_CONFIG}/configs_fast_2px.yml
python -u train_adv.py --data $DATA --config $CONFIG

