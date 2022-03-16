DATA=../../../BrainScore/fast_adversarial_transformers/ImageNet/Dataset/imagenet
NAME_CONFIG=crossvit-18-dagger-408-FAT-ROTINV-GRAY
CONFIG=../Configs/${NAME_CONFIG}/configs_fast_2px.yml
python -u train_adv.py --data $DATA --config $CONFIG
