cd src
python train.py mot --exp_id mot20_ft_mix_dla34 --load_model '../models/crowdhuman_dla34.pth' --num_epochs 60 --lr_step '15' --data_cfg '../src/lib/cfg/mot20.json'
cd ..