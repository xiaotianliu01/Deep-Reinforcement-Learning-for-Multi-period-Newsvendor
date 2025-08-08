#!/bin/bash
for c in 0.8; do
    for gamma in 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95; do
        python train.py --critical_ratio=$c --gamma=$gamma
        python baseline.py --exp_name='walmart_e2e' --log_dir='./results/walmart_e2e.txt' --bs=16 --epochs=300 --critical_ratio=$c --gamma=$gamma
        python baseline.py --exp_name='walmart_snn' --log_dir='./results/walmart_snn.txt' --bs=32 --epochs=300 --hidden_size=64 --layer_num=6 --critical_ratio=$c --gamma=$gamma
        python baseline.py --exp_name='walmart_pto' --log_dir='./results/walmart_pto.txt' --critical_ratio=$c --gamma=$gamma
        python baseline.py --exp_name='walmart_saa' --log_dir='./results/walmart_saa.txt' --critical_ratio=$c --gamma=$gamma
        python baseline.py --exp_name='walmart_ko' --log_dir='./results/walmart_ko.txt' --critical_ratio=$c --gamma=$gamma
    done
done

for c in 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95; do
    for gamma in 0.6; do
        python train.py --critical_ratio=$c --gamma=$gamma
        python baseline.py --exp_name='walmart_e2e' --log_dir='./results/walmart_e2e.txt' --bs=16 --epochs=300 --critical_ratio=$c --gamma=$gamma
        python baseline.py --exp_name='walmart_snn' --log_dir='./results/walmart_snn.txt' --bs=32 --epochs=300 --hidden_size=64 --layer_num=6 --critical_ratio=$c --gamma=$gamma
        python baseline.py --exp_name='walmart_pto' --log_dir='./results/walmart_pto.txt' --critical_ratio=$c --gamma=$gamma
        python baseline.py --exp_name='walmart_saa' --log_dir='./results/walmart_saa.txt' --critical_ratio=$c --gamma=$gamma
        python baseline.py --exp_name='walmart_ko' --log_dir='./results/walmart_ko.txt' --critical_ratio=$c --gamma=$gamma
    done
done