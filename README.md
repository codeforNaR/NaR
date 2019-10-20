# NaR
Code for reproducing results in AAAI-2020 submission.

# Dependencies
- pytorch-v1.0+
- numpy 1.16.3
- progress v1.5

# To run the code
```
python cifar_nar.py --arch resnet --arch2 resnet --depth 32 --depth2 32 --epochs 300 --schedule 150 225 --gamma 0.1 --wd 1e-4 --lam 0.5 --alpha 0.1 --dataset cifar100 --memo test --gpu-id 0 --manualSeed 201905
```

## You can try other types networks.
Configuration table:

Model | Depth | weight decay | gamma |epochs | schedule
---|---|---|---|---|---
plaincnn | 6 | 5e-4 | 0.1 | 200 | 60 120 160
resnet | 32 | 1e-4 | 0.1 | 300 | 150 225
preresnet | 110 | 1e-4 | 0.1 | 300 | 150 225
wrn | 28 | 5e-4 | 0.2 | 200 | 60 120 160

## Robustness test on models trained with NaR
Here, we compare robustness of the model trained with NaR and vanilla training strategy. The robust test consists of two parts, one is input noise test and the other is parameter noise test. The noise type is supposed to be Gaussian noise with std in [0.01, 0.05, 0.1, 0.15, 0.2, 0.3]. Note that we only test the last linear layer of the model's parameter.

For each noise level, we've run 10 times and report the average performance.
For input noise:
![image](https://github.com/codeforNaR/NaR/blob/master/images/input_noise.png)
For parameter noise:
![image](https://github.com/codeforNaR/NaR/blob/master/images/parameter_noise.png)
