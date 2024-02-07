# README.md

## Behavioral Cloning Training

To train the model using behavioral cloning, run the following command:

```bash
python run_hw1_bc.py
```
## Inverse Dynamique Training : 

To train the model using IDM, run the following command : 

```bash
python run_hw1_bc.py alg.n_iter=1 alg.do_dagger=false alg.train_idm=true
```

## DAgger

To train the model using DAgger, run the following command:

```bash
python run_hw1_bc.py alg.n_iter=5 alg.do_dagger=true alg.train_idm=false
```