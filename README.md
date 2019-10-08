# Link prediction for Chemical Synthesis pairs

```
# train
mpiexec -n 8 python train.py --batch_size 20 --epoch 1000 --lr 1e-4 --gpu --out 'result_all' --frequency 20 --decay_iter 100000 --gnn_dim 300 --n_layers 3 --nn_dim 100 --type 'all' --rich 'true' --train_path '../train.txt.proc' --valid_path '../test.txt.proc'

# inference
mpiexec -n 8 python inference.py --batch_size 20 --out 'result_1002_all/' --valid_path '../test.txt.proc' --type 'all' --rich 'true' --snapshot 'snapshot_51130'
```
