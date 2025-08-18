# python .\train.py .\configs\mp_rslds.json --force
# python train.py configs/mr_gnode.json --force 
# python train.py configs/cca.json --force
# python train.py configs/rrr.json --force


# python .\comparison.py .\configs\mp_rslds.json .\configs\mr_gnode.json .\configs\cca.json .\configs\rrr.json


# Run 5-fold CV for each model
python cross_validate.py configs/mr_gnode.json --n-folds 5 --train-trials 80 --test-trials 20
python cross_validate.py configs/mp_rslds.json --n-folds 5 --train-trials 80 --test-trials 20  
python cross_validate.py configs/cca.json --n-folds 5 --train-trials 80 --test-trials 20
python cross_validate.py configs/rrr.json --n-folds 5 --train-trials 80 --test-trials 20