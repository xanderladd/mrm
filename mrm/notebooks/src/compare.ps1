python .\train.py .\configs\mp_rslds.json --force
python train.py configs/mr_gnode.json --force 
python train.py configs/cca.json --force
python train.py configs/rrr.json --force


python .\comparison.py .\configs\mp_rslds.json .\configs\mr_gnode.json .\configs\cca.json .\configs\rrr.json