# Office-Home, Resnet 50, 30 epochs, Multi source
CUDA_VISIBLE_DEVICES=0 python mdama_OH.py data/office-home -d OfficeHome -s Cl Pr Rw -t Ar -a resnet50 --epochs 40 --bottleneck-dim 2048 --seed 0 --log logs/mamda_OH/OfficeHome_:2Ar
CUDA_VISIBLE_DEVICES=0 python mdama_OH.py data/office-home -d OfficeHome -s Ar Pr Rw -t Cl -a resnet50 --epochs 40 --bottleneck-dim 2048 --seed 0 --log logs/mamda_OH/OfficeHome_:2Cl
CUDA_VISIBLE_DEVICES=0 python mdama_OH.py data/office-home -d OfficeHome -s Cl Ar Rw -t Pr -a resnet50 --epochs 40 --bottleneck-dim 2048 --seed 0 --log logs/mamda_OH/OfficeHome_:2Pr
CUDA_VISIBLE_DEVICES=0 python mdama_OH.py data/office-home -d OfficeHome -s Ar Pr Cl -t Rw -a resnet50 --epochs 40 --bottleneck-dim 2048 --seed 0 --log logs/mamda_OH/OfficeHome_:2Rw
