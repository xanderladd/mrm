python .\trainer.py .\configs\pca_visp_config.json

# python .\trainer.py .\configs\pca_cp_config.json

cd visualization


python .\visualize_embeddings.py ..\configs\pca_visp_config.json

python .\visualize_embeddings.py ..\configs\pca_cp_config.json


python .\visualize_embeddings2.py ..\configs\pca_cp_config.json ..\configs\pca_visp_config.json

cd ../



