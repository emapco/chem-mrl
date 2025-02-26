.PHONY: install install-pep lint release-testing docker \
				benchmark_db optuna_db clear_benchmark_db clear_optuna_db,\
				process_all_smiles_datasets run_bionemo \
				genmol_dataset_gen run_genmol run_multiple_genmol download_genmol \
				chem_2d_mrl_training chem_2d_mrl_training_molmim_augmented pubchem_pretraining full_dataset_pretraining

########## LIBRARY RECIPES ##########

install:
	pip install -e .[dev,benchmark,data]

install-pep:
	pip install .[dev,benchmark,data] --use-pep517

lint:
	ruff check chem_mrl --fix --config ruff.toml
	ruff format chem_mrl --config ruff.toml
	ruff analyze graph --config ruff.toml
	ruff clean

release-testing: lint
	pip uninstall chem_mrl -y
	python -m build
	pip install dist/*.whl
	rm -r dist/
	CUDA_VISIBLE_DEVICES=-1 pytest tests
	pip uninstall chem_mrl -y
	make install

########## DOCKER RECIPES ##########

docker:
	docker compose up -d --build benchmark-postgres optuna-postgres

rapids:
	docker compose up -d --build rapids

benchmark_db:
	docker compose up -d --build benchmark-postgres

optuna_db:
	docker compose up -d --build optuna-postgres

clear_benchmark_db:
	sudo rm -r ~/dev-postgres/chem/
	make benchmark_db

clear_optuna_db:
	sudo rm -r ~/dev-postgres/optuna/
	make optuna_db

########## DATASET PREPROCESSING ##########

process_all_smiles_datasets:
	docker run --rm -it \
		--runtime=nvidia \
		--gpus all \
		--shm-size=20g \
		--ulimit memlock=-1 \
		--ulimit stack=67108864 \
		--user $(id -u):$(id -g) \
		-e CUDA_VISIBLE_DEVICES="0,1" \
		-v "$(pwd)".:/chem-mrl \
		nvcr.io/nvidia/rapidsai/notebooks:24.12-cuda12.5-py3.12 \
		bash -c "pip install -r /chem-mrl/dataset/rapids-requirements.txt && python /chem-mrl/dataset/process_all_smiles_datasets.py"

# used to run scripts that depend on bionemo framework
run_bionemo:
	docker run --rm -it \
		--runtime=nvidia \
		--gpus 1 \
		--shm-size=20g \
		--ulimit memlock=-1 \
		--ulimit stack=67108864 \
		--user $(id -u):$(id -g) \
		-e CUDA_VISIBLE_DEVICES=0 \
		-v "$(pwd)".:/workspace/bionemo/chem-mrl \
		nvcr.io/nvidia/clara/bionemo-framework:1.10.1 \
		bash

genmol_dataset_gen:
	python dataset/generate_genmol_fingerprint_similarity_dataset.py \
	dataset.path=data/processed/train_pubchem_dataset_less_than_128_tokens.parquet

# https://docs.nvidia.com/nim/bionemo/genmol/latest/getting-started.html 
# https://docs.nvidia.com/launchpad/ai/base-command-coe/latest/bc-coe-docker-basics-step-02.html
# need ngc account and api key to download
run_genmol:
	@docker run --rm --name $$CONTAINER_NAME \
		--runtime=nvidia --gpus=all -e CUDA_VISIBLE_DEVICES=$$CUDA_VISIBLE_DEVICES  \
		--shm-size=20G \
		--ulimit memlock=-1 \
		--ulimit stack=67108864 \
		-p $${HOST_PORT:-8000}:8000 \
		-v ./models/genmol:/opt/nim/.cache \
		nvcr.io/nim/nvidia/genmol:1.0.0

run_multiple_genmol:
	bash scripts/run_genmol.bash 8

download_genmol:
	docker run -it --rm --runtime=nvidia --gpus=all -e NGC_API_KEY \
	-v ./models/genmol:/opt/nim/.cache \
	--entrypoint download-to-cache \
	nvcr.io/nim/nvidia/genmol:1.0.0 \
	-p a525212dd4373ace3be568a38b5d189a6fb866e007adf56e750ccd11e896b036

########## BENCHMARKING RECIPES ##########

run_seed_db:
	python scripts/seed_benchmarking_db.py --mode chem_mrl \
		--chem_mrl_model_name output/chem-2dmrl-functional-1-epoch \
		--chem_mrl_dimensions 768

run_seed_db_base:
	python scripts/seed_benchmarking_db.py --mode base

run_seed_db_test:
	python scripts/seed_benchmarking_db.py --mode test

run_benchmark_db:
	python scripts/benchmark_zinc_db.py --num_rows 1000 --knn_k 10 \
		--model_name output/chem-2dmrl-functional-1-epoch \
		--chem_mrl_dimensions 768


########## TRAINING RECIPES ##########

pubchem_pretraining:
	CUDA_VISIBLE_DEVICES=0 python scripts/train_chemberta_mlm.py \
	--dataset_path data/processed/train_pubchem_dataset_less_than_512_tokens.parquet \
	--eval_dataset_path data/processed/val_pubchem_dataset_less_than_512_tokens.parquet \
	--roberta_config_key chemberta-base --run_name chemberta_pubchem_10m --max_length 512 \
	--per_device_train_batch_size 128 --save_total_limit 15 --num_train_epochs 30 \
	--checkpoint_path training_output/chemberta_pubchem_10m/checkpoint-130000

full_dataset_pretraining:
	CUDA_VISIBLE_DEVICES=0 python scripts/train_chemberta_mlm.py \
	--dataset_path data/processed/train_full_ds_less_than_512_tokens.parquet \
	--eval_dataset_path data/processed/val_full_ds_less_than_512_tokens.parquet \
	--roberta_config_key chemberta-base --run_name chemberta_full_ds --max_length 512 \
	--per_device_train_batch_size 128 --save_total_limit 15 --num_train_epochs 10 \
	--checkpoint_path training_output/chemberta_full_ds/checkpoint-140000 \
	--eval_steps 2500 --save_steps 2500
#	--checkpoint_path models/chemberta_pubchem_10m/checkpoint-143000 --new_training_dataset \

chem_2d_mrl_training:
	CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/train_chem_mrl.py \
	model=chem_2d_mrl use_fused_adamw=True use_tf32=True use_amp=True \
	train_dataset_path=data/fp/train_pubchem_dataset_less_than_512_func_fp_sim_4096.parquet \
	val_dataset_path=data/fp/val_pubchem_dataset_less_than_512_func_fp_sim_4096.parquet \
	num_epochs=1 evaluation_steps=3500 checkpoint_save_steps=3500 \
	wandb.run_name=pubchem_ds_functional wandb.project_name=chem-mrl \
	n_dataloader_workers=12 scheduler=warmupcosine warmup_steps_percent=0.6 \
  train_batch_size=48 eval_batch_size=96 checkpoint_save_total_limit=20 pin_memory=True

chem_2d_mrl_training_test:
	CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/train_chem_mrl.py \
	model=chem_2d_mrl use_fused_adamw=True use_tf32=True use_amp=True \
	train_dataset_path=data/fp_genmol/train_pubchem_dataset_less_than_128_tokens_genmol_augmented.parquet \
	val_dataset_path=data/fp_genmol/train_pubchem_dataset_less_than_128_tokens_genmol_augmented.parquet \
	num_epochs=1 evaluation_steps=100 checkpoint_save_steps=100 \
	wandb.run_name=pubchem_ds_functional wandb.project_name=chem-mrl \
	n_dataloader_workers=12 scheduler=warmupcosine warmup_steps_percent=0.6 \
  train_batch_size=48 eval_batch_size=96 checkpoint_save_total_limit=20 pin_memory=True \
	smiles_a_column_name="smiles" smiles_b_column_name="reference_smiles" \


chem_2d_mrl_training_molmim_augmented:
	CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/train_chem_mrl.py \
	model=chem_2d_mrl model.model_name=/home/manny/source/chem-mrl/output/chem-2dmrl-functional-1-epoch \
	train_dataset_path=data/molmim_augmented_func_fp_sim_4096.parquet \
	val_dataset_path=data/fp/val_pubchem_dataset_less_than_512_fp_sim_4096.parquet \
	num_epochs=1 lr_base=1e-6 evaluation_steps=0 checkpoint_save_steps=0 \
	wandb.run_name=molmim_augmented_after_pubchem wandb.project_name=chem-mrl \
	n_dataloader_workers=6 scheduler=warmupcosine warmup_steps_percent=0.2 \
	use_fused_adamw=True use_tf32=True use_amp=True checkpoint_save_total_limit=10 \
	train_batch_size=48 eval_batch_size=96 seed=999 pin_memory=True
