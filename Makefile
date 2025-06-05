.PHONY: install install-pep lint release-testing docker \
				rapids benchmark_db optuna_db clear_benchmark_db clear_optuna_db \
				process_all_smiles_datasets run_bionemo \
				genmol_dataset_gen run_genmol run_multiple_genmol download_genmol \
        run_seed_db run_seed_db_base run_seed_db_test run_benchmark_db

########## LIBRARY RECIPES ##########

install:
	pip install -e .[dev,benchmark,data]

install-pep:
	pip install .[dev,benchmark,data] --use-pep517

lint:
	ruff check chem_mrl --fix --config pyproject.toml
	ruff format chem_mrl --config pyproject.toml
	ruff analyze graph --config pyproject.toml
	ruff clean

release-testing: lint
	pip uninstall chem_mrl -y
	python -m build
	pip install dist/*.whl
	rm -r dist/
	(unset CUDA_VISIBLE_DEVICES; pytest tests)
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
	bash scripts/run_genmol.bash 4

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
