.PHONY: install install-pep lint release-testing docker rapids bionemo genmol \
				benchmark_db optuna_db clear_benchmark_db clear_optuna_db,\
				process_all_smiles_datasets run_bionemo \
				chem_2d_mrl_training pubchem_pretraining full_dataset_pretraining

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

docker:
	docker compose up -d --build benchmark-postgres optuna-postgres

rapids:
	docker compose up -d --build rapids-notebooks

bionemo:
	docker compose up -d --build bionemo

# https://docs.nvidia.com/launchpad/ai/base-command-coe/latest/bc-coe-docker-basics-step-02.html
# need ngc account and api key to download
genmol:
	sudo docker compose up -d --build genmol

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
		-e CUDA_VISIBLE_DEVICES="0" \
		-v "$(pwd)".:/workspace/bionemo/chem-mrl \
		nvcr.io/nvidia/clara/bionemo-framework:1.10.1 \
		bash
