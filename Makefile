.PHONY: docker, rapids, benchmark_db, optuna_db, clear_benchmark_db, clear_optuna_db

install:
	pip install -e .[dev,benchmark,data]

install-pep:
	pip install .[dev] --use-pep517

docker:
	docker compose up -d --build benchmark-postgres optuna-postgres

rapids:
	docker compose up -d --build rapids-notebooks

bionemo:
	docker compose up -d --build bionemo

molmim:
	docker compose up -d --build molmim

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
		-v "$(pwd)".:/home/rapids/notebooks/chem-mrl \
		nvcr.io/nvidia/rapidsai/notebooks:24.08-cuda12.2-py3.11 \
		bash -c "pip install -r /home/rapids/notebooks/chem-mrl/dataset/requirements.txt && python /home/rapids/notebooks/chem-mrl/dataset/process_all_smiles_datasets.py"
