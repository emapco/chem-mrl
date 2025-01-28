.PHONY: docker, rapids, benchmark_db, optuna_db, clear_benchmark_db, clear_optuna_db

install:
	pip install -e .[dev]

install-pep:
	pip install .[dev] --use-pep517

docker:
	docker compose up -d --build benchmark-postgres rapids-notebooks optuna-postgres

rapids:
	docker compose up -d --build rapids-notebooks

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
