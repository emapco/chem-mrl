.PHONY: docker

docker:
	docker-compose up -d --build zinc-postgres rapids_notebooks optuna-postgres

rapids:
	docker-compose up -d --build rapids_notebooks

zinc_db:
	docker-compose up -d --build zinc-postgres

optuna_db:
	docker-compose up -d --build optuna-postgres
