.PHONY: docker

docker:
	docker-compose up -d --build zinc-postgres rapids_notebooks
