FROM pgvector/pgvector:0.8.0-pg17

RUN apt-get update && apt-get install -y postgresql-contrib

ADD init.sql /docker-entrypoint-initdb.d
