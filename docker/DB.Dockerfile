FROM pgvector/pgvector:0.6.2-pg16

RUN apt-get update && apt-get install -y postgresql-contrib

ADD init.sql /docker-entrypoint-initdb.d
