CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS vector;

SET hnsw.ef_search = 100;
SET max_parallel_maintenance_workers = 7; -- plus leader
SET maintenance_work_mem = '16GB';

-- base transformer model prior to MRL training
CREATE TABLE base_768 (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    smiles TEXT NOT NULL,
    zinc_id TEXT NOT NULL,
    embedding VECTOR(768)
);
CREATE INDEX ON base_768 USING hnsw (embedding vector_cosine_ops) WITH (m = 32, ef_construction = 128);

-- create different tables for different embedding sizes and create a hnsw index on the embedding column
-- these tables and indicies are for comparing performance and accuracy of different embedding sizes

CREATE TABLE cme_768 (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    smiles TEXT NOT NULL,
    zinc_id TEXT NOT NULL,
    embedding VECTOR(768)
);
CREATE INDEX ON cme_768 USING hnsw (embedding vector_cosine_ops) WITH (m = 32, ef_construction = 128);

CREATE TABLE cme_512 (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    smiles TEXT NOT NULL,
    zinc_id TEXT NOT NULL,
    embedding VECTOR(512)
);
CREATE INDEX ON cme_512 USING hnsw (embedding vector_cosine_ops) WITH (m = 32, ef_construction = 128);

CREATE TABLE cme_256 (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    smiles TEXT NOT NULL,
    zinc_id TEXT NOT NULL,
    embedding VECTOR(256)
);
CREATE INDEX ON cme_256 USING hnsw (embedding vector_cosine_ops) WITH (m = 32, ef_construction = 128);

CREATE TABLE cme_128 (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    smiles TEXT NOT NULL,
    zinc_id TEXT NOT NULL,
    embedding VECTOR(128)
);
CREATE INDEX ON cme_128 USING hnsw (embedding vector_cosine_ops) WITH (m = 32, ef_construction = 128);

CREATE TABLE cme_64 (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    smiles TEXT NOT NULL,
    zinc_id TEXT NOT NULL,
    embedding VECTOR(64)
);
CREATE INDEX ON cme_64 USING hnsw (embedding vector_cosine_ops) WITH (m = 32, ef_construction = 128);

CREATE TABLE cme_32 (
  id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
  smiles TEXT NOT NULL,
  zinc_id TEXT NOT NULL,
  embedding VECTOR(32)
);
CREATE INDEX ON cme_32 USING hnsw (embedding vector_cosine_ops) WITH (m = 32, ef_construction = 128);

-- Table and index for morgen fingerprint embedding size 2000
-- 2000 is the maximum embedding size supported by pgvector
CREATE TABLE test_2000 (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    smiles TEXT NOT NULL,
    zinc_id TEXT NOT NULL,
    embedding VECTOR(2000)
);
CREATE INDEX ON test_2000 USING hnsw (embedding vector_cosine_ops) WITH (m = 32, ef_construction = 128);

CREATE TABLE test_768 (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    smiles TEXT NOT NULL,
    zinc_id TEXT NOT NULL,
    embedding VECTOR(768)
);
CREATE INDEX ON test_768 USING hnsw (embedding vector_cosine_ops) WITH (m = 32, ef_construction = 128);

CREATE TABLE test_512 (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    smiles TEXT NOT NULL,
    zinc_id TEXT NOT NULL,
    embedding VECTOR(512)
);
CREATE INDEX ON test_512 USING hnsw (embedding vector_cosine_ops) WITH (m = 32, ef_construction = 128);

CREATE TABLE test_256 (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    smiles TEXT NOT NULL,
    zinc_id TEXT NOT NULL,
    embedding VECTOR(256)
);
CREATE INDEX ON test_256 USING hnsw (embedding vector_cosine_ops) WITH (m = 32, ef_construction = 128);

CREATE TABLE test_128 (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    smiles TEXT NOT NULL,
    zinc_id TEXT NOT NULL,
    embedding VECTOR(128)
);
CREATE INDEX ON test_128 USING hnsw (embedding vector_cosine_ops) WITH (m = 32, ef_construction = 128);

CREATE TABLE test_64 (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    smiles TEXT NOT NULL,
    zinc_id TEXT NOT NULL,
    embedding VECTOR(64)
);
CREATE INDEX ON test_64 USING hnsw (embedding vector_cosine_ops) WITH (m = 32, ef_construction = 128);

CREATE TABLE test_32 (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    smiles TEXT NOT NULL,
    zinc_id TEXT NOT NULL,
    embedding VECTOR(32)
);
CREATE INDEX ON test_32 USING hnsw (embedding vector_cosine_ops) WITH (m = 32, ef_construction = 128);
