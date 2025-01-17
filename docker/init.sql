CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS vector;

SET hnsw.ef_search = 100;
SET max_parallel_maintenance_workers = 7; -- plus leader
SET maintenance_work_mem = '16GB';

-- create different tables for different embedding sizes
-- for testing the performance of full and truncated embedding sizes

-- Table and index for embedding size 768
CREATE TABLE cme_768 (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    smiles TEXT NOT NULL,
    zinc_id TEXT NOT NULL,
    embedding VECTOR(768)
);
CREATE INDEX ON cme_768 USING hnsw (embedding vector_cosine_ops) WITH (m = 32, ef_construction = 128);

-- Table and index for embedding size 512
CREATE TABLE cme_512 (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    smiles TEXT NOT NULL,
    zinc_id TEXT NOT NULL,
    embedding VECTOR(512)
);
CREATE INDEX ON cme_512 USING hnsw (embedding vector_cosine_ops) WITH (m = 32, ef_construction = 128);

-- Table and index for embedding size 384
CREATE TABLE cme_384 (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    smiles TEXT NOT NULL,
    zinc_id TEXT NOT NULL,
    embedding VECTOR(384)
);
CREATE INDEX ON cme_384 USING hnsw (embedding vector_cosine_ops) WITH (m = 32, ef_construction = 128);

-- Table and index for embedding size 256
CREATE TABLE cme_256 (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    smiles TEXT NOT NULL,
    zinc_id TEXT NOT NULL,
    embedding VECTOR(256)
);
CREATE INDEX ON cme_256 USING hnsw (embedding vector_cosine_ops) WITH (m = 32, ef_construction = 128);

-- Table and index for embedding size 128
CREATE TABLE cme_128 (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    smiles TEXT NOT NULL,
    zinc_id TEXT NOT NULL,
    embedding VECTOR(128)
);
CREATE INDEX ON cme_128 USING hnsw (embedding vector_cosine_ops) WITH (m = 32, ef_construction = 128);

-- Table and index for embedding size 64
CREATE TABLE cme_64 (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    smiles TEXT NOT NULL,
    zinc_id TEXT NOT NULL,
    embedding VECTOR(64)
);
CREATE INDEX ON cme_64 USING hnsw (embedding vector_cosine_ops) WITH (m = 32, ef_construction = 128);

-- Table and index for morgen fingerprint embedding size 2000
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

CREATE TABLE test_384 (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    smiles TEXT NOT NULL,
    zinc_id TEXT NOT NULL,
    embedding VECTOR(384)
);
CREATE INDEX ON test_384 USING hnsw (embedding vector_cosine_ops) WITH (m = 32, ef_construction = 128);

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

-- Table and index for morgen fingerprint embedding size 2000
CREATE TABLE val_2000 (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    smiles TEXT NOT NULL,
    zinc_id TEXT NOT NULL,
    embedding VECTOR(2000)
);
CREATE INDEX ON val_2000 USING hnsw (embedding vector_cosine_ops) WITH (m = 32, ef_construction = 128);

CREATE TABLE val_768 (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    smiles TEXT NOT NULL,
    zinc_id TEXT NOT NULL,
    embedding VECTOR(768)
);
CREATE INDEX ON val_768 USING hnsw (embedding vector_cosine_ops) WITH (m = 32, ef_construction = 128);

CREATE TABLE val_512 (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    smiles TEXT NOT NULL,
    zinc_id TEXT NOT NULL,
    embedding VECTOR(512)
);
CREATE INDEX ON val_512 USING hnsw (embedding vector_cosine_ops) WITH (m = 32, ef_construction = 128);

CREATE TABLE val_384 (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    smiles TEXT NOT NULL,
    zinc_id TEXT NOT NULL,
    embedding VECTOR(384)
);
CREATE INDEX ON val_384 USING hnsw (embedding vector_cosine_ops) WITH (m = 32, ef_construction = 128);

CREATE TABLE val_256 (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    smiles TEXT NOT NULL,
    zinc_id TEXT NOT NULL,
    embedding VECTOR(256)
);
CREATE INDEX ON val_256 USING hnsw (embedding vector_cosine_ops) WITH (m = 32, ef_construction = 128);

CREATE TABLE val_128 (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    smiles TEXT NOT NULL,
    zinc_id TEXT NOT NULL,
    embedding VECTOR(128)
);
CREATE INDEX ON val_128 USING hnsw (embedding vector_cosine_ops) WITH (m = 32, ef_construction = 128);

CREATE TABLE val_64 (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    smiles TEXT NOT NULL,
    zinc_id TEXT NOT NULL,
    embedding VECTOR(64)
);
CREATE INDEX ON val_64 USING hnsw (embedding vector_cosine_ops) WITH (m = 32, ef_construction = 128);
