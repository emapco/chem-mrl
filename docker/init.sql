CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS vector;

-- base transformer model prior to MRL training
CREATE TABLE IF NOT EXISTS base_768 (
    zinc_id TEXT NOT NULL PRIMARY KEY,
    smiles TEXT NOT NULL,
    embedding halfvec(768)
);
CREATE INDEX IF NOT EXISTS base_768_embedding_idx ON base_768 USING hnsw ((embedding::halfvec(768)) halfvec_cosine_ops) WITH (m = 16, ef_construction = 96);

-- these tables and indicies are for comparing performance and accuracy of different embedding sizes

CREATE TABLE IF NOT EXISTS cme_768 (
    zinc_id TEXT NOT NULL PRIMARY KEY,
    smiles TEXT NOT NULL,
    embedding halfvec(768)
);
CREATE INDEX IF NOT EXISTS cme_768_embedding_idx ON cme_768 USING hnsw ((embedding::halfvec(768)) halfvec_cosine_ops) WITH (m = 16, ef_construction = 96);

CREATE TABLE IF NOT EXISTS cme_512 (
    zinc_id TEXT NOT NULL PRIMARY KEY,
    smiles TEXT NOT NULL,
    embedding halfvec(512)
);
CREATE INDEX IF NOT EXISTS cme_512_embedding_idx ON cme_512 USING hnsw ((embedding::halfvec(512)) halfvec_cosine_ops) WITH (m = 16, ef_construction = 96);

CREATE TABLE IF NOT EXISTS cme_256 (
    zinc_id TEXT NOT NULL PRIMARY KEY,
    smiles TEXT NOT NULL,
    embedding halfvec(256)
);
CREATE INDEX IF NOT EXISTS cme_256_embedding_idx ON cme_256 USING hnsw ((embedding::halfvec(256)) halfvec_cosine_ops) WITH (m = 16, ef_construction = 96);

CREATE TABLE IF NOT EXISTS cme_128 (
    zinc_id TEXT NOT NULL PRIMARY KEY,
    smiles TEXT NOT NULL,
    embedding halfvec(128)
);
CREATE INDEX IF NOT EXISTS cme_128_embedding_idx ON cme_128 USING hnsw ((embedding::halfvec(128)) halfvec_cosine_ops) WITH (m = 16, ef_construction = 96);

CREATE TABLE IF NOT EXISTS cme_64 (
    zinc_id TEXT NOT NULL PRIMARY KEY,
    smiles TEXT NOT NULL,
    embedding halfvec(64)
);
CREATE INDEX IF NOT EXISTS cme_64_embedding_idx ON cme_64 USING hnsw ((embedding::halfvec(64)) halfvec_cosine_ops) WITH (m = 16, ef_construction = 96);

CREATE TABLE IF NOT EXISTS cme_32 (
    zinc_id TEXT NOT NULL PRIMARY KEY,
    smiles TEXT NOT NULL,
    embedding halfvec(32)
);
CREATE INDEX IF NOT EXISTS cme_32_embedding_idx ON cme_32 USING hnsw ((embedding::halfvec(32)) halfvec_cosine_ops) WITH (m = 16, ef_construction = 96);

-- Table and index for morgen fingerprint embedding size 2048
-- 2048 is the maximum embedding size supported by pgvector for halfvec
CREATE TABLE IF NOT EXISTS test_2048 (
    zinc_id TEXT NOT NULL PRIMARY KEY,
    smiles TEXT NOT NULL,
    embedding halfvec(2048)
);
CREATE INDEX IF NOT EXISTS test_2048_embedding_idx ON test_2048 USING hnsw ((embedding::halfvec(2048)) halfvec_cosine_ops) WITH (m = 16, ef_construction = 96);

CREATE TABLE IF NOT EXISTS test_768 (
    zinc_id TEXT NOT NULL PRIMARY KEY,
    smiles TEXT NOT NULL,
    embedding halfvec(768)
);
CREATE INDEX IF NOT EXISTS test_768_embedding_idx ON test_768 USING hnsw ((embedding::halfvec(768)) halfvec_cosine_ops) WITH (m = 16, ef_construction = 96);

CREATE TABLE IF NOT EXISTS test_512 (
    zinc_id TEXT NOT NULL PRIMARY KEY,
    smiles TEXT NOT NULL,
    embedding halfvec(512)
);
CREATE INDEX IF NOT EXISTS test_512_embedding_idx ON test_512 USING hnsw ((embedding::halfvec(512)) halfvec_cosine_ops) WITH (m = 16, ef_construction = 96);

CREATE TABLE IF NOT EXISTS test_256 (
    zinc_id TEXT NOT NULL PRIMARY KEY,
    smiles TEXT NOT NULL,
    embedding halfvec(256)
);
CREATE INDEX IF NOT EXISTS test_256_embedding_idx ON test_256 USING hnsw ((embedding::halfvec(256)) halfvec_cosine_ops) WITH (m = 16, ef_construction = 96);

CREATE TABLE IF NOT EXISTS test_128 (
    zinc_id TEXT NOT NULL PRIMARY KEY,
    smiles TEXT NOT NULL,
    embedding halfvec(128)
);
CREATE INDEX IF NOT EXISTS test_128_embedding_idx ON test_128 USING hnsw ((embedding::halfvec(128)) halfvec_cosine_ops) WITH (m = 16, ef_construction = 96);

CREATE TABLE IF NOT EXISTS test_64 (
    zinc_id TEXT NOT NULL PRIMARY KEY,
    smiles TEXT NOT NULL,
    embedding halfvec(64)
);
CREATE INDEX IF NOT EXISTS test_64_embedding_idx ON test_64 USING hnsw ((embedding::halfvec(64)) halfvec_cosine_ops) WITH (m = 16, ef_construction = 96);

CREATE TABLE IF NOT EXISTS test_32 (
    zinc_id TEXT NOT NULL PRIMARY KEY,
    smiles TEXT NOT NULL,
    embedding halfvec(32)
);
CREATE INDEX IF NOT EXISTS test_32_embedding_idx ON test_32 USING hnsw ((embedding::halfvec(32)) halfvec_cosine_ops) WITH (m = 16, ef_construction = 96);
