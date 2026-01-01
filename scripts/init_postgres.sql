-- =============================================================================
-- PostgreSQL Initialization Script for RAG System
-- =============================================================================
-- This script sets up the database schema for the Enterprise RAG System
-- with pgvector extension for vector similarity search

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =============================================================================
-- Documents Table
-- Stores metadata and tree structure for indexed documents
-- =============================================================================
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title VARCHAR(500) NOT NULL,
    file_path VARCHAR(1000) NOT NULL UNIQUE,
    file_type VARCHAR(50) NOT NULL,
    summary TEXT,
    metadata JSONB DEFAULT '{}',
    tree_structure JSONB,
    -- Indexing metadata
    qdrant_document_id VARCHAR(255),
    embedding_status VARCHAR(50) DEFAULT 'pending',
    -- Tracking
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    version INT DEFAULT 1,
    is_active BOOLEAN DEFAULT TRUE
);

-- =============================================================================
-- Sections Table
-- Stores hierarchical sections from document tree structure
-- =============================================================================
CREATE TABLE IF NOT EXISTS sections (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    parent_section_id UUID REFERENCES sections(id) ON DELETE CASCADE,
    heading VARCHAR(500) NOT NULL,
    level INT NOT NULL CHECK (level BETWEEN 1 AND 6),
    section_path VARCHAR(1000),
    summary TEXT,
    content TEXT,
    metadata JSONB DEFAULT '{}',
    -- Indexing metadata
    qdrant_section_id VARCHAR(255),
    position INT,
    -- Tracking
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- =============================================================================
-- Chunks Table
-- Stores individual chunks for precise retrieval
-- =============================================================================
CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    section_id UUID REFERENCES sections(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    token_count INT,
    chunk_position INT,
    metadata JSONB DEFAULT '{}',
    -- Indexing metadata
    qdrant_chunk_id VARCHAR(255),
    position INT,
    -- Tracking
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- =============================================================================
-- Cross-References Table
-- Stores relationships between documents
-- =============================================================================
CREATE TABLE IF NOT EXISTS cross_references (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_doc_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    target_doc_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    relation_type VARCHAR(50) NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(source_doc_id, target_doc_id, relation_type)
);

-- =============================================================================
-- Queries Table
-- Stores query history for analytics and improvement
-- =============================================================================
CREATE TABLE IF NOT EXISTS queries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_text TEXT NOT NULL,
    rewritten_query TEXT,
    query_type VARCHAR(50),
    user_id VARCHAR(255),
    session_id VARCHAR(255),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- =============================================================================
-- Query Results Table
-- Stores retrieval results for analytics
-- =============================================================================
CREATE TABLE IF NOT EXISTS query_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_id UUID NOT NULL REFERENCES queries(id) ON DELETE CASCADE,
    document_id UUID REFERENCES documents(id) ON DELETE SET NULL,
    section_id UUID REFERENCES sections(id) ON DELETE SET NULL,
    chunk_id UUID REFERENCES chunks(id) ON DELETE SET NULL,
    score FLOAT,
    position INT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- =============================================================================
-- Indexes for performance
-- =============================================================================

-- Documents indexes
CREATE INDEX IF NOT EXISTS idx_documents_file_path ON documents(file_path);
CREATE INDEX IF NOT EXISTS idx_documents_file_type ON documents(file_type);
CREATE INDEX IF NOT EXISTS idx_documents_embedding_status ON documents(embedding_status);
CREATE INDEX IF NOT EXISTS idx_documents_title_trgm ON documents USING gin(title gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_documents_metadata ON documents USING gin(metadata);
CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at DESC);

-- Sections indexes
CREATE INDEX IF NOT EXISTS idx_sections_document ON sections(document_id);
CREATE INDEX IF NOT EXISTS idx_sections_parent ON sections(parent_section_id);
CREATE INDEX IF NOT EXISTS idx_sections_level ON sections(level);
CREATE INDEX IF NOT EXISTS idx_sections_heading_trgm ON sections USING gin(heading gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_sections_metadata ON sections USING gin(metadata);

-- Chunks indexes
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_section ON chunks(section_id);
CREATE INDEX IF NOT EXISTS idx_chunks_content_trgm ON chunks USING gin(content gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_chunks_metadata ON chunks USING gin(metadata);

-- Cross-references indexes
CREATE INDEX IF NOT EXISTS idx_cross_references_source ON cross_references(source_doc_id);
CREATE INDEX IF NOT EXISTS idx_cross_references_target ON cross_references(target_doc_id);
CREATE INDEX IF NOT EXISTS idx_cross_references_type ON cross_references(relation_type);

-- Query indexes
CREATE INDEX IF NOT EXISTS idx_queries_user ON queries(user_id);
CREATE INDEX IF NOT EXISTS idx_queries_session ON queries(session_id);
CREATE INDEX IF NOT EXISTS idx_queries_created_at ON queries(created_at DESC);

-- Query results indexes
CREATE INDEX IF NOT EXISTS idx_query_results_query ON query_results(query_id);
CREATE INDEX IF NOT EXISTS idx_query_results_document ON query_results(document_id);

-- =============================================================================
-- Functions for automatic timestamp updates
-- =============================================================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for documents table
DROP TRIGGER IF EXISTS update_documents_updated_at ON documents;
CREATE TRIGGER update_documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- Helper function to check database health
-- =============================================================================
CREATE OR REPLACE FUNCTION health_check()
RETURNS JSONB AS $$
DECLARE
    result JSONB;
    doc_count INT;
    section_count INT;
    chunk_count INT;
BEGIN
    SELECT COUNT(*) INTO doc_count FROM documents WHERE is_active = TRUE;
    SELECT COUNT(*) INTO section_count FROM sections;
    SELECT COUNT(*) INTO chunk_count FROM chunks;

    result = jsonb_build_object(
        'status', 'healthy',
        'timestamp', NOW(),
        'documents', doc_count,
        'sections', section_count,
        'chunks', chunk_count
    );

    RETURN result;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- Grant permissions (adjust user as needed)
-- =============================================================================
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO raguser;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO raguser;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO raguser;

-- =============================================================================
-- Initialization complete
-- =============================================================================
DO $$
BEGIN
    RAISE NOTICE 'Database schema initialized successfully for RAG System';
END $$;
