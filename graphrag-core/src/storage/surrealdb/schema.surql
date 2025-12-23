-- SurrealDB Schema for GraphRAG Storage
-- This schema defines tables for documents, entities, chunks, and relationships
-- Using SCHEMALESS to allow flexible Rust struct serialization

-- =============================================================================
-- Documents Table
-- =============================================================================
-- Stores document metadata and content

DEFINE TABLE document SCHEMALESS;
DEFINE INDEX idx_document_id ON document FIELDS id UNIQUE;

-- =============================================================================
-- Entities Table
-- =============================================================================
-- Stores extracted entities with support for graph relationships

DEFINE TABLE entity SCHEMALESS;
DEFINE INDEX idx_entity_id ON entity FIELDS id UNIQUE;
DEFINE INDEX idx_entity_type ON entity FIELDS entity_type;
DEFINE INDEX idx_entity_name ON entity FIELDS name;

-- =============================================================================
-- Chunks Table
-- =============================================================================
-- Stores text chunks with embeddings for vector similarity search

DEFINE TABLE chunk SCHEMALESS;
DEFINE INDEX idx_chunk_id ON chunk FIELDS id UNIQUE;
DEFINE INDEX idx_chunk_document ON chunk FIELDS document_id;

-- =============================================================================
-- Relationships Table (Graph Edges)
-- =============================================================================
-- Stores relationships between entities as graph edges
-- This enables graph traversal queries using SurrealDB's graph syntax

DEFINE TABLE relates_to TYPE RELATION IN entity OUT entity;
DEFINE FIELD relation_type ON relates_to TYPE string;
DEFINE FIELD confidence ON relates_to TYPE float;
DEFINE FIELD context ON relates_to TYPE array DEFAULT [];
DEFINE INDEX idx_relates_type ON relates_to FIELDS relation_type;

-- =============================================================================
-- Vector Table (Embeddings)
-- =============================================================================
-- Stores vector embeddings for similarity search
-- Supports multiple distance metrics: cosine, euclidean, manhattan

DEFINE TABLE vector SCHEMALESS;
DEFINE INDEX idx_vector_id ON vector FIELDS id UNIQUE;
DEFINE INDEX idx_vector_source ON vector FIELDS source_type, source_id;

-- MTREE index for approximate nearest neighbor search
-- Note: Adjust DIMENSION to match your embedding model:
--   384: all-MiniLM-L6-v2
--   768: all-mpnet-base-v2
--   1024: voyage-3-large, bge-large-en-v1.5
--   1536: text-embedding-3-small (OpenAI)
-- Uncomment and adjust as needed:
-- DEFINE INDEX idx_vector_embedding ON vector FIELDS embedding MTREE DIMENSION 384 TYPE F32 DIST COSINE;
