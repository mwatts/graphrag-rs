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
