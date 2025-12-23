# SurrealDB Storage Integration Design

## Overview

This document describes the design for integrating SurrealDB as a storage backend for graphrag-core. SurrealDB is particularly well-suited for GraphRAG workloads due to its native multi-model support (document, graph, relational, and vector), which aligns with the core data structures: documents, entities, chunks, and relationships.

**Implementation Status**: The `AsyncStorage` trait implementation is complete with 16 unit tests and 13 integration tests passing.

## Scope

### Completed (Phase 1)
1. `AsyncStorage` trait implementation for documents, entities, and chunks
2. Configuration for SurrealDB connection (memory, RocksDB, WebSocket, HTTP)
3. Feature flag integration for optional compilation
4. Schema definition with indexes
5. Comprehensive test coverage

### Future Work (Phase 2 & 3)
- **Phase 2**: Vector store implementation (`AsyncVectorStore` trait) - See [Vector Store Design](#vector-store-design)
- **Phase 3**: Graph store implementation (`AsyncGraphStore` trait) - See [Graph Store Design](#graph-store-design)

## Feature Flag

In `graphrag-core/Cargo.toml`:

```toml
[features]
# SurrealDB storage backend
surrealdb-storage = ["surrealdb", "async"]

[dependencies]
# SurrealDB (optional, for surrealdb-storage feature)
surrealdb = { version = "2.3", optional = true, default-features = false, features = ["kv-mem", "kv-rocksdb"] }
```

Feature notes:
- `kv-mem`: In-memory storage for development/testing
- `kv-rocksdb`: Persistent storage for production single-node deployments
- Remote features (`protocol-ws`, `protocol-http`) can be added for distributed deployments

## Core Types Mapping

### GraphRAG Types to SurrealDB Tables

| GraphRAG Type | SurrealDB Table | Model Type |
|---------------|-----------------|------------|
| `Document` | `document` | Document (flexible metadata) |
| `Entity` | `entity` | Document + Graph (relationships) |
| `TextChunk` | `chunk` | Document (with vector field) |
| `Relationship` | `relates_to` | Graph (RELATION type) |

### Record ID Strategy

SurrealDB uses composite record IDs (`table:id`). GraphRAG IDs are mapped directly:

```rust
// DocumentId("doc123") -> document:doc123
// EntityId("ent456") -> entity:ent456  
// ChunkId("chunk789") -> chunk:chunk789
```

**Implementation Note**: The actual implementation uses `type::thing('table', $id)` for record references and `meta::id(id)` to convert SurrealDB's `Thing` type back to strings for retrieval.

## Schema Definition

The schema uses `SCHEMALESS` tables to allow flexible Rust struct serialization:

```sql
-- graphrag-core/src/storage/surrealdb/schema.surql

-- Documents Table
DEFINE TABLE document SCHEMALESS;
DEFINE INDEX idx_document_id ON document FIELDS id UNIQUE;

-- Entities Table
DEFINE TABLE entity SCHEMALESS;
DEFINE INDEX idx_entity_id ON entity FIELDS id UNIQUE;
DEFINE INDEX idx_entity_type ON entity FIELDS entity_type;
DEFINE INDEX idx_entity_name ON entity FIELDS name;

-- Chunks Table
DEFINE TABLE chunk SCHEMALESS;
DEFINE INDEX idx_chunk_id ON chunk FIELDS id UNIQUE;
DEFINE INDEX idx_chunk_document ON chunk FIELDS document_id;

-- Relationships as Graph Edges
DEFINE TABLE relates_to TYPE RELATION IN entity OUT entity;
DEFINE FIELD relation_type ON relates_to TYPE string;
DEFINE FIELD confidence ON relates_to TYPE float;
DEFINE FIELD context ON relates_to TYPE array DEFAULT [];
DEFINE INDEX idx_relates_type ON relates_to FIELDS relation_type;
```

**Design Decision**: `SCHEMALESS` was chosen over `SCHEMAFULL` because:
1. Rust structs with newtype wrappers (e.g., `DocumentId(String)`) require flexible serialization
2. The `#[serde(transparent)]` attribute on ID types ensures they serialize as plain strings
3. Allows for future field additions without schema migrations

## Implementation

### Module Structure

```
graphrag-core/src/
├── storage/
│   ├── mod.rs           # Storage module exports with feature gate
│   └── surrealdb/
│       ├── mod.rs       # SurrealDB submodule exports
│       ├── storage.rs   # AsyncStorage implementation
│       ├── config.rs    # Configuration types
│       ├── error.rs     # SurrealDB-specific error types
│       └── schema.surql # Schema definitions
```

### Configuration

```rust
// graphrag-core/src/storage/surrealdb/config.rs

/// SurrealDB connection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurrealDbConfig {
    /// Connection endpoint (mem://, rocksdb://, ws://, http://)
    pub endpoint: String,
    /// Namespace for data isolation
    pub namespace: String,
    /// Database name within namespace
    pub database: String,
    /// Optional authentication credentials
    pub credentials: Option<SurrealDbCredentials>,
    /// Whether to initialize schema on connect
    pub auto_init_schema: bool,
}

impl SurrealDbConfig {
    pub fn memory() -> Self { /* ... */ }
    pub fn rocksdb(path: impl Into<String>) -> Self { /* ... */ }
    pub fn websocket(host: impl Into<String>, port: u16) -> Self { /* ... */ }
    pub fn http(host: impl Into<String>, port: u16) -> Self { /* ... */ }
    
    // Builder methods
    pub fn with_namespace(self, namespace: impl Into<String>) -> Self { /* ... */ }
    pub fn with_database(self, database: impl Into<String>) -> Self { /* ... */ }
    pub fn with_credentials(self, username: impl Into<String>, password: impl Into<String>) -> Self { /* ... */ }
    pub fn without_auto_schema(self) -> Self { /* ... */ }
}
```

### Error Types

```rust
// graphrag-core/src/storage/surrealdb/error.rs

#[derive(Debug, Error)]
pub enum SurrealDbStorageError {
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),
    
    #[error("Authentication failed: {0}")]
    AuthenticationFailed(String),
    
    #[error("Schema initialization failed: {0}")]
    SchemaInitFailed(String),
    
    #[error("Query execution failed: {0}")]
    QueryFailed(String),
    
    #[error("Serialization error: {0}")]
    SerializationError(String),
}

impl From<SurrealDbStorageError> for GraphRAGError { /* ... */ }
```

### Storage Implementation

The implementation uses raw SQL queries with bind parameters to handle SurrealDB's type system:

```rust
// graphrag-core/src/storage/surrealdb/storage.rs

pub struct SurrealDbStorage {
    db: Arc<Surreal<Any>>,
    config: SurrealDbConfig,
}

impl SurrealDbStorage {
    pub async fn new(config: SurrealDbConfig) -> Result<Self> {
        // Uses surrealdb::engine::any::connect for dynamic endpoint parsing
        let db = surrealdb::engine::any::connect(&config.endpoint).await?;
        db.use_ns(&config.namespace).use_db(&config.database).await?;
        // ... authentication and schema initialization
    }
    
    pub fn client(&self) -> &Surreal<Any> { &self.db }
    pub fn client_arc(&self) -> Arc<Surreal<Any>> { Arc::clone(&self.db) }
}

#[async_trait]
impl AsyncStorage for SurrealDbStorage {
    type Entity = Entity;
    type Document = Document;
    type Chunk = TextChunk;
    type Error = GraphRAGError;

    async fn store_entity(&mut self, entity: Self::Entity) -> Result<String> {
        let id = entity.id.to_string();
        // Serialize to JSON first to handle newtype wrappers
        let json_str = serde_json::to_string(&entity)?;
        self.db
            .query("UPSERT type::thing('entity', $id) CONTENT $data")
            .bind(("id", id.clone()))
            .bind(("data", serde_json::from_str::<serde_json::Value>(&json_str)?))
            .await?;
        Ok(id)
    }

    async fn retrieve_entity(&self, id: &str) -> Result<Option<Self::Entity>> {
        // Use meta::id(id) to convert SurrealDB Thing back to string
        let mut response = self.db
            .query("SELECT *, meta::id(id) as id FROM type::thing('entity', $id)")
            .bind(("id", id.to_string()))
            .await?;
        let results: Vec<serde_json::Value> = response.take(0)?;
        // ... parse and return
    }
    
    // Similar implementations for documents, chunks, batch operations...
}
```

**Key Implementation Details**:

1. **Serialization Workaround**: SurrealDB's Rust SDK has limitations with complex types. We serialize to `serde_json::Value` first, then pass to SurrealDB.

2. **ID Handling**: The `meta::id(id)` function extracts the string ID from SurrealDB's `Thing` type during retrieval.

3. **Upsert Semantics**: All store operations use `UPSERT` for idempotent writes.

4. **Transactions**: Batch operations use `BEGIN TRANSACTION` / `COMMIT TRANSACTION` for atomicity.

## Testing Strategy

### Unit Tests (16 tests)
- Configuration tests: memory, rocksdb, websocket, http, builder methods
- Error conversion tests
- Storage CRUD operations with in-memory backend

### Integration Tests (13 tests)
Located in `graphrag-core/tests/surrealdb_storage_integration.rs`:

1. **Lifecycle tests**: Entity, document, chunk full CRUD
2. **Cross-reference integrity**: Relationships between types
3. **Batch operations**: Transaction behavior
4. **Concurrent access**: Interleaved read/write patterns
5. **Special characters**: Unicode and escaping
6. **Schema auto-init**: Automatic table creation
7. **Isolated databases**: Namespace/database separation

---

## Vector Store Design

### Overview

The `AsyncVectorStore` trait provides vector similarity search for embeddings. SurrealDB supports native vector types and similarity functions, making it ideal for semantic search.

### Trait Interface

```rust
#[async_trait]
pub trait AsyncVectorStore: Send + Sync {
    type Error: std::error::Error + Send + Sync + 'static;

    async fn add_vector(&mut self, id: String, vector: Vec<f32>, metadata: VectorMetadata) -> Result<()>;
    async fn add_vectors_batch(&mut self, vectors: VectorBatch) -> Result<()>;
    async fn search(&self, query_vector: &[f32], k: usize) -> Result<Vec<SearchResult>>;
    async fn search_with_threshold(&self, query_vector: &[f32], k: usize, threshold: f32) -> Result<Vec<SearchResult>>;
    async fn remove_vector(&mut self, id: &str) -> Result<bool>;
    async fn len(&self) -> usize;
    async fn is_empty(&self) -> bool;
    async fn build_index(&mut self) -> Result<()>;
}
```

### Schema Extensions

```sql
-- Vector table for embeddings with HNSW index support
DEFINE TABLE vector SCHEMALESS;
DEFINE FIELD id ON vector TYPE string;
DEFINE FIELD embedding ON vector TYPE array<float>;
DEFINE FIELD metadata ON vector FLEXIBLE TYPE object;
DEFINE FIELD source_type ON vector TYPE string;  -- "entity", "chunk", "document"
DEFINE FIELD source_id ON vector TYPE string;
DEFINE FIELD created_at ON vector TYPE datetime DEFAULT time::now();

DEFINE INDEX idx_vector_id ON vector FIELDS id UNIQUE;
DEFINE INDEX idx_vector_source ON vector FIELDS source_type, source_id;

-- HNSW index for approximate nearest neighbor search
-- Note: SurrealDB vector index syntax may vary by version
DEFINE INDEX idx_vector_embedding ON vector FIELDS embedding 
    MTREE DIMENSION 384 
    TYPE F32 
    DIST COSINE;
```

### Implementation

```rust
// graphrag-core/src/storage/surrealdb/vector.rs

pub struct SurrealDbVectorStore {
    db: Arc<Surreal<Any>>,
    config: SurrealDbVectorConfig,
}

#[derive(Debug, Clone)]
pub struct SurrealDbVectorConfig {
    /// Dimension of vectors (must match embedding model)
    pub dimension: usize,
    /// Distance metric: "cosine", "euclidean", "manhattan"
    pub distance_metric: DistanceMetric,
    /// Table name for vectors (default: "vector")
    pub table_name: String,
    /// Whether to auto-build index after batch inserts
    pub auto_index: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum DistanceMetric {
    Cosine,
    Euclidean,
    Manhattan,
}

impl DistanceMetric {
    fn as_surql_function(&self) -> &'static str {
        match self {
            DistanceMetric::Cosine => "vector::similarity::cosine",
            DistanceMetric::Euclidean => "vector::distance::euclidean",
            DistanceMetric::Manhattan => "vector::distance::manhattan",
        }
    }
}

#[async_trait]
impl AsyncVectorStore for SurrealDbVectorStore {
    type Error = GraphRAGError;

    async fn add_vector(
        &mut self,
        id: String,
        vector: Vec<f32>,
        metadata: VectorMetadata,
    ) -> Result<()> {
        // Validate dimension
        if vector.len() != self.config.dimension {
            return Err(GraphRAGError::Validation {
                message: format!(
                    "Vector dimension mismatch: expected {}, got {}",
                    self.config.dimension,
                    vector.len()
                ),
            });
        }

        let record = VectorRecord {
            id: id.clone(),
            embedding: vector,
            metadata,
            source_type: None,
            source_id: None,
        };

        let json_value = serde_json::to_value(&record)?;
        self.db
            .query("UPSERT type::thing($table, $id) CONTENT $data")
            .bind(("table", &self.config.table_name))
            .bind(("id", id))
            .bind(("data", json_value))
            .await?;

        Ok(())
    }

    async fn add_vectors_batch(&mut self, vectors: VectorBatch) -> Result<()> {
        self.db.query("BEGIN TRANSACTION").await?;

        for (id, vector, metadata) in vectors {
            if let Err(e) = self.add_vector(id, vector, metadata).await {
                self.db.query("CANCEL TRANSACTION").await.ok();
                return Err(e);
            }
        }

        self.db.query("COMMIT TRANSACTION").await?;
        
        if self.config.auto_index {
            self.build_index().await?;
        }

        Ok(())
    }

    async fn search(&self, query_vector: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        let distance_fn = self.config.distance_metric.as_surql_function();
        
        // SurrealDB vector similarity search
        let query = format!(
            "SELECT id, {distance_fn}(embedding, $query) AS score, metadata \
             FROM {table} \
             ORDER BY score DESC \
             LIMIT $k",
            distance_fn = distance_fn,
            table = self.config.table_name
        );

        let mut response = self.db
            .query(&query)
            .bind(("query", query_vector.to_vec()))
            .bind(("k", k))
            .await?;

        let results: Vec<VectorSearchResult> = response.take(0)?;
        
        Ok(results
            .into_iter()
            .map(|r| SearchResult {
                id: r.id,
                distance: 1.0 - r.score, // Convert similarity to distance
                metadata: r.metadata,
            })
            .collect())
    }

    async fn search_with_threshold(
        &self,
        query_vector: &[f32],
        k: usize,
        threshold: f32,
    ) -> Result<Vec<SearchResult>> {
        let distance_fn = self.config.distance_metric.as_surql_function();
        let min_score = 1.0 - threshold; // Convert distance threshold to similarity
        
        let query = format!(
            "SELECT id, {distance_fn}(embedding, $query) AS score, metadata \
             FROM {table} \
             WHERE {distance_fn}(embedding, $query) >= $min_score \
             ORDER BY score DESC \
             LIMIT $k",
            distance_fn = distance_fn,
            table = self.config.table_name
        );

        let mut response = self.db
            .query(&query)
            .bind(("query", query_vector.to_vec()))
            .bind(("k", k))
            .bind(("min_score", min_score))
            .await?;

        let results: Vec<VectorSearchResult> = response.take(0)?;
        
        Ok(results
            .into_iter()
            .map(|r| SearchResult {
                id: r.id,
                distance: 1.0 - r.score,
                metadata: r.metadata,
            })
            .collect())
    }

    async fn remove_vector(&mut self, id: &str) -> Result<bool> {
        let result = self.db
            .query("DELETE type::thing($table, $id) RETURN BEFORE")
            .bind(("table", &self.config.table_name))
            .bind(("id", id))
            .await?;
        
        // Check if anything was deleted
        Ok(!result.is_empty())
    }

    async fn len(&self) -> usize {
        let query = format!("SELECT count() FROM {} GROUP ALL", self.config.table_name);
        let mut response = self.db.query(&query).await.unwrap_or_default();
        
        #[derive(serde::Deserialize)]
        struct CountResult { count: usize }
        
        response
            .take::<Vec<CountResult>>(0)
            .ok()
            .and_then(|v| v.into_iter().next())
            .map(|r| r.count)
            .unwrap_or(0)
    }

    async fn build_index(&mut self) -> Result<()> {
        // SurrealDB builds indexes automatically
        // This method can be used to force index rebuild if needed
        let query = format!(
            "REBUILD INDEX idx_vector_embedding ON {}",
            self.config.table_name
        );
        self.db.query(&query).await?;
        Ok(())
    }
}

// Helper types
#[derive(Debug, Serialize, Deserialize)]
struct VectorRecord {
    id: String,
    embedding: Vec<f32>,
    metadata: Option<HashMap<String, String>>,
    source_type: Option<String>,
    source_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct VectorSearchResult {
    id: String,
    score: f32,
    metadata: Option<HashMap<String, String>>,
}
```

### Usage Example

```rust
use graphrag_core::storage::surrealdb::{
    SurrealDbConfig, SurrealDbVectorStore, SurrealDbVectorConfig, DistanceMetric,
};

// Create vector store with 384-dimensional embeddings (e.g., for all-MiniLM-L6-v2)
let config = SurrealDbVectorConfig {
    dimension: 384,
    distance_metric: DistanceMetric::Cosine,
    table_name: "chunk_vectors".to_string(),
    auto_index: true,
};

let db_config = SurrealDbConfig::rocksdb("./data/vectors");
let mut vector_store = SurrealDbVectorStore::new(db_config, config).await?;

// Add vectors
vector_store.add_vector(
    "chunk_1".to_string(),
    embedding_model.embed("Some text").await?,
    Some(HashMap::from([("doc_id".to_string(), "doc_1".to_string())])),
).await?;

// Search
let query_embedding = embedding_model.embed("search query").await?;
let results = vector_store.search(&query_embedding, 10).await?;

for result in results {
    println!("ID: {}, Distance: {}", result.id, result.distance);
}
```

---

## Graph Store Design

### Overview

The `AsyncGraphStore` trait provides graph operations for the knowledge graph. SurrealDB's native graph support with `RELATE` statements and graph traversal syntax makes it ideal for relationship management.

### Trait Interface

```rust
#[async_trait]
pub trait AsyncGraphStore: Send + Sync {
    type Node: Send + Sync;
    type Edge: Send + Sync;
    type Error: std::error::Error + Send + Sync + 'static;

    async fn add_node(&mut self, node: Self::Node) -> Result<String>;
    async fn add_nodes_batch(&mut self, nodes: Vec<Self::Node>) -> Result<Vec<String>>;
    async fn add_edge(&mut self, from_id: &str, to_id: &str, edge: Self::Edge) -> Result<String>;
    async fn add_edges_batch(&mut self, edges: Vec<(String, String, Self::Edge)>) -> Result<Vec<String>>;
    async fn find_nodes(&self, criteria: &str) -> Result<Vec<Self::Node>>;
    async fn get_neighbors(&self, node_id: &str) -> Result<Vec<Self::Node>>;
    async fn traverse(&self, start_id: &str, max_depth: usize) -> Result<Vec<Self::Node>>;
    async fn stats(&self) -> GraphStats;
}
```

### Schema (Already Defined)

The graph schema uses SurrealDB's `RELATION` table type:

```sql
-- Relationships as graph edges (already in schema.surql)
DEFINE TABLE relates_to TYPE RELATION IN entity OUT entity;
DEFINE FIELD relation_type ON relates_to TYPE string;
DEFINE FIELD confidence ON relates_to TYPE float;
DEFINE FIELD context ON relates_to TYPE array DEFAULT [];
DEFINE INDEX idx_relates_type ON relates_to FIELDS relation_type;
```

### Implementation

```rust
// graphrag-core/src/storage/surrealdb/graph.rs

use crate::core::{Entity, Relationship};

pub struct SurrealDbGraphStore {
    db: Arc<Surreal<Any>>,
    config: SurrealDbGraphConfig,
}

#[derive(Debug, Clone)]
pub struct SurrealDbGraphConfig {
    /// Node table name (default: "entity")
    pub node_table: String,
    /// Edge table name (default: "relates_to")
    pub edge_table: String,
    /// Maximum traversal depth for safety
    pub max_traversal_depth: usize,
}

impl Default for SurrealDbGraphConfig {
    fn default() -> Self {
        Self {
            node_table: "entity".to_string(),
            edge_table: "relates_to".to_string(),
            max_traversal_depth: 10,
        }
    }
}

#[async_trait]
impl AsyncGraphStore for SurrealDbGraphStore {
    type Node = Entity;
    type Edge = Relationship;
    type Error = GraphRAGError;

    async fn add_node(&mut self, node: Self::Node) -> Result<String> {
        let id = node.id.to_string();
        let json_value = serde_json::to_value(&node)?;
        
        self.db
            .query("UPSERT type::thing($table, $id) CONTENT $data")
            .bind(("table", &self.config.node_table))
            .bind(("id", id.clone()))
            .bind(("data", json_value))
            .await?;
        
        Ok(id)
    }

    async fn add_edge(
        &mut self,
        from_id: &str,
        to_id: &str,
        edge: Self::Edge,
    ) -> Result<String> {
        // Use SurrealDB's RELATE syntax for graph edges
        let query = format!(
            "RELATE type::thing('{node_table}', $from_id) \
             -> {edge_table} -> \
             type::thing('{node_table}', $to_id) \
             CONTENT $data",
            node_table = self.config.node_table,
            edge_table = self.config.edge_table
        );

        let edge_data = EdgeData {
            relation_type: edge.relation_type.clone(),
            confidence: edge.confidence,
            context: edge.context.iter().map(|c| c.to_string()).collect(),
        };

        let mut response = self.db
            .query(&query)
            .bind(("from_id", from_id))
            .bind(("to_id", to_id))
            .bind(("data", serde_json::to_value(&edge_data)?))
            .await?;

        // Extract the created edge ID
        let results: Vec<serde_json::Value> = response.take(0)?;
        let edge_id = results
            .first()
            .and_then(|v| v.get("id"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("{}->{}:{}", from_id, to_id, edge.relation_type));

        Ok(edge_id)
    }

    async fn find_nodes(&self, criteria: &str) -> Result<Vec<Self::Node>> {
        // Parse criteria as a simple field=value or use as raw SurrealQL WHERE clause
        let query = if criteria.contains('=') || criteria.contains("WHERE") {
            format!(
                "SELECT *, meta::id(id) as id FROM {} WHERE {}",
                self.config.node_table, criteria
            )
        } else {
            // Treat as name search
            format!(
                "SELECT *, meta::id(id) as id FROM {} WHERE name CONTAINS $criteria",
                self.config.node_table
            )
        };

        let mut response = self.db
            .query(&query)
            .bind(("criteria", criteria))
            .await?;

        let results: Vec<serde_json::Value> = response.take(0)?;
        
        results
            .into_iter()
            .map(|v| serde_json::from_value(v).map_err(|e| GraphRAGError::Serialization {
                message: e.to_string(),
            }))
            .collect()
    }

    async fn get_neighbors(&self, node_id: &str) -> Result<Vec<Self::Node>> {
        // Use SurrealDB graph traversal syntax
        let query = format!(
            "SELECT *, meta::id(id) as id FROM \
             type::thing('{table}', $node_id)->{edge_table}->*",
            table = self.config.node_table,
            edge_table = self.config.edge_table
        );

        let mut response = self.db
            .query(&query)
            .bind(("node_id", node_id))
            .await?;

        let results: Vec<serde_json::Value> = response.take(0)?;
        
        results
            .into_iter()
            .map(|v| serde_json::from_value(v).map_err(|e| GraphRAGError::Serialization {
                message: e.to_string(),
            }))
            .collect()
    }

    async fn traverse(&self, start_id: &str, max_depth: usize) -> Result<Vec<Self::Node>> {
        let depth = max_depth.min(self.config.max_traversal_depth);
        
        // Recursive graph traversal with depth limit
        let query = format!(
            "SELECT *, meta::id(id) as id FROM \
             type::thing('{table}', $start_id)->{edge_table}->(*..{depth})",
            table = self.config.node_table,
            edge_table = self.config.edge_table,
            depth = depth
        );

        let mut response = self.db
            .query(&query)
            .bind(("start_id", start_id))
            .await?;

        let results: Vec<serde_json::Value> = response.take(0)?;
        
        results
            .into_iter()
            .map(|v| serde_json::from_value(v).map_err(|e| GraphRAGError::Serialization {
                message: e.to_string(),
            }))
            .collect()
    }

    async fn stats(&self) -> GraphStats {
        let node_count = self.count_nodes().await.unwrap_or(0);
        let edge_count = self.count_edges().await.unwrap_or(0);
        
        let average_degree = if node_count > 0 {
            (edge_count as f32 * 2.0) / node_count as f32
        } else {
            0.0
        };

        GraphStats {
            node_count,
            edge_count,
            average_degree,
            max_depth: self.config.max_traversal_depth,
        }
    }
}

impl SurrealDbGraphStore {
    async fn count_nodes(&self) -> Result<usize> {
        let query = format!(
            "SELECT count() FROM {} GROUP ALL",
            self.config.node_table
        );
        
        #[derive(serde::Deserialize)]
        struct CountResult { count: usize }
        
        let mut response = self.db.query(&query).await?;
        let results: Vec<CountResult> = response.take(0)?;
        Ok(results.first().map(|r| r.count).unwrap_or(0))
    }

    async fn count_edges(&self) -> Result<usize> {
        let query = format!(
            "SELECT count() FROM {} GROUP ALL",
            self.config.edge_table
        );
        
        #[derive(serde::Deserialize)]
        struct CountResult { count: usize }
        
        let mut response = self.db.query(&query).await?;
        let results: Vec<CountResult> = response.take(0)?;
        Ok(results.first().map(|r| r.count).unwrap_or(0))
    }
}

// Helper types
#[derive(Debug, Serialize, Deserialize)]
struct EdgeData {
    relation_type: String,
    confidence: f32,
    context: Vec<String>,
}
```

### Advanced Graph Queries

```rust
impl SurrealDbGraphStore {
    /// Find shortest path between two entities
    pub async fn find_path(
        &self,
        from_id: &str,
        to_id: &str,
        max_depth: usize,
    ) -> Result<Vec<Entity>> {
        let depth = max_depth.min(self.config.max_traversal_depth);
        
        // SurrealDB path finding with BFS
        let query = format!(
            "SELECT *, meta::id(id) as id FROM \
             type::thing('{table}', $from_id)->{edge_table}->(*..{depth}) \
             WHERE id = type::thing('{table}', $to_id) \
             LIMIT 1",
            table = self.config.node_table,
            edge_table = self.config.edge_table,
            depth = depth
        );

        let mut response = self.db
            .query(&query)
            .bind(("from_id", from_id))
            .bind(("to_id", to_id))
            .await?;

        let results: Vec<serde_json::Value> = response.take(0)?;
        // Parse path...
        Ok(vec![])
    }

    /// Get entities by relationship type
    pub async fn get_by_relationship(
        &self,
        entity_id: &str,
        relation_type: &str,
    ) -> Result<Vec<Entity>> {
        let query = format!(
            "SELECT *, meta::id(id) as id FROM \
             type::thing('{table}', $entity_id)->{edge_table} \
             WHERE relation_type = $relation_type \
             ->{table}",
            table = self.config.node_table,
            edge_table = self.config.edge_table
        );

        let mut response = self.db
            .query(&query)
            .bind(("entity_id", entity_id))
            .bind(("relation_type", relation_type))
            .await?;

        let results: Vec<serde_json::Value> = response.take(0)?;
        results
            .into_iter()
            .map(|v| serde_json::from_value(v).map_err(|e| GraphRAGError::Serialization {
                message: e.to_string(),
            }))
            .collect()
    }

    /// Get subgraph around an entity
    pub async fn get_subgraph(
        &self,
        center_id: &str,
        radius: usize,
    ) -> Result<(Vec<Entity>, Vec<Relationship>)> {
        // Get nodes in subgraph
        let nodes = self.traverse(center_id, radius).await?;
        
        // Get edges between those nodes
        let node_ids: Vec<String> = nodes.iter().map(|n| n.id.to_string()).collect();
        
        let edges_query = format!(
            "SELECT * FROM {} WHERE in IN $node_ids AND out IN $node_ids",
            self.config.edge_table
        );
        
        let mut response = self.db
            .query(&edges_query)
            .bind(("node_ids", node_ids))
            .await?;
        
        let edge_results: Vec<serde_json::Value> = response.take(0)?;
        let edges: Vec<Relationship> = edge_results
            .into_iter()
            .filter_map(|v| parse_relationship(v).ok())
            .collect();
        
        Ok((nodes, edges))
    }
}
```

### Usage Example

```rust
use graphrag_core::storage::surrealdb::{
    SurrealDbConfig, SurrealDbGraphStore, SurrealDbGraphConfig,
};
use graphrag_core::core::{Entity, EntityId, Relationship};

let db_config = SurrealDbConfig::rocksdb("./data/graph");
let graph_config = SurrealDbGraphConfig::default();
let mut graph_store = SurrealDbGraphStore::new(db_config, graph_config).await?;

// Add entities
let entity1 = Entity {
    id: EntityId::new("person_alice".to_string()),
    name: "Alice".to_string(),
    entity_type: "person".to_string(),
    confidence: 0.95,
    mentions: vec![],
    embedding: None,
};

let entity2 = Entity {
    id: EntityId::new("company_acme".to_string()),
    name: "Acme Corp".to_string(),
    entity_type: "organization".to_string(),
    confidence: 0.90,
    mentions: vec![],
    embedding: None,
};

graph_store.add_node(entity1).await?;
graph_store.add_node(entity2).await?;

// Add relationship
let relationship = Relationship {
    source: EntityId::new("person_alice".to_string()),
    target: EntityId::new("company_acme".to_string()),
    relation_type: "works_for".to_string(),
    confidence: 0.88,
    context: vec![],
};

graph_store.add_edge("person_alice", "company_acme", relationship).await?;

// Query graph
let neighbors = graph_store.get_neighbors("person_alice").await?;
println!("Alice's connections: {:?}", neighbors);

let stats = graph_store.stats().await;
println!("Graph has {} nodes and {} edges", stats.node_count, stats.edge_count);
```

---

## Unified Storage Interface

### Combined Storage

For convenience, a unified storage interface can combine all three stores:

```rust
// graphrag-core/src/storage/surrealdb/unified.rs

pub struct SurrealDbUnifiedStorage {
    storage: SurrealDbStorage,
    vector_store: SurrealDbVectorStore,
    graph_store: SurrealDbGraphStore,
}

impl SurrealDbUnifiedStorage {
    pub async fn new(config: SurrealDbConfig) -> Result<Self> {
        let db = surrealdb::engine::any::connect(&config.endpoint).await?;
        db.use_ns(&config.namespace).use_db(&config.database).await?;
        
        let db = Arc::new(db);
        
        // Share the connection across all stores
        Ok(Self {
            storage: SurrealDbStorage::from_client(Arc::clone(&db), config.clone())?,
            vector_store: SurrealDbVectorStore::from_client(
                Arc::clone(&db),
                SurrealDbVectorConfig::default(),
            )?,
            graph_store: SurrealDbGraphStore::from_client(
                Arc::clone(&db),
                SurrealDbGraphConfig::default(),
            )?,
        })
    }
    
    pub fn storage(&mut self) -> &mut SurrealDbStorage { &mut self.storage }
    pub fn vectors(&mut self) -> &mut SurrealDbVectorStore { &mut self.vector_store }
    pub fn graph(&mut self) -> &mut SurrealDbGraphStore { &mut self.graph_store }
}
```

---

## Configuration via TOML

The SurrealDB storage configuration integrates with the existing graphrag-rs configuration system, following the same patterns used by other top-level sections like `[embeddings]`, `[ollama]`, and `[graph]`.

### Basic Configuration

```toml
# =============================================================================
# GraphRAG Configuration with SurrealDB Storage
# =============================================================================

[general]
input_document_path = "path/to/your/document.txt"
output_dir = "./output"
log_level = "info"

# -----------------------------------------------------------------------------
# SurrealDB Storage Backend
# -----------------------------------------------------------------------------
[surrealdb]
# Enable SurrealDB storage backend
enabled = true

# Connection endpoint:
# - "mem://" for in-memory (development/testing)
# - "rocksdb://./data/graphrag" for persistent local storage
# - "ws://localhost:8000" for remote WebSocket connection
# - "http://localhost:8000" for remote HTTP connection
endpoint = "rocksdb://./data/graphrag"

# Namespace for data isolation (multi-tenant support)
namespace = "graphrag"

# Database name within namespace
database = "default"

# Whether to initialize schema on connect
auto_init_schema = true

# Optional authentication (required for remote connections)
[surrealdb.credentials]
username = "root"
password = "secret"

# -----------------------------------------------------------------------------
# Vector Store Configuration (Phase 2)
# -----------------------------------------------------------------------------
[surrealdb.vector]
# Dimension of vectors (must match embedding model)
# - 384 for all-MiniLM-L6-v2
# - 1024 for voyage-3-large, bge-large-en-v1.5
# - 1536 for text-embedding-3-small
dimension = 384

# Distance metric: "cosine", "euclidean", "manhattan"
distance_metric = "cosine"

# Table name for vector storage
table_name = "embeddings"

# Auto-build index after batch inserts
auto_index = true

# -----------------------------------------------------------------------------
# Graph Store Configuration (Phase 3)
# -----------------------------------------------------------------------------
[surrealdb.graph]
# Table name for graph nodes (entities)
node_table = "entity"

# Table name for graph edges (relationships)
edge_table = "relates_to"

# Maximum traversal depth for safety
max_traversal_depth = 10

# -----------------------------------------------------------------------------
# Embeddings Configuration (existing graphrag-rs pattern)
# -----------------------------------------------------------------------------
[embeddings]
# Provider: "huggingface", "openai", "voyage", "ollama", etc.
provider = "huggingface"
model = "sentence-transformers/all-MiniLM-L6-v2"
batch_size = 32

# -----------------------------------------------------------------------------
# Ollama Configuration (existing graphrag-rs pattern)
# -----------------------------------------------------------------------------
[ollama]
enabled = true
host = "http://localhost"
port = 11434
chat_model = "llama3.1:8b"
embedding_model = "nomic-embed-text"
timeout_seconds = 60

# -----------------------------------------------------------------------------
# Graph Construction (existing graphrag-rs pattern)
# -----------------------------------------------------------------------------
[graph]
max_connections = 15
similarity_threshold = 0.5
extract_relationships = true
relationship_confidence_threshold = 0.5
```

### Environment-Specific Configurations

#### Development (In-Memory)

```toml
[surrealdb]
enabled = true
endpoint = "mem://"
namespace = "graphrag"
database = "dev"
auto_init_schema = true
# No credentials needed for in-memory
```

#### Production (Persistent Local)

```toml
[surrealdb]
enabled = true
endpoint = "rocksdb://./data/graphrag-prod"
namespace = "graphrag"
database = "production"
auto_init_schema = false  # Schema managed externally

[surrealdb.credentials]
username = "admin"
password = "${SURREALDB_PASSWORD}"  # Environment variable substitution
```

#### Distributed (Remote Server)

```toml
[surrealdb]
enabled = true
endpoint = "ws://surrealdb.example.com:8000"
namespace = "graphrag"
database = "distributed"
auto_init_schema = false

[surrealdb.credentials]
username = "app_user"
password = "${SURREALDB_PASSWORD}"
```

### Rust Configuration Loading

The configuration integrates with the existing `SetConfig` system:

```rust
use graphrag_core::config::SetConfig;
use graphrag_core::storage::surrealdb::{SurrealDbConfig, SurrealDbStorage};

// Load from TOML file (uses existing graphrag-rs loader)
let set_config: SetConfig = toml::from_str(&config_content)?;

// Extract SurrealDB config section
let surrealdb_config = SurrealDbConfig {
    endpoint: set_config.surrealdb.endpoint,
    namespace: set_config.surrealdb.namespace,
    database: set_config.surrealdb.database,
    credentials: set_config.surrealdb.credentials.map(|c| SurrealDbCredentials {
        username: c.username,
        password: c.password,
    }),
    auto_init_schema: set_config.surrealdb.auto_init_schema,
};

// Create storage instance
let storage = SurrealDbStorage::new(surrealdb_config).await?;
```

### Programmatic Configuration

```rust
use graphrag_core::storage::surrealdb::SurrealDbConfig;

// Builder pattern (matching existing graphrag-rs style)
let config = SurrealDbConfig::rocksdb("./data/graphrag")
    .with_namespace("myapp")
    .with_database("production")
    .with_credentials("admin", "secret")
    .without_auto_schema();

// Or use factory methods
let dev_config = SurrealDbConfig::memory();
let ws_config = SurrealDbConfig::websocket("localhost", 8000);
let http_config = SurrealDbConfig::http("surrealdb.example.com", 8000);
```

---

## Dependencies Summary

```toml
[dependencies]
surrealdb = { version = "2.3", optional = true, default-features = false, features = [
    "kv-mem",      # In-memory storage
    "kv-rocksdb",  # Persistent local storage
    # Optional for remote:
    # "protocol-ws",   # WebSocket client
    # "protocol-http", # HTTP client
] }
```

## Compatibility Notes

- **SurrealDB Version**: 2.0.0 to 2.3.x (tested with 2.3)
- **Rust MSRV**: 1.80.1+ (matches SurrealDB SDK requirement)
- **Async Runtime**: Tokio (already required by graphrag-core with `async` feature)
- **Feature Interaction**: Requires `async` feature (SurrealDB SDK is async-only)

## Migration Path

1. **Phase 1** (Complete): AsyncStorage implementation
2. **Phase 2** (Planned): AsyncVectorStore implementation
3. **Phase 3** (Planned): AsyncGraphStore implementation
4. **Phase 4** (Future): Live queries for real-time updates, full-text search integration
