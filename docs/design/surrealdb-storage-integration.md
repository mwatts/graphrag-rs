# SurrealDB Storage Integration Design

## Overview

This document describes the design for integrating SurrealDB as a storage backend for graphrag-core. SurrealDB is particularly well-suited for GraphRAG workloads due to its native multi-model support (document, graph, relational, and vector), which aligns with the core data structures: documents, entities, chunks, and relationships.

**Implementation Status**: ✅ **COMPLETE** - All four phases implemented with 46 integration tests and comprehensive unit tests passing.

## Scope

### Completed Phases

| Phase | Component | Status | Tests |
|-------|-----------|--------|-------|
| **Phase 1** | `AsyncStorage` trait implementation | ✅ Complete | 13 integration tests |
| **Phase 2** | `AsyncVectorStore` trait implementation | ✅ Complete | 14 integration tests |
| **Phase 3** | `AsyncGraphStore` trait implementation | ✅ Complete | 14 integration tests |
| **Phase 4** | Unified Storage & TOML Config | ✅ Complete | 5 integration tests |

**Total: 46 integration tests passing**

### Phase 1: AsyncStorage (Complete)
- Document, Entity, and Chunk CRUD operations
- Configuration for SurrealDB connection (memory, RocksDB, WebSocket, HTTP)
- Feature flag integration for optional compilation
- Schema definition with indexes
- Batch operations with transaction support

### Phase 2: Vector Store (Complete)
- Vector similarity search with cosine, euclidean, and manhattan distance metrics
- Batch vector operations
- Metadata filtering
- MTREE indexing support
- Dimension validation

### Phase 3: Graph Store (Complete)
- Entity nodes with `add_node`, `get_node`, `find_nodes`
- Relationship edges with `add_edge` using SurrealDB `RELATE` statements
- Graph traversal with `get_neighbors`, `traverse` (BFS-based)
- Advanced queries: `find_path`, `get_by_relationship`, `get_subgraph`
- Removal operations: `remove_node`, `remove_edge`

### Phase 4: Unified Storage & Config (Complete)
- `SurrealDbUnifiedStorage` combining all three stores with shared connection
- `SurrealDbUnifiedConfig` for unified configuration
- `from_client` and `from_client_with_init` factory methods on all stores
- `SurrealDbSetConfig` integration with TOML configuration system
- Config template: `config/templates/surrealdb_unified.toml`

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

## Module Structure

```
graphrag-core/src/
├── storage/
│   ├── mod.rs           # Storage module exports with feature gate
│   └── surrealdb/
│       ├── mod.rs       # SurrealDB submodule exports
│       ├── config.rs    # SurrealDbConfig, SurrealDbCredentials
│       ├── error.rs     # SurrealDbStorageError
│       ├── storage.rs   # SurrealDbStorage (AsyncStorage impl)
│       ├── vector.rs    # SurrealDbVectorStore (AsyncVectorStore impl)
│       ├── graph.rs     # SurrealDbGraphStore (AsyncGraphStore impl)
│       ├── unified.rs   # SurrealDbUnifiedStorage (combined stores)
│       └── schema.surql # Schema definitions
├── config/
│   ├── mod.rs           # Exports SurrealDbSetConfig
│   └── setconfig.rs     # StorageConfig with surrealdb field
└── tests/
    └── surrealdb_storage_integration.rs  # 46 integration tests
```

## Core Types Mapping

### GraphRAG Types to SurrealDB Tables

| GraphRAG Type | SurrealDB Table | Model Type |
|---------------|-----------------|------------|
| `Document` | `document` | Document (flexible metadata) |
| `Entity` | `entity` | Document + Graph node |
| `TextChunk` | `chunk` | Document (with vector field) |
| `Relationship` | `relates_to` | Graph edge (RELATION type) |
| `Vector` | `vector` | Document (embedding + metadata) |

### Record ID Strategy

SurrealDB uses composite record IDs (`table:id`). GraphRAG IDs are mapped using backtick literals:

```rust
// DocumentId("doc123") -> document:`doc123`
// EntityId("ent456") -> entity:`ent456`
// ChunkId("chunk789") -> chunk:`chunk789`
```

**Implementation Note**: The implementation uses backtick record ID literals (e.g., `table:\`id\``) for `RELATE` statements, and `meta::id(id)` to convert SurrealDB's `Thing` type back to strings for retrieval.

## Schema Definition

```sql
-- graphrag-core/src/storage/surrealdb/schema.surql

-- Documents Table
DEFINE TABLE document SCHEMALESS;
DEFINE INDEX idx_document_id ON document FIELDS id UNIQUE;

-- Entities Table (Graph Nodes)
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

-- Vector Table for Embeddings
DEFINE TABLE vector SCHEMALESS;
DEFINE INDEX idx_vector_id ON vector FIELDS id UNIQUE;
DEFINE INDEX idx_vector_source ON vector FIELDS source_type, source_id;
```

---

## Configuration

### SurrealDbConfig

```rust
// graphrag-core/src/storage/surrealdb/config.rs

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurrealDbConfig {
    pub endpoint: String,
    pub namespace: String,
    pub database: String,
    pub credentials: Option<SurrealDbCredentials>,
    pub auto_init_schema: bool,
}

impl SurrealDbConfig {
    pub fn memory() -> Self { /* ... */ }
    pub fn rocksdb(path: impl Into<String>) -> Self { /* ... */ }
    pub fn websocket(host: impl Into<String>, port: u16) -> Self { /* ... */ }
    pub fn http(host: impl Into<String>, port: u16) -> Self { /* ... */ }
    
    pub fn with_namespace(self, namespace: impl Into<String>) -> Self { /* ... */ }
    pub fn with_database(self, database: impl Into<String>) -> Self { /* ... */ }
    pub fn with_credentials(self, username: impl Into<String>, password: impl Into<String>) -> Self { /* ... */ }
    pub fn without_auto_schema(self) -> Self { /* ... */ }
}
```

### SurrealDbVectorConfig

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurrealDbVectorConfig {
    pub dimension: usize,           // Must match embedding model (e.g., 384, 768, 1536)
    pub distance_metric: DistanceMetric,  // Cosine, Euclidean, Manhattan
    pub table_name: String,         // Default: "vector"
    pub auto_index: bool,           // Default: true
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DistanceMetric {
    #[default]
    Cosine,
    Euclidean,
    Manhattan,
}
```

### SurrealDbGraphConfig

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurrealDbGraphConfig {
    pub node_table: String,         // Default: "entity"
    pub edge_table: String,         // Default: "relates_to"
    pub max_traversal_depth: usize, // Default: 10
    pub auto_init_schema: bool,     // Default: true
}
```

### SurrealDbUnifiedConfig

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurrealDbUnifiedConfig {
    pub vector: SurrealDbVectorConfig,
    pub graph: SurrealDbGraphConfig,
    pub enable_vector: bool,  // Default: true
    pub enable_graph: bool,   // Default: true
}

impl SurrealDbUnifiedConfig {
    pub fn with_vector_dimension(dimension: usize) -> Self { /* ... */ }
    pub fn without_vector(self) -> Self { /* ... */ }
    pub fn without_graph(self) -> Self { /* ... */ }
}
```

---

## Storage Implementations

### SurrealDbStorage (Phase 1)

Implements `AsyncStorage` for documents, entities, and chunks.

```rust
pub struct SurrealDbStorage {
    db: Arc<Surreal<Any>>,
    config: SurrealDbConfig,
}

impl SurrealDbStorage {
    pub async fn new(config: SurrealDbConfig) -> Result<Self>;
    pub fn from_client(db: Arc<Surreal<Any>>, config: SurrealDbConfig) -> Self;
    pub async fn from_client_with_init(db: Arc<Surreal<Any>>, config: SurrealDbConfig) -> Result<Self>;
    
    pub fn client(&self) -> &Surreal<Any>;
    pub fn client_arc(&self) -> Arc<Surreal<Any>>;
    pub fn config(&self) -> &SurrealDbConfig;
}

#[async_trait]
impl AsyncStorage for SurrealDbStorage {
    type Entity = Entity;
    type Document = Document;
    type Chunk = TextChunk;
    type Error = GraphRAGError;

    async fn store_entity(&mut self, entity: Self::Entity) -> Result<String>;
    async fn retrieve_entity(&self, id: &str) -> Result<Option<Self::Entity>>;
    async fn store_document(&mut self, doc: Self::Document) -> Result<String>;
    async fn retrieve_document(&self, id: &str) -> Result<Option<Self::Document>>;
    async fn store_chunk(&mut self, chunk: Self::Chunk) -> Result<String>;
    async fn retrieve_chunk(&self, id: &str) -> Result<Option<Self::Chunk>>;
    // ... batch operations, flush, health_check
}
```

### SurrealDbVectorStore (Phase 2)

Implements `AsyncVectorStore` for vector similarity search.

```rust
pub struct SurrealDbVectorStore {
    db: Arc<Surreal<Any>>,
    config: SurrealDbVectorConfig,
    db_config: SurrealDbConfig,
}

impl SurrealDbVectorStore {
    pub async fn new(db_config: SurrealDbConfig, vector_config: SurrealDbVectorConfig) -> Result<Self>;
    pub fn from_client(db: Arc<Surreal<Any>>, db_config: SurrealDbConfig, vector_config: SurrealDbVectorConfig) -> Result<Self>;
    pub async fn from_client_with_init(db: Arc<Surreal<Any>>, db_config: SurrealDbConfig, vector_config: SurrealDbVectorConfig) -> Result<Self>;
}

#[async_trait]
impl AsyncVectorStore for SurrealDbVectorStore {
    type Error = GraphRAGError;

    async fn add_vector(&mut self, id: String, vector: Vec<f32>, metadata: VectorMetadata) -> Result<()>;
    async fn add_vectors_batch(&mut self, vectors: VectorBatch) -> Result<()>;
    async fn search(&self, query_vector: &[f32], k: usize) -> Result<Vec<SearchResult>>;
    async fn search_with_threshold(&self, query_vector: &[f32], k: usize, threshold: f32) -> Result<Vec<SearchResult>>;
    async fn remove_vector(&mut self, id: &str) -> Result<bool>;
    async fn len(&self) -> usize;
    async fn is_empty(&self) -> bool;
    async fn build_index(&mut self) -> Result<()>;
    async fn update_vector(&mut self, id: &str, vector: Vec<f32>, metadata: VectorMetadata) -> Result<bool>;
}
```

**Distance Metrics**: Uses SurrealDB's native vector functions:
- `vector::similarity::cosine` - Cosine similarity (higher = more similar)
- `vector::distance::euclidean` - Euclidean/L2 distance
- `vector::distance::manhattan` - Manhattan/L1 distance

### SurrealDbGraphStore (Phase 3)

Implements `AsyncGraphStore` for graph operations.

```rust
pub struct SurrealDbGraphStore {
    db: Arc<Surreal<Any>>,
    config: SurrealDbGraphConfig,
    db_config: SurrealDbConfig,
}

impl SurrealDbGraphStore {
    pub async fn new(db_config: SurrealDbConfig, graph_config: SurrealDbGraphConfig) -> Result<Self>;
    pub fn from_client(db: Arc<Surreal<Any>>, db_config: SurrealDbConfig, graph_config: SurrealDbGraphConfig) -> Result<Self>;
    pub async fn from_client_with_init(db: Arc<Surreal<Any>>, db_config: SurrealDbConfig, graph_config: SurrealDbGraphConfig) -> Result<Self>;
}

#[async_trait]
impl AsyncGraphStore for SurrealDbGraphStore {
    type Node = Entity;
    type Edge = Relationship;
    type Error = GraphRAGError;

    async fn add_node(&mut self, node: Self::Node) -> Result<String>;
    async fn add_nodes_batch(&mut self, nodes: Vec<Self::Node>) -> Result<Vec<String>>;
    async fn add_edge(&mut self, from_id: &str, to_id: &str, edge: Self::Edge) -> Result<String>;
    async fn add_edges_batch(&mut self, edges: Vec<(String, String, Self::Edge)>) -> Result<Vec<String>>;
    async fn find_nodes(&self, criteria: &str) -> Result<Vec<Self::Node>>;
    async fn get_neighbors(&self, node_id: &str) -> Result<Vec<Self::Node>>;
    async fn traverse(&self, start_id: &str, max_depth: usize) -> Result<Vec<Self::Node>>;
    async fn stats(&self) -> GraphStats;
    async fn health_check(&self) -> Result<bool>;
    async fn optimize(&mut self) -> Result<()>;
}

// Advanced graph queries
impl SurrealDbGraphStore {
    pub async fn get_node(&self, id: &str) -> Result<Option<Entity>>;
    pub async fn remove_node(&mut self, id: &str) -> Result<bool>;
    pub async fn remove_edge(&mut self, edge_id: &str) -> Result<bool>;
    pub async fn find_path(&self, from_id: &str, to_id: &str, max_depth: usize) -> Result<Vec<Entity>>;
    pub async fn get_by_relationship(&self, entity_id: &str, relation_type: Option<&str>) -> Result<Vec<Relationship>>;
    pub async fn get_subgraph(&self, center_id: &str, radius: usize) -> Result<(Vec<Entity>, Vec<Relationship>)>;
    pub async fn get_incoming(&self, entity_id: &str) -> Result<Vec<(Entity, Relationship)>>;
    pub async fn get_outgoing(&self, entity_id: &str) -> Result<Vec<(Entity, Relationship)>>;
}
```

**Key SurrealDB Graph Patterns**:

1. **Adding Edges** (using `RELATE`):
```sql
RELATE entity:`from_id`->relates_to->entity:`to_id` CONTENT $data
```

2. **Getting Neighbors** (graph traversal):
```sql
SELECT *, meta::id(id) as id FROM entity:`node_id`->relates_to->entity
```

3. **BFS Traversal** (iterative approach):
The `traverse` method uses iterative BFS since SurrealDB doesn't support the `(*..n)` recursive syntax.

### SurrealDbUnifiedStorage (Phase 4)

Combines all three stores with a shared database connection.

```rust
pub struct SurrealDbUnifiedStorage {
    db: Arc<Surreal<Any>>,
    db_config: SurrealDbConfig,
    unified_config: SurrealDbUnifiedConfig,
    storage: SurrealDbStorage,
    vector_store: Option<SurrealDbVectorStore>,
    graph_store: Option<SurrealDbGraphStore>,
}

impl SurrealDbUnifiedStorage {
    pub async fn new(db_config: SurrealDbConfig, unified_config: SurrealDbUnifiedConfig) -> Result<Self>;
    pub fn from_client(db: Arc<Surreal<Any>>, db_config: SurrealDbConfig, unified_config: SurrealDbUnifiedConfig) -> Result<Self>;
    pub async fn from_client_with_init(db: Arc<Surreal<Any>>, db_config: SurrealDbConfig, unified_config: SurrealDbUnifiedConfig) -> Result<Self>;

    // Accessors
    pub fn storage(&self) -> &SurrealDbStorage;
    pub fn storage_mut(&mut self) -> &mut SurrealDbStorage;
    pub fn vector_store(&self) -> Option<&SurrealDbVectorStore>;
    pub fn vector_store_mut(&mut self) -> Option<&mut SurrealDbVectorStore>;
    pub fn graph_store(&self) -> Option<&SurrealDbGraphStore>;
    pub fn graph_store_mut(&mut self) -> Option<&mut SurrealDbGraphStore>;
    
    pub fn client(&self) -> &Surreal<Any>;
    pub fn client_arc(&self) -> Arc<Surreal<Any>>;
    pub fn has_vector_store(&self) -> bool;
    pub fn has_graph_store(&self) -> bool;

    // Health & Stats
    pub async fn health_check(&self) -> Result<bool>;
    pub async fn stats(&self) -> UnifiedStorageStats;
    pub async fn flush(&self) -> Result<()>;
}

pub struct UnifiedStorageStats {
    pub vector_count: usize,
    pub graph_stats: Option<GraphStats>,
}
```

---

## TOML Configuration

### StorageConfig in SetConfig

```rust
// graphrag-core/src/config/setconfig.rs

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    pub database_type: String,  // "sqlite", "postgresql", "neo4j", or "surrealdb"
    pub database_path: String,
    pub enable_wal: bool,
    pub postgresql: Option<PostgreSQLConfig>,
    pub neo4j: Option<Neo4jConfig>,
    pub surrealdb: Option<SurrealDbSetConfig>,  // NEW
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurrealDbSetConfig {
    pub endpoint: String,           // "mem://", "rocksdb://./data", "ws://localhost:8000"
    pub namespace: String,          // Default: "graphrag"
    pub database: String,           // Default: "default"
    pub username: Option<String>,
    pub password: Option<String>,
    pub auto_init_schema: bool,     // Default: true
    pub vector_dimension: usize,    // Default: 384
    pub distance_metric: String,    // Default: "cosine"
    pub max_traversal_depth: usize, // Default: 10
    pub enable_vector_store: bool,  // Default: true
    pub enable_graph_store: bool,   // Default: true
}
```

### Config Template

See `config/templates/surrealdb_unified.toml` for a complete configuration example with:
- Development, production, and distributed deployment examples
- Vector store configuration
- Graph store configuration
- Integration with Ollama and embeddings settings

### Example TOML Configuration

```toml
[storage]
database_type = "surrealdb"

[storage.surrealdb]
endpoint = "rocksdb://./data/graphrag"
namespace = "graphrag"
database = "default"
auto_init_schema = true

# Vector store settings
vector_dimension = 384  # Match your embedding model
distance_metric = "cosine"
enable_vector_store = true

# Graph store settings
max_traversal_depth = 10
enable_graph_store = true

# Optional authentication (for remote connections)
# username = "admin"
# password = "secret"
```

---

## Usage Examples

### Basic Storage Usage

```rust
use graphrag_core::storage::surrealdb::{SurrealDbConfig, SurrealDbStorage};
use graphrag_core::core::traits::AsyncStorage;

let config = SurrealDbConfig::memory();
let mut storage = SurrealDbStorage::new(config).await?;

// Store and retrieve entities
let entity = Entity { /* ... */ };
let id = storage.store_entity(entity).await?;
let retrieved = storage.retrieve_entity(&id).await?;
```

### Vector Search Usage

```rust
use graphrag_core::storage::surrealdb::{
    SurrealDbConfig, SurrealDbVectorStore, SurrealDbVectorConfig, DistanceMetric,
};
use graphrag_core::core::traits::AsyncVectorStore;

let db_config = SurrealDbConfig::rocksdb("./data/vectors");
let vector_config = SurrealDbVectorConfig::with_dimension(384)
    .with_metric(DistanceMetric::Cosine);

let mut vector_store = SurrealDbVectorStore::new(db_config, vector_config).await?;

// Add vectors
vector_store.add_vector("chunk_1".to_string(), embedding, None).await?;

// Search
let results = vector_store.search(&query_embedding, 10).await?;
```

### Graph Operations Usage

```rust
use graphrag_core::storage::surrealdb::{
    SurrealDbConfig, SurrealDbGraphStore, SurrealDbGraphConfig,
};
use graphrag_core::core::traits::AsyncGraphStore;

let db_config = SurrealDbConfig::rocksdb("./data/graph");
let graph_config = SurrealDbGraphConfig::default();
let mut graph_store = SurrealDbGraphStore::new(db_config, graph_config).await?;

// Add entities and relationships
graph_store.add_node(entity).await?;
graph_store.add_edge("alice", "acme", relationship).await?;

// Query graph
let neighbors = graph_store.get_neighbors("alice").await?;
let subgraph = graph_store.get_subgraph("alice", 2).await?;
```

### Unified Storage Usage

```rust
use graphrag_core::storage::surrealdb::{
    SurrealDbConfig, SurrealDbUnifiedStorage, SurrealDbUnifiedConfig,
};

let db_config = SurrealDbConfig::memory();
let unified_config = SurrealDbUnifiedConfig::with_vector_dimension(384);

let mut storage = SurrealDbUnifiedStorage::new(db_config, unified_config).await?;

// Use document storage
storage.storage_mut().store_document(doc).await?;

// Use vector store
if let Some(vectors) = storage.vector_store_mut() {
    vectors.add_vector("id".to_string(), embedding, None).await?;
}

// Use graph store
if let Some(graph) = storage.graph_store_mut() {
    graph.add_node(entity).await?;
}

// Get unified stats
let stats = storage.stats().await;
println!("Vectors: {}, Nodes: {}", stats.vector_count, stats.graph_stats.unwrap().node_count);
```

---

## Testing

### Test Summary

| Test Category | Count | Location |
|---------------|-------|----------|
| Storage integration | 13 | `surrealdb_storage_integration.rs` |
| Vector store integration | 14 | `surrealdb_storage_integration.rs` |
| Graph store integration | 14 | `surrealdb_storage_integration.rs` |
| Unified storage integration | 5 | `surrealdb_storage_integration.rs` |
| Unit tests (config, parsing) | ~20 | Module `#[cfg(test)]` blocks |

### Running Tests

```bash
# All SurrealDB integration tests
cargo test --package graphrag-core --test surrealdb_storage_integration --features surrealdb-storage

# Specific test categories
cargo test --features surrealdb-storage test_graph
cargo test --features surrealdb-storage test_vector
cargo test --features surrealdb-storage test_unified

# Unit tests
cargo test --package graphrag-core --lib --features surrealdb-storage
```

---

## Dependencies

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

## Key Implementation Learnings

1. **Record ID Literals**: Use backticks for record IDs in `RELATE` statements: `entity:\`id\``
2. **Type Conversion**: Use `string::concat("", meta::id(field))` to get plain strings from Thing types
3. **Graph Traversal**: SurrealDB doesn't support recursive `(*..n)` syntax; use iterative BFS instead
4. **Serialization**: Serialize to `serde_json::Value` first for complex types with newtype wrappers
5. **ID Extraction**: Use `meta::id(id) as id` in SELECT to get string IDs from SurrealDB Things
