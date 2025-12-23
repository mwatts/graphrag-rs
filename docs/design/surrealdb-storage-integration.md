# SurrealDB Storage Integration Design

## Overview

This document describes the design for integrating SurrealDB as a storage backend for graphrag-core, implementing the `AsyncStorage` trait. SurrealDB is particularly well-suited for GraphRAG workloads due to its native multi-model support (document, graph, relational, and vector), which aligns with the core data structures: documents, entities, chunks, and relationships.

## Scope

This design is constrained to:
1. Implementing the `AsyncStorage` trait from `graphrag-core/src/core/traits.rs`
2. Required configuration for the SurrealDB connection
3. Feature flag integration for optional compilation

Out of scope:
- Vector store implementation (`AsyncVectorStore` trait)
- Graph store implementation (`AsyncGraphStore` trait)
- These may be implemented separately as SurrealDB supports all these models natively

## Feature Flag

Add to `graphrag-core/Cargo.toml`:

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

SurrealDB uses composite record IDs (`table:id`). Map GraphRAG IDs directly:

```rust
// DocumentId("doc123") -> document:doc123
// EntityId("ent456") -> entity:ent456  
// ChunkId("chunk789") -> chunk:chunk789
```

## Schema Definition

```sql
-- Documents table (document model for flexible metadata)
DEFINE TABLE document SCHEMAFULL;
DEFINE FIELD id ON document TYPE string;
DEFINE FIELD title ON document TYPE string;
DEFINE FIELD content ON document TYPE string;
DEFINE FIELD metadata ON document FLEXIBLE TYPE object;
DEFINE FIELD created_at ON document TYPE datetime DEFAULT time::now();
DEFINE INDEX idx_document_id ON document FIELDS id UNIQUE;

-- Entities table (hybrid: document + graph via relationships)
DEFINE TABLE entity SCHEMAFULL;
DEFINE FIELD id ON entity TYPE string;
DEFINE FIELD name ON entity TYPE string;
DEFINE FIELD entity_type ON entity TYPE string;
DEFINE FIELD confidence ON entity TYPE float;
DEFINE FIELD embedding ON entity TYPE option<array<float>>;
DEFINE FIELD mentions ON entity TYPE array<object>;
DEFINE INDEX idx_entity_id ON entity FIELDS id UNIQUE;
DEFINE INDEX idx_entity_type ON entity FIELDS entity_type;
DEFINE INDEX idx_entity_name ON entity FIELDS name;

-- Chunks table (document model with vector support)
DEFINE TABLE chunk SCHEMAFULL;
DEFINE FIELD id ON chunk TYPE string;
DEFINE FIELD document_id ON chunk TYPE string;
DEFINE FIELD content ON chunk TYPE string;
DEFINE FIELD start_offset ON chunk TYPE int;
DEFINE FIELD end_offset ON chunk TYPE int;
DEFINE FIELD embedding ON chunk TYPE option<array<float>>;
DEFINE FIELD entity_ids ON chunk TYPE array<string>;
DEFINE FIELD metadata ON chunk TYPE object;
DEFINE INDEX idx_chunk_id ON chunk FIELDS id UNIQUE;
DEFINE INDEX idx_chunk_document ON chunk FIELDS document_id;

-- Relationships as graph edges
DEFINE TABLE relates_to TYPE RELATION IN entity OUT entity;
DEFINE FIELD relation_type ON relates_to TYPE string;
DEFINE FIELD confidence ON relates_to TYPE float;
DEFINE FIELD context ON relates_to TYPE array<string>;
DEFINE INDEX idx_relates_type ON relates_to FIELDS relation_type;
```

## Implementation

### Module Structure

```
graphrag-core/src/
├── storage/
│   ├── mod.rs           # Storage module exports (existing)
│   └── surrealdb/
│       ├── mod.rs       # SurrealDB submodule exports, feature gate
│       ├── storage.rs   # AsyncStorage implementation
│       ├── config.rs    # Configuration types
│       ├── error.rs     # SurrealDB-specific error types
│       └── schema.surql # Schema definitions
```

### Configuration

```rust
// graphrag-core/src/storage/surrealdb/config.rs

use serde::{Deserialize, Serialize};

/// SurrealDB connection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurrealDbConfig {
    /// Connection endpoint
    /// - "mem://" for in-memory (development)
    /// - "rocksdb://path/to/db" for persistent local
    /// - "ws://host:port" for remote WebSocket
    /// - "http://host:port" for remote HTTP
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurrealDbCredentials {
    pub username: String,
    pub password: String,
}

impl Default for SurrealDbConfig {
    fn default() -> Self {
        Self {
            endpoint: "mem://".to_string(),
            namespace: "graphrag".to_string(),
            database: "default".to_string(),
            credentials: None,
            auto_init_schema: true,
        }
    }
}

impl SurrealDbConfig {
    /// Create configuration for in-memory storage (development/testing)
    pub fn memory() -> Self {
        Self::default()
    }
    
    /// Create configuration for persistent local storage
    pub fn rocksdb(path: impl Into<String>) -> Self {
        Self {
            endpoint: format!("rocksdb://{}", path.into()),
            ..Self::default()
        }
    }
    
    /// Create configuration for remote WebSocket connection
    pub fn websocket(host: impl Into<String>, port: u16) -> Self {
        Self {
            endpoint: format!("ws://{}:{}", host.into(), port),
            ..Self::default()
        }
    }
}
```

### Error Types

```rust
// graphrag-core/src/storage/surrealdb/error.rs

use thiserror::Error;

/// Errors specific to SurrealDB storage operations
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
    
    #[error("Record not found: {table}:{id}")]
    NotFound { table: String, id: String },
    
    #[error("Duplicate record: {table}:{id}")]
    DuplicateRecord { table: String, id: String },
}

impl From<surrealdb::Error> for SurrealDbStorageError {
    fn from(err: surrealdb::Error) -> Self {
        Self::QueryFailed(err.to_string())
    }
}
```

### Storage Implementation

```rust
// graphrag-core/src/storage/surrealdb/storage.rs

use std::sync::Arc;
use async_trait::async_trait;
use surrealdb::{engine::any::Any, Surreal};

use crate::core::{
    traits::AsyncStorage,
    Document, Entity, TextChunk, Result, GraphRAGError,
};
use super::config::SurrealDbConfig;
use super::error::SurrealDbStorageError;

/// SurrealDB implementation of AsyncStorage
pub struct SurrealDbStorage {
    db: Arc<Surreal<Any>>,
    config: SurrealDbConfig,
}

impl SurrealDbStorage {
    /// Create a new SurrealDB storage instance
    ///
    /// # Arguments
    /// * `config` - Configuration for the SurrealDB connection
    ///
    /// # Example
    /// ```rust,ignore
    /// let config = SurrealDbConfig::memory();
    /// let storage = SurrealDbStorage::new(config).await?;
    /// ```
    pub async fn new(config: SurrealDbConfig) -> Result<Self> {
        let db = Surreal::new::<Any>(&config.endpoint)
            .await
            .map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to connect to SurrealDB: {}", e),
            })?;
        
        // Select namespace and database
        db.use_ns(&config.namespace)
            .use_db(&config.database)
            .await
            .map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to select namespace/database: {}", e),
            })?;
        
        // Authenticate if credentials provided
        if let Some(ref creds) = config.credentials {
            db.signin(surrealdb::opt::auth::Root {
                username: &creds.username,
                password: &creds.password,
            })
            .await
            .map_err(|e| GraphRAGError::Storage {
                message: format!("Authentication failed: {}", e),
            })?;
        }
        
        let storage = Self {
            db: Arc::new(db),
            config,
        };
        
        // Initialize schema if configured
        if storage.config.auto_init_schema {
            storage.init_schema().await?;
        }
        
        Ok(storage)
    }
    
    /// Initialize the database schema
    async fn init_schema(&self) -> Result<()> {
        let schema = include_str!("schema.surql");
        self.db.query(schema).await.map_err(|e| GraphRAGError::Storage {
            message: format!("Schema initialization failed: {}", e),
        })?;
        Ok(())
    }
    
    /// Get reference to the underlying SurrealDB client
    pub fn client(&self) -> &Surreal<Any> {
        &self.db
    }
}

#[async_trait]
impl AsyncStorage for SurrealDbStorage {
    type Entity = Entity;
    type Document = Document;
    type Chunk = TextChunk;
    type Error = GraphRAGError;
    
    async fn store_entity(&mut self, entity: Self::Entity) -> Result<String> {
        let id = entity.id.to_string();
        
        // Use upsert semantics for idempotent storage
        let _: Option<Entity> = self.db
            .upsert(("entity", &id))
            .content(&entity)
            .await
            .map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to store entity: {}", e),
            })?;
        
        Ok(id)
    }
    
    async fn retrieve_entity(&self, id: &str) -> Result<Option<Self::Entity>> {
        let entity: Option<Entity> = self.db
            .select(("entity", id))
            .await
            .map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to retrieve entity: {}", e),
            })?;
        
        Ok(entity)
    }
    
    async fn store_document(&mut self, document: Self::Document) -> Result<String> {
        let id = document.id.to_string();
        
        let _: Option<Document> = self.db
            .upsert(("document", &id))
            .content(&document)
            .await
            .map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to store document: {}", e),
            })?;
        
        Ok(id)
    }
    
    async fn retrieve_document(&self, id: &str) -> Result<Option<Self::Document>> {
        let document: Option<Document> = self.db
            .select(("document", id))
            .await
            .map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to retrieve document: {}", e),
            })?;
        
        Ok(document)
    }
    
    async fn store_chunk(&mut self, chunk: Self::Chunk) -> Result<String> {
        let id = chunk.id.to_string();
        
        let _: Option<TextChunk> = self.db
            .upsert(("chunk", &id))
            .content(&chunk)
            .await
            .map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to store chunk: {}", e),
            })?;
        
        Ok(id)
    }
    
    async fn retrieve_chunk(&self, id: &str) -> Result<Option<Self::Chunk>> {
        let chunk: Option<TextChunk> = self.db
            .select(("chunk", id))
            .await
            .map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to retrieve chunk: {}", e),
            })?;
        
        Ok(chunk)
    }
    
    async fn list_entities(&self) -> Result<Vec<String>> {
        let mut response = self.db
            .query("SELECT id FROM entity")
            .await
            .map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to list entities: {}", e),
            })?;
        
        let entities: Vec<EntityIdRecord> = response.take(0).map_err(|e| {
            GraphRAGError::Storage {
                message: format!("Failed to parse entity list: {}", e),
            }
        })?;
        
        Ok(entities.into_iter().map(|r| r.id).collect())
    }
    
    async fn store_entities_batch(&mut self, entities: Vec<Self::Entity>) -> Result<Vec<String>> {
        let mut ids = Vec::with_capacity(entities.len());
        
        // Use transaction for batch atomicity
        self.db.query("BEGIN TRANSACTION").await.map_err(|e| {
            GraphRAGError::Storage {
                message: format!("Failed to begin transaction: {}", e),
            }
        })?;
        
        for entity in entities {
            let id = entity.id.to_string();
            let _: Option<Entity> = self.db
                .upsert(("entity", &id))
                .content(&entity)
                .await
                .map_err(|e| {
                    // Attempt rollback on error
                    let _ = futures::executor::block_on(
                        self.db.query("CANCEL TRANSACTION")
                    );
                    GraphRAGError::Storage {
                        message: format!("Failed to store entity in batch: {}", e),
                    }
                })?;
            ids.push(id);
        }
        
        self.db.query("COMMIT TRANSACTION").await.map_err(|e| {
            GraphRAGError::Storage {
                message: format!("Failed to commit transaction: {}", e),
            }
        })?;
        
        Ok(ids)
    }
    
    async fn health_check(&self) -> Result<bool> {
        // Simple query to verify connection
        self.db
            .query("RETURN true")
            .await
            .map(|_| true)
            .map_err(|e| GraphRAGError::Storage {
                message: format!("Health check failed: {}", e),
            })
    }
    
    async fn flush(&mut self) -> Result<()> {
        // SurrealDB handles persistence automatically
        // For RocksDB backend, this is a no-op as writes are durable
        Ok(())
    }
}

// Helper struct for parsing entity ID queries
#[derive(serde::Deserialize)]
struct EntityIdRecord {
    id: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{DocumentId, EntityId, ChunkId, ChunkMetadata};
    
    #[tokio::test]
    async fn test_surrealdb_storage_memory() {
        let config = SurrealDbConfig::memory();
        let mut storage = SurrealDbStorage::new(config).await.unwrap();
        
        // Test entity storage
        let entity = Entity {
            id: EntityId::new("test_entity".to_string()),
            name: "Test Entity".to_string(),
            entity_type: "test".to_string(),
            confidence: 0.95,
            mentions: vec![],
            embedding: None,
        };
        
        let id = storage.store_entity(entity.clone()).await.unwrap();
        assert_eq!(id, "test_entity");
        
        let retrieved = storage.retrieve_entity(&id).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().name, "Test Entity");
    }
    
    #[tokio::test]
    async fn test_health_check() {
        let config = SurrealDbConfig::memory();
        let storage = SurrealDbStorage::new(config).await.unwrap();
        
        let healthy = storage.health_check().await.unwrap();
        assert!(healthy);
    }
}
```

### Schema File

```sql
-- graphrag-core/src/storage/surrealdb/schema.surql

-- Documents table
DEFINE TABLE document SCHEMAFULL;
DEFINE FIELD id ON document TYPE string;
DEFINE FIELD title ON document TYPE string;
DEFINE FIELD content ON document TYPE string;
DEFINE FIELD metadata ON document FLEXIBLE TYPE object;
DEFINE FIELD chunks ON document TYPE array<object> DEFAULT [];
DEFINE FIELD created_at ON document TYPE datetime DEFAULT time::now();
DEFINE INDEX idx_document_id ON document FIELDS id UNIQUE;

-- Entities table
DEFINE TABLE entity SCHEMAFULL;
DEFINE FIELD id ON entity TYPE string;
DEFINE FIELD name ON entity TYPE string;
DEFINE FIELD entity_type ON entity TYPE string;
DEFINE FIELD confidence ON entity TYPE float;
DEFINE FIELD embedding ON entity TYPE option<array<float>>;
DEFINE FIELD mentions ON entity TYPE array<object> DEFAULT [];
DEFINE INDEX idx_entity_id ON entity FIELDS id UNIQUE;
DEFINE INDEX idx_entity_type ON entity FIELDS entity_type;
DEFINE INDEX idx_entity_name ON entity FIELDS name;

-- Chunks table
DEFINE TABLE chunk SCHEMAFULL;
DEFINE FIELD id ON chunk TYPE string;
DEFINE FIELD document_id ON chunk TYPE string;
DEFINE FIELD content ON chunk TYPE string;
DEFINE FIELD start_offset ON chunk TYPE int;
DEFINE FIELD end_offset ON chunk TYPE int;
DEFINE FIELD embedding ON chunk TYPE option<array<float>>;
DEFINE FIELD entities ON chunk TYPE array<string> DEFAULT [];
DEFINE FIELD metadata ON chunk TYPE object;
DEFINE INDEX idx_chunk_id ON chunk FIELDS id UNIQUE;
DEFINE INDEX idx_chunk_document ON chunk FIELDS document_id;

-- Relationships as graph edges (for future AsyncGraphStore implementation)
DEFINE TABLE relates_to TYPE RELATION IN entity OUT entity;
DEFINE FIELD relation_type ON relates_to TYPE string;
DEFINE FIELD confidence ON relates_to TYPE float;
DEFINE FIELD context ON relates_to TYPE array<string> DEFAULT [];
DEFINE INDEX idx_relates_type ON relates_to FIELDS relation_type;
```

### Module Exports

```rust
// graphrag-core/src/storage/surrealdb/mod.rs

//! SurrealDB storage backend for GraphRAG
//!
//! This module provides a SurrealDB implementation of the `AsyncStorage` trait,
//! enabling persistent storage of documents, entities, and chunks.
//!
//! ## Features
//!
//! Enable with the `surrealdb-storage` feature:
//! ```toml
//! [dependencies]
//! graphrag-core = { version = "0.1", features = ["surrealdb-storage"] }
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use graphrag_core::surrealdb::{SurrealDbStorage, SurrealDbConfig};
//!
//! // In-memory for development
//! let config = SurrealDbConfig::memory();
//! let storage = SurrealDbStorage::new(config).await?;
//!
//! // Persistent local storage
//! let config = SurrealDbConfig::rocksdb("./data/graphrag");
//! let storage = SurrealDbStorage::new(config).await?;
//! ```

mod config;
mod error;
mod storage;

pub use config::{SurrealDbConfig, SurrealDbCredentials};
pub use error::SurrealDbStorageError;
pub use storage::SurrealDbStorage;
```

## Integration with graphrag-core

Update `graphrag-core/src/storage/mod.rs` to include the SurrealDB submodule:

```rust
// graphrag-core/src/storage/mod.rs

//! Storage layer for GraphRAG
//!
//! This module provides abstractions and implementations for storing
//! knowledge graph data, vectors, and metadata.

// ... existing code ...

// SurrealDB storage backend (optional)
#[cfg(feature = "surrealdb-storage")]
pub mod surrealdb;

#[cfg(feature = "surrealdb-storage")]
pub use surrealdb::{SurrealDbConfig, SurrealDbCredentials, SurrealDbStorage};
```

The storage module is already re-exported from `graphrag-core/src/lib.rs`, so users can access it via:

```rust
use graphrag_core::storage::surrealdb::{SurrealDbConfig, SurrealDbStorage};
// or via the re-export:
use graphrag_core::storage::{SurrealDbConfig, SurrealDbStorage};
```

## Configuration via TOML

Example configuration file support:

```toml
[storage]
backend = "surrealdb"

[storage.surrealdb]
endpoint = "rocksdb://./data/graphrag"
namespace = "graphrag"
database = "production"
auto_init_schema = true

[storage.surrealdb.credentials]
username = "root"
password = "secret"
```

## Testing Strategy

1. **Unit Tests**: Test individual CRUD operations with in-memory backend
2. **Integration Tests**: Test with RocksDB backend for persistence verification
3. **Feature Gate Tests**: Ensure code compiles with/without `surrealdb-storage` feature

```rust
#[cfg(test)]
#[cfg(feature = "surrealdb-storage")]
mod integration_tests {
    use super::*;
    use tempfile::TempDir;
    
    #[tokio::test]
    async fn test_persistent_storage() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_str().unwrap();
        
        // Store data
        {
            let config = SurrealDbConfig::rocksdb(path);
            let mut storage = SurrealDbStorage::new(config).await.unwrap();
            
            let entity = Entity { /* ... */ };
            storage.store_entity(entity).await.unwrap();
        }
        
        // Verify persistence after reconnect
        {
            let config = SurrealDbConfig::rocksdb(path);
            let storage = SurrealDbStorage::new(config).await.unwrap();
            
            let retrieved = storage.retrieve_entity("test_entity").await.unwrap();
            assert!(retrieved.is_some());
        }
    }
}
```

## Future Extensions

This design provides a foundation for additional SurrealDB integrations:

1. **AsyncVectorStore**: Leverage SurrealDB's native vector type and `vector::similarity::cosine()` for semantic search
2. **AsyncGraphStore**: Use `RELATE` statements and graph traversal for the knowledge graph
3. **Live Queries**: Subscribe to real-time updates for incremental graph processing
4. **Full-Text Search**: Use SurrealDB's built-in search capabilities

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

- **SurrealDB Version**: 2.0.0 to 2.3.x
- **Rust MSRV**: 1.80.1+ (matches SurrealDB SDK requirement)
- **Async Runtime**: Tokio (already required by graphrag-core with `async` feature)
- **Feature Interaction**: Requires `async` feature (SurrealDB SDK is async-only)
