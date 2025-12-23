//! SurrealDB storage backend for GraphRAG
//!
//! This module provides a SurrealDB implementation of the `AsyncStorage` trait,
//! enabling persistent storage of documents, entities, and chunks using
//! SurrealDB's multi-model database capabilities.
//!
//! ## Features
//!
//! - **Multi-model storage**: Documents, entities, chunks stored as documents
//! - **Graph relationships**: Entity relationships stored as graph edges
//! - **Multiple backends**: In-memory, RocksDB, or remote connections
//! - **Automatic schema**: Optional schema initialization on connect
//!
//! ## Enable
//!
//! Add the `surrealdb-storage` feature to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! graphrag-core = { version = "0.1", features = ["surrealdb-storage"] }
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use graphrag_core::storage::surrealdb::{SurrealDbConfig, SurrealDbStorage};
//! use graphrag_core::core::traits::AsyncStorage;
//!
//! // In-memory for development
//! let config = SurrealDbConfig::memory();
//! let mut storage = SurrealDbStorage::new(config).await?;
//!
//! // Persistent local storage
//! let config = SurrealDbConfig::rocksdb("./data/graphrag");
//! let mut storage = SurrealDbStorage::new(config).await?;
//!
//! // Remote WebSocket connection
//! let config = SurrealDbConfig::websocket("localhost", 8000)
//!     .with_credentials("admin", "password");
//! let mut storage = SurrealDbStorage::new(config).await?;
//!
//! // Use AsyncStorage trait methods
//! let entity = Entity { /* ... */ };
//! let id = storage.store_entity(entity).await?;
//! let retrieved = storage.retrieve_entity(&id).await?;
//! ```
//!
//! ## Configuration
//!
//! The [`SurrealDbConfig`] struct supports builder-style configuration:
//!
//! ```rust,ignore
//! let config = SurrealDbConfig::rocksdb("./data")
//!     .with_namespace("myapp")
//!     .with_database("production")
//!     .with_credentials("user", "pass")
//!     .without_auto_schema();
//! ```
//!
//! ## Schema
//!
//! When `auto_init_schema` is enabled (default), the storage automatically
//! creates the required tables and indexes:
//!
//! - `document`: Stores Document records
//! - `entity`: Stores Entity records with indexes on id, type, and name
//! - `chunk`: Stores TextChunk records with document_id index
//! - `relates_to`: Graph edges for entity relationships

mod config;
mod error;
mod storage;
mod vector;

pub use config::{SurrealDbConfig, SurrealDbCredentials};
pub use error::SurrealDbStorageError;
pub use storage::SurrealDbStorage;
pub use vector::{DistanceMetric, SurrealDbVectorConfig, SurrealDbVectorStore};
