//! SurrealDB Unified Storage
//!
//! This module provides a unified storage interface that combines the document/entity
//! storage, vector store, and graph store into a single cohesive unit. This enables
//! using a single SurrealDB connection for all GraphRAG storage needs.
//!
//! ## Features
//!
//! - **Single connection**: All stores share one database connection
//! - **Unified configuration**: Single config point for all storage types
//! - **Coordinated lifecycle**: Initialize all schemas together
//! - **Type-safe access**: Individual store accessors with proper typing
//!
//! ## Usage
//!
//! ```rust,ignore
//! use graphrag_core::storage::surrealdb::{
//!     SurrealDbConfig, SurrealDbUnifiedStorage, SurrealDbUnifiedConfig,
//! };
//!
//! // Create unified storage with defaults
//! let config = SurrealDbConfig::memory();
//! let unified_config = SurrealDbUnifiedConfig::default();
//! let storage = SurrealDbUnifiedStorage::new(config, unified_config).await?;
//!
//! // Access individual stores
//! let doc_store = storage.storage();
//! let vector_store = storage.vector_store();
//! let graph_store = storage.graph_store();
//! ```

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use surrealdb::engine::any::Any;
use surrealdb::Surreal;

use crate::core::error::{GraphRAGError, Result};
use crate::core::traits::{AsyncGraphStore, AsyncVectorStore};

use super::config::SurrealDbConfig;
use super::graph::{SurrealDbGraphConfig, SurrealDbGraphStore};
use super::storage::SurrealDbStorage;
use super::vector::{SurrealDbVectorConfig, SurrealDbVectorStore};

// =============================================================================
// Unified Configuration
// =============================================================================

/// Unified configuration for all SurrealDB storage components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurrealDbUnifiedConfig {
    /// Vector store configuration
    #[serde(default)]
    pub vector: SurrealDbVectorConfig,

    /// Graph store configuration
    #[serde(default)]
    pub graph: SurrealDbGraphConfig,

    /// Whether to enable the vector store
    #[serde(default = "default_enable_vector")]
    pub enable_vector: bool,

    /// Whether to enable the graph store
    #[serde(default = "default_enable_graph")]
    pub enable_graph: bool,
}

fn default_enable_vector() -> bool {
    true
}

fn default_enable_graph() -> bool {
    true
}

impl Default for SurrealDbUnifiedConfig {
    fn default() -> Self {
        Self {
            vector: SurrealDbVectorConfig::default(),
            graph: SurrealDbGraphConfig::default(),
            enable_vector: default_enable_vector(),
            enable_graph: default_enable_graph(),
        }
    }
}

impl SurrealDbUnifiedConfig {
    /// Create a config with custom vector dimensions
    pub fn with_vector_dimension(dimension: usize) -> Self {
        Self {
            vector: SurrealDbVectorConfig::with_dimension(dimension),
            ..Default::default()
        }
    }

    /// Set the vector configuration
    pub fn with_vector_config(mut self, config: SurrealDbVectorConfig) -> Self {
        self.vector = config;
        self
    }

    /// Set the graph configuration
    pub fn with_graph_config(mut self, config: SurrealDbGraphConfig) -> Self {
        self.graph = config;
        self
    }

    /// Disable the vector store
    pub fn without_vector(mut self) -> Self {
        self.enable_vector = false;
        self
    }

    /// Disable the graph store
    pub fn without_graph(mut self) -> Self {
        self.enable_graph = false;
        self
    }

    /// Validate all configurations
    pub fn validate(&self) -> Result<()> {
        if self.enable_vector {
            self.vector.validate()?;
        }
        if self.enable_graph {
            self.graph.validate()?;
        }
        Ok(())
    }
}

// =============================================================================
// Unified Storage
// =============================================================================

/// Unified SurrealDB storage combining document, vector, and graph stores
///
/// This struct provides a single entry point for all SurrealDB storage needs,
/// sharing a single database connection across all store types.
pub struct SurrealDbUnifiedStorage {
    /// Shared database connection
    db: Arc<Surreal<Any>>,

    /// Database configuration
    db_config: SurrealDbConfig,

    /// Unified configuration
    unified_config: SurrealDbUnifiedConfig,

    /// Document/entity storage
    storage: SurrealDbStorage,

    /// Vector store (optional based on config)
    vector_store: Option<SurrealDbVectorStore>,

    /// Graph store (optional based on config)
    graph_store: Option<SurrealDbGraphStore>,
}

impl std::fmt::Debug for SurrealDbUnifiedStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SurrealDbUnifiedStorage")
            .field("db_config", &self.db_config)
            .field("unified_config", &self.unified_config)
            .field("vector_enabled", &self.vector_store.is_some())
            .field("graph_enabled", &self.graph_store.is_some())
            .finish_non_exhaustive()
    }
}

impl SurrealDbUnifiedStorage {
    /// Create a new unified storage with the given configurations
    ///
    /// This establishes a single database connection and initializes all
    /// configured storage components.
    ///
    /// # Arguments
    ///
    /// * `db_config` - Database connection configuration
    /// * `unified_config` - Configuration for all storage components
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let db_config = SurrealDbConfig::memory();
    /// let unified_config = SurrealDbUnifiedConfig::with_vector_dimension(384);
    ///
    /// let storage = SurrealDbUnifiedStorage::new(db_config, unified_config).await?;
    /// ```
    pub async fn new(
        db_config: SurrealDbConfig,
        unified_config: SurrealDbUnifiedConfig,
    ) -> Result<Self> {
        unified_config.validate()?;

        // Create single database connection
        let db = surrealdb::engine::any::connect(&db_config.endpoint)
            .await
            .map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to connect to SurrealDB: {}", e),
            })?;

        // Select namespace and database
        db.use_ns(&db_config.namespace)
            .use_db(&db_config.database)
            .await
            .map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to select namespace/database: {}", e),
            })?;

        // Authenticate if credentials provided
        if let Some(ref creds) = db_config.credentials {
            db.signin(surrealdb::opt::auth::Root {
                username: &creds.username,
                password: &creds.password,
            })
            .await
            .map_err(|e| GraphRAGError::Storage {
                message: format!("Authentication failed: {}", e),
            })?;
        }

        let db = Arc::new(db);

        // Create storage from shared connection
        let storage = if db_config.auto_init_schema {
            SurrealDbStorage::from_client_with_init(Arc::clone(&db), db_config.clone()).await?
        } else {
            SurrealDbStorage::from_client(Arc::clone(&db), db_config.clone())
        };

        // Create vector store if enabled
        let vector_store = if unified_config.enable_vector {
            let store = if db_config.auto_init_schema {
                SurrealDbVectorStore::from_client_with_init(
                    Arc::clone(&db),
                    db_config.clone(),
                    unified_config.vector.clone(),
                )
                .await?
            } else {
                SurrealDbVectorStore::from_client(
                    Arc::clone(&db),
                    db_config.clone(),
                    unified_config.vector.clone(),
                )?
            };
            Some(store)
        } else {
            None
        };

        // Create graph store if enabled
        let graph_store = if unified_config.enable_graph {
            let store = if db_config.auto_init_schema {
                SurrealDbGraphStore::from_client_with_init(
                    Arc::clone(&db),
                    db_config.clone(),
                    unified_config.graph.clone(),
                )
                .await?
            } else {
                SurrealDbGraphStore::from_client(
                    Arc::clone(&db),
                    db_config.clone(),
                    unified_config.graph.clone(),
                )?
            };
            Some(store)
        } else {
            None
        };

        Ok(Self {
            db,
            db_config,
            unified_config,
            storage,
            vector_store,
            graph_store,
        })
    }

    /// Create unified storage from an existing database connection
    ///
    /// Useful when you already have an established connection.
    pub fn from_client(
        db: Arc<Surreal<Any>>,
        db_config: SurrealDbConfig,
        unified_config: SurrealDbUnifiedConfig,
    ) -> Result<Self> {
        unified_config.validate()?;

        let storage = SurrealDbStorage::from_client(Arc::clone(&db), db_config.clone());

        let vector_store = if unified_config.enable_vector {
            Some(SurrealDbVectorStore::from_client(
                Arc::clone(&db),
                db_config.clone(),
                unified_config.vector.clone(),
            )?)
        } else {
            None
        };

        let graph_store = if unified_config.enable_graph {
            Some(SurrealDbGraphStore::from_client(
                Arc::clone(&db),
                db_config.clone(),
                unified_config.graph.clone(),
            )?)
        } else {
            None
        };

        Ok(Self {
            db,
            db_config,
            unified_config,
            storage,
            vector_store,
            graph_store,
        })
    }

    /// Create unified storage with schema initialization
    pub async fn from_client_with_init(
        db: Arc<Surreal<Any>>,
        db_config: SurrealDbConfig,
        unified_config: SurrealDbUnifiedConfig,
    ) -> Result<Self> {
        unified_config.validate()?;

        let storage =
            SurrealDbStorage::from_client_with_init(Arc::clone(&db), db_config.clone()).await?;

        let vector_store = if unified_config.enable_vector {
            Some(
                SurrealDbVectorStore::from_client_with_init(
                    Arc::clone(&db),
                    db_config.clone(),
                    unified_config.vector.clone(),
                )
                .await?,
            )
        } else {
            None
        };

        let graph_store = if unified_config.enable_graph {
            Some(
                SurrealDbGraphStore::from_client_with_init(
                    Arc::clone(&db),
                    db_config.clone(),
                    unified_config.graph.clone(),
                )
                .await?,
            )
        } else {
            None
        };

        Ok(Self {
            db,
            db_config,
            unified_config,
            storage,
            vector_store,
            graph_store,
        })
    }

    // =========================================================================
    // Accessors
    // =========================================================================

    /// Get a reference to the document/entity storage
    pub fn storage(&self) -> &SurrealDbStorage {
        &self.storage
    }

    /// Get a mutable reference to the document/entity storage
    pub fn storage_mut(&mut self) -> &mut SurrealDbStorage {
        &mut self.storage
    }

    /// Get a reference to the vector store (if enabled)
    pub fn vector_store(&self) -> Option<&SurrealDbVectorStore> {
        self.vector_store.as_ref()
    }

    /// Get a mutable reference to the vector store (if enabled)
    pub fn vector_store_mut(&mut self) -> Option<&mut SurrealDbVectorStore> {
        self.vector_store.as_mut()
    }

    /// Get a reference to the graph store (if enabled)
    pub fn graph_store(&self) -> Option<&SurrealDbGraphStore> {
        self.graph_store.as_ref()
    }

    /// Get a mutable reference to the graph store (if enabled)
    pub fn graph_store_mut(&mut self) -> Option<&mut SurrealDbGraphStore> {
        self.graph_store.as_mut()
    }

    /// Get the underlying database client
    pub fn client(&self) -> &Surreal<Any> {
        &self.db
    }

    /// Get a shared reference to the database client
    pub fn client_arc(&self) -> Arc<Surreal<Any>> {
        Arc::clone(&self.db)
    }

    /// Get the database configuration
    pub fn db_config(&self) -> &SurrealDbConfig {
        &self.db_config
    }

    /// Get the unified configuration
    pub fn unified_config(&self) -> &SurrealDbUnifiedConfig {
        &self.unified_config
    }

    /// Check if the vector store is enabled
    pub fn has_vector_store(&self) -> bool {
        self.vector_store.is_some()
    }

    /// Check if the graph store is enabled
    pub fn has_graph_store(&self) -> bool {
        self.graph_store.is_some()
    }

    // =========================================================================
    // Health & Maintenance
    // =========================================================================

    /// Perform a health check on all enabled stores
    pub async fn health_check(&self) -> Result<bool> {
        // Check basic connectivity
        self.db
            .query("RETURN true")
            .await
            .map_err(|e| GraphRAGError::Storage {
                message: format!("Health check failed: {}", e),
            })?;

        Ok(true)
    }

    /// Get statistics about all stores
    pub async fn stats(&self) -> UnifiedStorageStats {
        let graph_stats = if let Some(ref graph) = self.graph_store {
            Some(graph.stats().await)
        } else {
            None
        };

        // Count vectors if vector store is enabled
        let vector_count = if let Some(ref vector) = self.vector_store {
            vector.len().await
        } else {
            0
        };

        UnifiedStorageStats {
            vector_count,
            graph_stats,
        }
    }

    /// Flush all pending operations
    pub async fn flush(&self) -> Result<()> {
        // SurrealDB is transactional, so this is mainly for interface consistency
        Ok(())
    }
}

/// Statistics for the unified storage
#[derive(Debug, Clone)]
pub struct UnifiedStorageStats {
    /// Number of vectors in the vector store
    pub vector_count: usize,

    /// Graph statistics (if graph store is enabled)
    pub graph_stats: Option<crate::core::traits::GraphStats>,
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unified_config_default() {
        let config = SurrealDbUnifiedConfig::default();
        assert!(config.enable_vector);
        assert!(config.enable_graph);
        assert_eq!(config.vector.dimension, 384);
    }

    #[test]
    fn test_unified_config_with_dimension() {
        let config = SurrealDbUnifiedConfig::with_vector_dimension(768);
        assert_eq!(config.vector.dimension, 768);
    }

    #[test]
    fn test_unified_config_without_vector() {
        let config = SurrealDbUnifiedConfig::default().without_vector();
        assert!(!config.enable_vector);
        assert!(config.enable_graph);
    }

    #[test]
    fn test_unified_config_without_graph() {
        let config = SurrealDbUnifiedConfig::default().without_graph();
        assert!(config.enable_vector);
        assert!(!config.enable_graph);
    }

    #[test]
    fn test_unified_config_validate() {
        let config = SurrealDbUnifiedConfig::default();
        assert!(config.validate().is_ok());
    }

    #[tokio::test]
    async fn test_unified_storage_new() {
        let db_config = SurrealDbConfig::memory();
        let unified_config = SurrealDbUnifiedConfig::default();

        let storage = SurrealDbUnifiedStorage::new(db_config, unified_config)
            .await
            .unwrap();

        assert!(storage.has_vector_store());
        assert!(storage.has_graph_store());
    }

    #[tokio::test]
    async fn test_unified_storage_health_check() {
        let db_config = SurrealDbConfig::memory();
        let unified_config = SurrealDbUnifiedConfig::default();

        let storage = SurrealDbUnifiedStorage::new(db_config, unified_config)
            .await
            .unwrap();

        assert!(storage.health_check().await.unwrap());
    }

    #[tokio::test]
    async fn test_unified_storage_without_optional_stores() {
        let db_config = SurrealDbConfig::memory();
        let unified_config = SurrealDbUnifiedConfig::default()
            .without_vector()
            .without_graph();

        let storage = SurrealDbUnifiedStorage::new(db_config, unified_config)
            .await
            .unwrap();

        assert!(!storage.has_vector_store());
        assert!(!storage.has_graph_store());
    }

    #[tokio::test]
    async fn test_unified_storage_stats() {
        let db_config = SurrealDbConfig::memory();
        let unified_config = SurrealDbUnifiedConfig::default();

        let storage = SurrealDbUnifiedStorage::new(db_config, unified_config)
            .await
            .unwrap();

        let stats = storage.stats().await;
        assert_eq!(stats.vector_count, 0);
        assert!(stats.graph_stats.is_some());
    }
}
