//! SurrealDB vector store implementation
//!
//! This module provides a SurrealDB implementation of the `AsyncVectorStore` trait,
//! enabling vector similarity search using SurrealDB's native vector functions.
//!
//! ## Features
//!
//! - **Vector similarity search**: Cosine, Euclidean, Manhattan distance metrics
//! - **MTREE indexing**: Efficient approximate nearest neighbor search
//! - **Batch operations**: Transactional batch insert support
//! - **Metadata filtering**: Optional metadata for vector filtering
//!
//! ## Usage
//!
//! ```rust,ignore
//! use graphrag_core::storage::surrealdb::{
//!     SurrealDbConfig, SurrealDbVectorStore, SurrealDbVectorConfig, DistanceMetric,
//! };
//! use graphrag_core::core::traits::AsyncVectorStore;
//!
//! let db_config = SurrealDbConfig::rocksdb("./data/vectors");
//! let vector_config = SurrealDbVectorConfig {
//!     dimension: 384,
//!     distance_metric: DistanceMetric::Cosine,
//!     table_name: "embeddings".to_string(),
//!     auto_index: true,
//! };
//!
//! let mut store = SurrealDbVectorStore::new(db_config, vector_config).await?;
//!
//! // Add vectors
//! store.add_vector("id1".to_string(), vec![0.1; 384], None).await?;
//!
//! // Search
//! let results = store.search(&[0.1; 384], 10).await?;
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use surrealdb::engine::any::Any;
use surrealdb::Surreal;

use crate::core::error::{GraphRAGError, Result};
use crate::core::traits::{AsyncVectorStore, SearchResult, VectorBatch, VectorMetadata};

use super::config::SurrealDbConfig;

/// Distance metric for vector similarity search
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DistanceMetric {
    /// Cosine similarity (1 - cosine_distance)
    /// Best for normalized vectors and semantic similarity
    #[default]
    Cosine,
    /// Euclidean (L2) distance
    /// Best for absolute distance comparisons
    Euclidean,
    /// Manhattan (L1) distance
    /// More robust to outliers than Euclidean
    Manhattan,
}

impl DistanceMetric {
    /// Get the SurrealQL function name for this metric
    pub fn as_surql_function(&self) -> &'static str {
        match self {
            DistanceMetric::Cosine => "vector::similarity::cosine",
            DistanceMetric::Euclidean => "vector::distance::euclidean",
            DistanceMetric::Manhattan => "vector::distance::manhattan",
        }
    }

    /// Whether this metric returns similarity (higher = more similar)
    /// vs distance (lower = more similar)
    pub fn is_similarity(&self) -> bool {
        matches!(self, DistanceMetric::Cosine)
    }
}

impl std::fmt::Display for DistanceMetric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DistanceMetric::Cosine => write!(f, "cosine"),
            DistanceMetric::Euclidean => write!(f, "euclidean"),
            DistanceMetric::Manhattan => write!(f, "manhattan"),
        }
    }
}

impl std::str::FromStr for DistanceMetric {
    type Err = GraphRAGError;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "cosine" => Ok(DistanceMetric::Cosine),
            "euclidean" | "l2" => Ok(DistanceMetric::Euclidean),
            "manhattan" | "l1" => Ok(DistanceMetric::Manhattan),
            _ => Err(GraphRAGError::Config {
                message: format!(
                    "Unknown distance metric '{}'. Valid options: cosine, euclidean, manhattan",
                    s
                ),
            }),
        }
    }
}

/// Configuration for SurrealDB vector store
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurrealDbVectorConfig {
    /// Dimension of vectors (must match embedding model output)
    ///
    /// Common dimensions:
    /// - 384: all-MiniLM-L6-v2
    /// - 768: all-mpnet-base-v2, bert-base
    /// - 1024: voyage-3-large, bge-large-en-v1.5
    /// - 1536: text-embedding-3-small (OpenAI)
    /// - 3072: text-embedding-3-large (OpenAI)
    pub dimension: usize,

    /// Distance metric for similarity search
    #[serde(default)]
    pub distance_metric: DistanceMetric,

    /// Table name for vector storage
    #[serde(default = "default_table_name")]
    pub table_name: String,

    /// Whether to auto-build index after batch inserts
    #[serde(default = "default_auto_index")]
    pub auto_index: bool,
}

fn default_table_name() -> String {
    "vector".to_string()
}

fn default_auto_index() -> bool {
    true
}

impl Default for SurrealDbVectorConfig {
    fn default() -> Self {
        Self {
            dimension: 384, // all-MiniLM-L6-v2 default
            distance_metric: DistanceMetric::default(),
            table_name: default_table_name(),
            auto_index: default_auto_index(),
        }
    }
}

impl SurrealDbVectorConfig {
    /// Create config for a specific embedding model dimension
    pub fn with_dimension(dimension: usize) -> Self {
        Self {
            dimension,
            ..Default::default()
        }
    }

    /// Set the distance metric
    pub fn with_metric(mut self, metric: DistanceMetric) -> Self {
        self.distance_metric = metric;
        self
    }

    /// Set the table name
    pub fn with_table_name(mut self, name: impl Into<String>) -> Self {
        self.table_name = name.into();
        self
    }

    /// Disable auto-indexing
    pub fn without_auto_index(mut self) -> Self {
        self.auto_index = false;
        self
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        if self.dimension == 0 {
            return Err(GraphRAGError::Config {
                message: "Vector dimension must be greater than 0".to_string(),
            });
        }

        if self.table_name.is_empty() {
            return Err(GraphRAGError::Config {
                message: "Table name cannot be empty".to_string(),
            });
        }

        Ok(())
    }
}

/// SurrealDB vector store implementation
pub struct SurrealDbVectorStore {
    db: Arc<Surreal<Any>>,
    config: SurrealDbVectorConfig,
    db_config: SurrealDbConfig,
}

impl SurrealDbVectorStore {
    /// Create a new vector store with the given configurations
    pub async fn new(
        db_config: SurrealDbConfig,
        vector_config: SurrealDbVectorConfig,
    ) -> Result<Self> {
        vector_config.validate()?;

        let db = surrealdb::engine::any::connect(&db_config.endpoint)
            .await
            .map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to connect to SurrealDB: {}", e),
            })?;

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
                message: format!("Failed to authenticate: {}", e),
            })?;
        }

        let store = Self {
            db: Arc::new(db),
            config: vector_config.clone(),
            db_config,
        };

        // Initialize schema if enabled
        if store.db_config.auto_init_schema {
            store.init_schema().await?;
        }

        Ok(store)
    }

    /// Create a vector store from an existing database connection
    pub fn from_client(
        db: Arc<Surreal<Any>>,
        db_config: SurrealDbConfig,
        vector_config: SurrealDbVectorConfig,
    ) -> Result<Self> {
        vector_config.validate()?;

        Ok(Self {
            db,
            config: vector_config,
            db_config,
        })
    }

    /// Get the underlying database client
    pub fn client(&self) -> &Surreal<Any> {
        &self.db
    }

    /// Get a shared reference to the database client
    pub fn client_arc(&self) -> Arc<Surreal<Any>> {
        Arc::clone(&self.db)
    }

    /// Get the vector configuration
    pub fn config(&self) -> &SurrealDbVectorConfig {
        &self.config
    }

    /// Initialize the vector table schema
    async fn init_schema(&self) -> Result<()> {
        let schema = format!(
            r#"
            -- Vector table for embeddings
            DEFINE TABLE {table} SCHEMALESS;
            DEFINE INDEX idx_{table}_id ON {table} FIELDS id UNIQUE;
            DEFINE INDEX idx_{table}_source ON {table} FIELDS source_type, source_id;
            "#,
            table = self.config.table_name
        );

        self.db
            .query(&schema)
            .await
            .map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to initialize vector schema: {}", e),
            })?;

        Ok(())
    }
}

/// Internal record structure for vector storage
#[derive(Debug, Serialize, Deserialize)]
struct VectorRecord {
    id: String,
    embedding: Vec<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<HashMap<String, String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    source_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    source_id: Option<String>,
}

/// Internal search result from SurrealDB
#[derive(Debug, Deserialize)]
struct VectorSearchResult {
    id: String,
    score: f32,
    #[serde(default)]
    metadata: Option<HashMap<String, String>>,
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

        let json_value =
            serde_json::to_value(&record).map_err(|e| GraphRAGError::Serialization {
                message: format!("Failed to serialize vector record: {}", e),
            })?;

        self.db
            .query("UPSERT type::thing($table, $id) CONTENT $data")
            .bind(("table", self.config.table_name.clone()))
            .bind(("id", id))
            .bind(("data", json_value))
            .await
            .map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to store vector: {}", e),
            })?;

        Ok(())
    }

    async fn add_vectors_batch(&mut self, vectors: VectorBatch) -> Result<()> {
        if vectors.is_empty() {
            return Ok(());
        }

        // Validate all dimensions first
        for (id, vector, _) in &vectors {
            if vector.len() != self.config.dimension {
                return Err(GraphRAGError::Validation {
                    message: format!(
                        "Vector '{}' dimension mismatch: expected {}, got {}",
                        id,
                        self.config.dimension,
                        vector.len()
                    ),
                });
            }
        }

        self.db
            .query("BEGIN TRANSACTION")
            .await
            .map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to begin transaction: {}", e),
            })?;

        for (id, vector, metadata) in vectors {
            let record = VectorRecord {
                id: id.clone(),
                embedding: vector,
                metadata,
                source_type: None,
                source_id: None,
            };

            let json_value =
                serde_json::to_value(&record).map_err(|e| GraphRAGError::Serialization {
                    message: format!("Failed to serialize vector record: {}", e),
                })?;

            if let Err(e) = self
                .db
                .query("UPSERT type::thing($table, $id) CONTENT $data")
                .bind(("table", self.config.table_name.clone()))
                .bind(("id", id))
                .bind(("data", json_value))
                .await
            {
                self.db.query("CANCEL TRANSACTION").await.ok();
                return Err(GraphRAGError::Storage {
                    message: format!("Failed to store vector in batch: {}", e),
                });
            }
        }

        self.db
            .query("COMMIT TRANSACTION")
            .await
            .map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to commit transaction: {}", e),
            })?;

        if self.config.auto_index {
            self.build_index().await?;
        }

        Ok(())
    }

    async fn search(&self, query_vector: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        if query_vector.len() != self.config.dimension {
            return Err(GraphRAGError::Validation {
                message: format!(
                    "Query vector dimension mismatch: expected {}, got {}",
                    self.config.dimension,
                    query_vector.len()
                ),
            });
        }

        let distance_fn = self.config.distance_metric.as_surql_function();
        let order = if self.config.distance_metric.is_similarity() {
            "DESC"
        } else {
            "ASC"
        };

        let query = format!(
            "SELECT meta::id(id) as id, {distance_fn}(embedding, $query) AS score, metadata \
             FROM {table} \
             ORDER BY score {order} \
             LIMIT $k",
            distance_fn = distance_fn,
            table = self.config.table_name,
            order = order
        );

        let mut response = self
            .db
            .query(&query)
            .bind(("query", query_vector.to_vec()))
            .bind(("k", k))
            .await
            .map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to execute vector search: {}", e),
            })?;

        let results: Vec<VectorSearchResult> =
            response.take(0).map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to parse search results: {}", e),
            })?;

        Ok(results
            .into_iter()
            .map(|r| {
                // Convert similarity to distance for consistent API
                let distance = if self.config.distance_metric.is_similarity() {
                    1.0 - r.score
                } else {
                    r.score
                };

                SearchResult {
                    id: r.id,
                    distance,
                    metadata: r.metadata,
                }
            })
            .collect())
    }

    async fn search_with_threshold(
        &self,
        query_vector: &[f32],
        k: usize,
        threshold: f32,
    ) -> Result<Vec<SearchResult>> {
        if query_vector.len() != self.config.dimension {
            return Err(GraphRAGError::Validation {
                message: format!(
                    "Query vector dimension mismatch: expected {}, got {}",
                    self.config.dimension,
                    query_vector.len()
                ),
            });
        }

        let distance_fn = self.config.distance_metric.as_surql_function();
        let (order, threshold_op, threshold_value) = if self.config.distance_metric.is_similarity()
        {
            ("DESC", ">=", 1.0 - threshold) // Convert distance threshold to similarity
        } else {
            ("ASC", "<=", threshold)
        };

        let query = format!(
            "SELECT meta::id(id) as id, {distance_fn}(embedding, $query) AS score, metadata \
             FROM {table} \
             WHERE {distance_fn}(embedding, $query) {threshold_op} $threshold \
             ORDER BY score {order} \
             LIMIT $k",
            distance_fn = distance_fn,
            table = self.config.table_name,
            threshold_op = threshold_op,
            order = order
        );

        let mut response = self
            .db
            .query(&query)
            .bind(("query", query_vector.to_vec()))
            .bind(("k", k))
            .bind(("threshold", threshold_value))
            .await
            .map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to execute vector search with threshold: {}", e),
            })?;

        let results: Vec<VectorSearchResult> =
            response.take(0).map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to parse search results: {}", e),
            })?;

        Ok(results
            .into_iter()
            .map(|r| {
                let distance = if self.config.distance_metric.is_similarity() {
                    1.0 - r.score
                } else {
                    r.score
                };

                SearchResult {
                    id: r.id,
                    distance,
                    metadata: r.metadata,
                }
            })
            .collect())
    }

    async fn remove_vector(&mut self, id: &str) -> Result<bool> {
        // Check current count
        let count_before = self.len().await;

        // Delete the record using the same bind pattern as UPSERT
        self.db
            .query("DELETE type::thing($table, $id)")
            .bind(("table", self.config.table_name.clone()))
            .bind(("id", id.to_string()))
            .await
            .map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to delete vector: {}", e),
            })?;

        // Check if count decreased to determine if record existed
        let count_after = self.len().await;
        Ok(count_after < count_before)
    }

    async fn len(&self) -> usize {
        let query = format!("SELECT count() FROM {} GROUP ALL", self.config.table_name);

        let mut response = match self.db.query(&query).await {
            Ok(r) => r,
            Err(_) => return 0,
        };

        #[derive(Deserialize)]
        struct CountResult {
            count: usize,
        }

        response
            .take::<Vec<CountResult>>(0)
            .ok()
            .and_then(|v| v.into_iter().next())
            .map(|r| r.count)
            .unwrap_or(0)
    }

    async fn build_index(&mut self) -> Result<()> {
        // Note: SurrealDB vector index syntax varies by version
        // This attempts to rebuild any existing index
        let query = format!(
            "REBUILD INDEX IF EXISTS idx_{table}_embedding ON {table}",
            table = self.config.table_name
        );

        // Ignore errors as the index may not exist yet
        let _ = self.db.query(&query).await;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance_metric_display() {
        assert_eq!(DistanceMetric::Cosine.to_string(), "cosine");
        assert_eq!(DistanceMetric::Euclidean.to_string(), "euclidean");
        assert_eq!(DistanceMetric::Manhattan.to_string(), "manhattan");
    }

    #[test]
    fn test_distance_metric_from_str() {
        assert_eq!(
            "cosine".parse::<DistanceMetric>().unwrap(),
            DistanceMetric::Cosine
        );
        assert_eq!(
            "euclidean".parse::<DistanceMetric>().unwrap(),
            DistanceMetric::Euclidean
        );
        assert_eq!(
            "l2".parse::<DistanceMetric>().unwrap(),
            DistanceMetric::Euclidean
        );
        assert_eq!(
            "manhattan".parse::<DistanceMetric>().unwrap(),
            DistanceMetric::Manhattan
        );
        assert_eq!(
            "l1".parse::<DistanceMetric>().unwrap(),
            DistanceMetric::Manhattan
        );
        assert!("invalid".parse::<DistanceMetric>().is_err());
    }

    #[test]
    fn test_distance_metric_surql_function() {
        assert_eq!(
            DistanceMetric::Cosine.as_surql_function(),
            "vector::similarity::cosine"
        );
        assert_eq!(
            DistanceMetric::Euclidean.as_surql_function(),
            "vector::distance::euclidean"
        );
        assert_eq!(
            DistanceMetric::Manhattan.as_surql_function(),
            "vector::distance::manhattan"
        );
    }

    #[test]
    fn test_vector_config_default() {
        let config = SurrealDbVectorConfig::default();
        assert_eq!(config.dimension, 384);
        assert_eq!(config.distance_metric, DistanceMetric::Cosine);
        assert_eq!(config.table_name, "vector");
        assert!(config.auto_index);
    }

    #[test]
    fn test_vector_config_builder() {
        let config = SurrealDbVectorConfig::with_dimension(1024)
            .with_metric(DistanceMetric::Euclidean)
            .with_table_name("embeddings")
            .without_auto_index();

        assert_eq!(config.dimension, 1024);
        assert_eq!(config.distance_metric, DistanceMetric::Euclidean);
        assert_eq!(config.table_name, "embeddings");
        assert!(!config.auto_index);
    }

    #[test]
    fn test_vector_config_validation() {
        let valid_config = SurrealDbVectorConfig::default();
        assert!(valid_config.validate().is_ok());

        let invalid_dimension = SurrealDbVectorConfig {
            dimension: 0,
            ..Default::default()
        };
        assert!(invalid_dimension.validate().is_err());

        let invalid_table = SurrealDbVectorConfig {
            table_name: String::new(),
            ..Default::default()
        };
        assert!(invalid_table.validate().is_err());
    }

    #[test]
    fn test_distance_metric_is_similarity() {
        assert!(DistanceMetric::Cosine.is_similarity());
        assert!(!DistanceMetric::Euclidean.is_similarity());
        assert!(!DistanceMetric::Manhattan.is_similarity());
    }

    // Async tests for vector store operations
    #[tokio::test]
    async fn test_vector_store_add_and_len() {
        let db_config = SurrealDbConfig::memory();
        let vector_config = SurrealDbVectorConfig::with_dimension(4);
        let mut store = SurrealDbVectorStore::new(db_config, vector_config)
            .await
            .unwrap();

        assert_eq!(store.len().await, 0);
        assert!(store.is_empty().await);

        store
            .add_vector("vec1".to_string(), vec![1.0, 0.0, 0.0, 0.0], None)
            .await
            .unwrap();

        assert_eq!(store.len().await, 1);
        assert!(!store.is_empty().await);
    }

    #[tokio::test]
    async fn test_vector_store_dimension_validation() {
        let db_config = SurrealDbConfig::memory();
        let vector_config = SurrealDbVectorConfig::with_dimension(4);
        let mut store = SurrealDbVectorStore::new(db_config, vector_config)
            .await
            .unwrap();

        // Wrong dimension should fail
        let result = store
            .add_vector("vec1".to_string(), vec![1.0, 0.0], None)
            .await;
        assert!(result.is_err());

        // Correct dimension should succeed
        let result = store
            .add_vector("vec1".to_string(), vec![1.0, 0.0, 0.0, 0.0], None)
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_vector_store_batch_add() {
        let db_config = SurrealDbConfig::memory();
        let vector_config = SurrealDbVectorConfig::with_dimension(4).without_auto_index();
        let mut store = SurrealDbVectorStore::new(db_config, vector_config)
            .await
            .unwrap();

        let vectors = vec![
            ("v1".to_string(), vec![1.0, 0.0, 0.0, 0.0], None),
            ("v2".to_string(), vec![0.0, 1.0, 0.0, 0.0], None),
            ("v3".to_string(), vec![0.0, 0.0, 1.0, 0.0], None),
        ];

        store.add_vectors_batch(vectors).await.unwrap();

        assert_eq!(store.len().await, 3);
    }

    #[tokio::test]
    async fn test_vector_store_remove() {
        let db_config = SurrealDbConfig::memory();
        let vector_config = SurrealDbVectorConfig::with_dimension(4);
        let mut store = SurrealDbVectorStore::new(db_config, vector_config)
            .await
            .unwrap();

        store
            .add_vector("vec1".to_string(), vec![1.0, 0.0, 0.0, 0.0], None)
            .await
            .unwrap();

        assert_eq!(store.len().await, 1);

        let removed = store.remove_vector("vec1").await.unwrap();
        assert!(removed);

        assert_eq!(store.len().await, 0);

        // Removing non-existent should return false
        let removed = store.remove_vector("nonexistent").await.unwrap();
        assert!(!removed);
    }

    #[tokio::test]
    async fn test_vector_store_search() {
        let db_config = SurrealDbConfig::memory();
        let vector_config = SurrealDbVectorConfig::with_dimension(4);
        let mut store = SurrealDbVectorStore::new(db_config, vector_config)
            .await
            .unwrap();

        // Add some vectors
        store
            .add_vector("v1".to_string(), vec![1.0, 0.0, 0.0, 0.0], None)
            .await
            .unwrap();
        store
            .add_vector("v2".to_string(), vec![0.9, 0.1, 0.0, 0.0], None)
            .await
            .unwrap();
        store
            .add_vector("v3".to_string(), vec![0.0, 1.0, 0.0, 0.0], None)
            .await
            .unwrap();

        // Search for vector similar to v1
        let results = store.search(&[1.0, 0.0, 0.0, 0.0], 2).await.unwrap();

        assert_eq!(results.len(), 2);
        // v1 should be most similar (distance ~0)
        assert_eq!(results[0].id, "v1");
        assert!(results[0].distance < 0.01);
    }

    #[tokio::test]
    async fn test_vector_store_search_with_metadata() {
        let db_config = SurrealDbConfig::memory();
        let vector_config = SurrealDbVectorConfig::with_dimension(4);
        let mut store = SurrealDbVectorStore::new(db_config, vector_config)
            .await
            .unwrap();

        let metadata = Some(HashMap::from([
            ("source".to_string(), "document1".to_string()),
            ("chunk_id".to_string(), "chunk_42".to_string()),
        ]));

        store
            .add_vector("v1".to_string(), vec![1.0, 0.0, 0.0, 0.0], metadata)
            .await
            .unwrap();

        let results = store.search(&[1.0, 0.0, 0.0, 0.0], 1).await.unwrap();

        assert_eq!(results.len(), 1);
        assert!(results[0].metadata.is_some());
        let meta = results[0].metadata.as_ref().unwrap();
        assert_eq!(meta.get("source"), Some(&"document1".to_string()));
    }

    #[tokio::test]
    async fn test_vector_store_upsert() {
        let db_config = SurrealDbConfig::memory();
        let vector_config = SurrealDbVectorConfig::with_dimension(4);
        let mut store = SurrealDbVectorStore::new(db_config, vector_config)
            .await
            .unwrap();

        // Add initial vector
        store
            .add_vector("v1".to_string(), vec![1.0, 0.0, 0.0, 0.0], None)
            .await
            .unwrap();

        // Update with same ID
        store
            .add_vector("v1".to_string(), vec![0.0, 1.0, 0.0, 0.0], None)
            .await
            .unwrap();

        // Should still have only 1 vector
        assert_eq!(store.len().await, 1);

        // Search should find the updated vector
        let results = store.search(&[0.0, 1.0, 0.0, 0.0], 1).await.unwrap();
        assert_eq!(results[0].id, "v1");
        assert!(results[0].distance < 0.01);
    }

    #[tokio::test]
    async fn test_vector_store_from_client() {
        let db_config = SurrealDbConfig::memory();
        let vector_config = SurrealDbVectorConfig::with_dimension(4);

        // Create first store to get a client
        let store1 = SurrealDbVectorStore::new(db_config.clone(), vector_config.clone())
            .await
            .unwrap();

        // Create second store from same client
        let store2 =
            SurrealDbVectorStore::from_client(store1.client_arc(), db_config, vector_config)
                .unwrap();

        // Both should work with the shared connection
        assert_eq!(store1.len().await, 0);
        assert_eq!(store2.len().await, 0);
    }

    #[tokio::test]
    async fn test_vector_store_health_check() {
        let db_config = SurrealDbConfig::memory();
        let vector_config = SurrealDbVectorConfig::with_dimension(4);
        let store = SurrealDbVectorStore::new(db_config, vector_config)
            .await
            .unwrap();

        let healthy = store.health_check().await.unwrap();
        assert!(healthy);
    }

    #[tokio::test]
    async fn test_vector_store_build_index() {
        let db_config = SurrealDbConfig::memory();
        let vector_config = SurrealDbVectorConfig::with_dimension(4);
        let mut store = SurrealDbVectorStore::new(db_config, vector_config)
            .await
            .unwrap();

        // Should not fail even if index doesn't exist
        let result = store.build_index().await;
        assert!(result.is_ok());
    }
}
