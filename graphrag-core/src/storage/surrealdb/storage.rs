//! SurrealDB AsyncStorage implementation
//!
//! This module provides the `SurrealDbStorage` type which implements
//! the `AsyncStorage` trait for persisting GraphRAG data to SurrealDB.

use std::sync::Arc;

use async_trait::async_trait;
use surrealdb::engine::any::Any;
use surrealdb::Surreal;

use crate::core::{traits::AsyncStorage, Document, Entity, GraphRAGError, Result, TextChunk};

use super::config::SurrealDbConfig;

/// SurrealDB implementation of AsyncStorage
///
/// Provides persistent storage for documents, entities, and chunks
/// using SurrealDB's multi-model database capabilities.
///
/// # Example
///
/// ```rust,ignore
/// use graphrag_core::storage::surrealdb::{SurrealDbConfig, SurrealDbStorage};
///
/// let config = SurrealDbConfig::memory();
/// let mut storage = SurrealDbStorage::new(config).await?;
///
/// // Store an entity
/// let entity = Entity { /* ... */ };
/// let id = storage.store_entity(entity).await?;
/// ```
pub struct SurrealDbStorage {
    db: Arc<Surreal<Any>>,
    #[allow(dead_code)]
    config: SurrealDbConfig,
}

impl std::fmt::Debug for SurrealDbStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SurrealDbStorage")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl SurrealDbStorage {
    /// Create a new SurrealDB storage instance
    ///
    /// Connects to the database using the provided configuration,
    /// selects the namespace and database, authenticates if credentials
    /// are provided, and optionally initializes the schema.
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration for the SurrealDB connection
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Connection to SurrealDB fails
    /// - Namespace/database selection fails
    /// - Authentication fails (if credentials provided)
    /// - Schema initialization fails (if auto_init_schema is true)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let config = SurrealDbConfig::memory();
    /// let storage = SurrealDbStorage::new(config).await?;
    /// ```
    pub async fn new(config: SurrealDbConfig) -> Result<Self> {
        // Use surrealdb::engine::any::connect for dynamic endpoint parsing
        let db = surrealdb::engine::any::connect(&config.endpoint)
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
    ///
    /// Creates the required tables and indexes for storing
    /// documents, entities, chunks, and relationships.
    async fn init_schema(&self) -> Result<()> {
        let schema = include_str!("schema.surql");
        self.db
            .query(schema)
            .await
            .map_err(|e| GraphRAGError::Storage {
                message: format!("Schema initialization failed: {}", e),
            })?;
        Ok(())
    }

    /// Create a storage instance from an existing database connection
    ///
    /// This is useful when you want to share a single database connection
    /// across multiple storage instances (e.g., with vector store and graph store).
    ///
    /// # Arguments
    ///
    /// * `db` - Arc-wrapped SurrealDB client
    /// * `config` - Configuration (used for reference, connection already established)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let config = SurrealDbConfig::memory();
    /// let storage = SurrealDbStorage::new(config.clone()).await?;
    /// let shared_client = storage.client_arc();
    ///
    /// // Create another storage instance sharing the same connection
    /// let storage2 = SurrealDbStorage::from_client(shared_client, config);
    /// ```
    pub fn from_client(db: Arc<Surreal<Any>>, config: SurrealDbConfig) -> Self {
        Self { db, config }
    }

    /// Create a storage instance from an existing connection with schema init
    ///
    /// Like `from_client`, but also initializes the schema if configured.
    pub async fn from_client_with_init(
        db: Arc<Surreal<Any>>,
        config: SurrealDbConfig,
    ) -> Result<Self> {
        let storage = Self::from_client(db, config);

        if storage.config.auto_init_schema {
            storage.init_schema().await?;
        }

        Ok(storage)
    }

    /// Get reference to the underlying SurrealDB client
    ///
    /// This allows advanced operations not covered by the AsyncStorage trait.
    pub fn client(&self) -> &Surreal<Any> {
        &self.db
    }

    /// Get a clone of the Arc-wrapped client for sharing across tasks
    pub fn client_arc(&self) -> Arc<Surreal<Any>> {
        Arc::clone(&self.db)
    }

    /// Get the configuration
    pub fn config(&self) -> &SurrealDbConfig {
        &self.config
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

        // Serialize to JSON string first, then use raw SQL to insert
        // This avoids SurrealDB's serializer limitations with complex Rust types
        let json_str = serde_json::to_string(&entity).map_err(|e| GraphRAGError::Storage {
            message: format!("Failed to serialize entity: {}", e),
        })?;

        // Use UPSERT with JSON parsing for idempotent storage
        self.db
            .query("UPSERT type::thing('entity', $id) CONTENT $data")
            .bind(("id", id.clone()))
            .bind((
                "data",
                serde_json::from_str::<serde_json::Value>(&json_str).unwrap(),
            ))
            .await
            .map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to store entity: {}", e),
            })?;

        Ok(id)
    }

    async fn retrieve_entity(&self, id: &str) -> Result<Option<Self::Entity>> {
        // Use raw query to avoid SurrealDB's Thing type in the id field
        // Convert to owned String for 'static lifetime requirement
        let id_owned = id.to_string();
        let mut response = self
            .db
            .query("SELECT *, meta::id(id) as id FROM type::thing('entity', $id)")
            .bind(("id", id_owned))
            .await
            .map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to retrieve entity: {}", e),
            })?;

        // Get the result as JSON and manually parse to handle SurrealDB's Thing type
        let results: Vec<serde_json::Value> =
            response.take(0).map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to parse entity result: {}", e),
            })?;

        if results.is_empty() {
            return Ok(None);
        }

        // Parse the JSON value into our Entity type
        let entity: Entity =
            serde_json::from_value(results.into_iter().next().unwrap()).map_err(|e| {
                GraphRAGError::Storage {
                    message: format!("Failed to deserialize entity: {}", e),
                }
            })?;

        Ok(Some(entity))
    }

    async fn store_document(&mut self, document: Self::Document) -> Result<String> {
        let id = document.id.to_string();

        // Serialize to JSON string first, then use raw SQL to insert
        let json_str = serde_json::to_string(&document).map_err(|e| GraphRAGError::Storage {
            message: format!("Failed to serialize document: {}", e),
        })?;

        // Use UPSERT with JSON parsing for idempotent storage
        self.db
            .query("UPSERT type::thing('document', $id) CONTENT $data")
            .bind(("id", id.clone()))
            .bind((
                "data",
                serde_json::from_str::<serde_json::Value>(&json_str).unwrap(),
            ))
            .await
            .map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to store document: {}", e),
            })?;

        Ok(id)
    }

    async fn retrieve_document(&self, id: &str) -> Result<Option<Self::Document>> {
        // Use raw query to avoid SurrealDB's Thing type in the id field
        let id_owned = id.to_string();
        let mut response = self
            .db
            .query("SELECT *, meta::id(id) as id FROM type::thing('document', $id)")
            .bind(("id", id_owned))
            .await
            .map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to retrieve document: {}", e),
            })?;

        let results: Vec<serde_json::Value> =
            response.take(0).map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to parse document result: {}", e),
            })?;

        if results.is_empty() {
            return Ok(None);
        }

        let document: Document = serde_json::from_value(results.into_iter().next().unwrap())
            .map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to deserialize document: {}", e),
            })?;

        Ok(Some(document))
    }

    async fn store_chunk(&mut self, chunk: Self::Chunk) -> Result<String> {
        let id = chunk.id.to_string();

        // Serialize to JSON string first, then use raw SQL to insert
        let json_str = serde_json::to_string(&chunk).map_err(|e| GraphRAGError::Storage {
            message: format!("Failed to serialize chunk: {}", e),
        })?;

        // Use UPSERT with JSON parsing for idempotent storage
        self.db
            .query("UPSERT type::thing('chunk', $id) CONTENT $data")
            .bind(("id", id.clone()))
            .bind((
                "data",
                serde_json::from_str::<serde_json::Value>(&json_str).unwrap(),
            ))
            .await
            .map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to store chunk: {}", e),
            })?;

        Ok(id)
    }

    async fn retrieve_chunk(&self, id: &str) -> Result<Option<Self::Chunk>> {
        // Use raw query to avoid SurrealDB's Thing type in the id field
        let id_owned = id.to_string();
        let mut response = self
            .db
            .query("SELECT *, meta::id(id) as id FROM type::thing('chunk', $id)")
            .bind(("id", id_owned))
            .await
            .map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to retrieve chunk: {}", e),
            })?;

        let results: Vec<serde_json::Value> =
            response.take(0).map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to parse chunk result: {}", e),
            })?;

        if results.is_empty() {
            return Ok(None);
        }

        let chunk: TextChunk = serde_json::from_value(results.into_iter().next().unwrap())
            .map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to deserialize chunk: {}", e),
            })?;

        Ok(Some(chunk))
    }

    async fn list_entities(&self) -> Result<Vec<String>> {
        // Query for the entity's own 'id' field, not the SurrealDB record id
        // Use meta::id to get the record ID as a string
        let mut response = self
            .db
            .query("SELECT meta::id(id) as entity_id FROM entity")
            .await
            .map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to list entities: {}", e),
            })?;

        // Parse as JSON to handle any type conversion
        let results: Vec<serde_json::Value> =
            response.take(0).map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to parse entity list: {}", e),
            })?;

        let ids: Vec<String> = results
            .into_iter()
            .filter_map(|v| {
                v.get("entity_id")
                    .and_then(|id| id.as_str())
                    .map(|s| s.to_string())
            })
            .collect();

        Ok(ids)
    }

    async fn store_entities_batch(&mut self, entities: Vec<Self::Entity>) -> Result<Vec<String>> {
        if entities.is_empty() {
            return Ok(Vec::new());
        }

        let mut ids = Vec::with_capacity(entities.len());

        // Use transaction for batch atomicity
        self.db
            .query("BEGIN TRANSACTION")
            .await
            .map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to begin transaction: {}", e),
            })?;

        for entity in entities {
            let id = entity.id.to_string();

            // Serialize to JSON string first
            let json_str = match serde_json::to_string(&entity) {
                Ok(s) => s,
                Err(e) => {
                    let _ = self.db.query("CANCEL TRANSACTION").await;
                    return Err(GraphRAGError::Storage {
                        message: format!("Failed to serialize entity: {}", e),
                    });
                },
            };

            let result = self
                .db
                .query("UPSERT type::thing('entity', $id) CONTENT $data")
                .bind(("id", id.clone()))
                .bind((
                    "data",
                    serde_json::from_str::<serde_json::Value>(&json_str).unwrap(),
                ))
                .await;

            match result {
                Ok(_) => {
                    ids.push(id);
                },
                Err(e) => {
                    // Attempt rollback
                    let _ = self.db.query("CANCEL TRANSACTION").await;
                    return Err(GraphRAGError::Storage {
                        message: format!("Failed to store entity in batch: {}", e),
                    });
                },
            }
        }

        self.db
            .query("COMMIT TRANSACTION")
            .await
            .map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to commit transaction: {}", e),
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
        // For RocksDB backend, writes are durable by default
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{ChunkId, ChunkMetadata, DocumentId, EntityId};
    use indexmap::IndexMap;

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
        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.name, "Test Entity");
        assert_eq!(retrieved.entity_type, "test");
    }

    #[tokio::test]
    async fn test_document_storage() {
        let config = SurrealDbConfig::memory();
        let mut storage = SurrealDbStorage::new(config).await.unwrap();

        let document = Document {
            id: DocumentId::new("doc1".to_string()),
            title: "Test Document".to_string(),
            content: "This is test content.".to_string(),
            metadata: IndexMap::new(),
            chunks: vec![],
        };

        let id = storage.store_document(document).await.unwrap();
        assert_eq!(id, "doc1");

        let retrieved = storage.retrieve_document(&id).await.unwrap();
        assert!(retrieved.is_some());
        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.title, "Test Document");
    }

    #[tokio::test]
    async fn test_chunk_storage() {
        let config = SurrealDbConfig::memory();
        let mut storage = SurrealDbStorage::new(config).await.unwrap();

        let chunk = TextChunk {
            id: ChunkId::new("chunk1".to_string()),
            document_id: DocumentId::new("doc1".to_string()),
            content: "This is a chunk.".to_string(),
            start_offset: 0,
            end_offset: 16,
            embedding: None,
            entities: vec![],
            metadata: ChunkMetadata::default(),
        };

        let id = storage.store_chunk(chunk).await.unwrap();
        assert_eq!(id, "chunk1");

        let retrieved = storage.retrieve_chunk(&id).await.unwrap();
        assert!(retrieved.is_some());
        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.content, "This is a chunk.");
    }

    #[tokio::test]
    async fn test_batch_entity_storage() {
        let config = SurrealDbConfig::memory();
        let mut storage = SurrealDbStorage::new(config).await.unwrap();

        let entities = vec![
            Entity {
                id: EntityId::new("batch1".to_string()),
                name: "Entity 1".to_string(),
                entity_type: "test".to_string(),
                confidence: 0.9,
                mentions: vec![],
                embedding: None,
            },
            Entity {
                id: EntityId::new("batch2".to_string()),
                name: "Entity 2".to_string(),
                entity_type: "test".to_string(),
                confidence: 0.8,
                mentions: vec![],
                embedding: None,
            },
        ];

        let ids = storage.store_entities_batch(entities).await.unwrap();
        assert_eq!(ids.len(), 2);
        assert_eq!(ids[0], "batch1");
        assert_eq!(ids[1], "batch2");

        // Verify both were stored
        let e1 = storage.retrieve_entity("batch1").await.unwrap();
        let e2 = storage.retrieve_entity("batch2").await.unwrap();
        assert!(e1.is_some());
        assert!(e2.is_some());
    }

    #[tokio::test]
    async fn test_list_entities() {
        let config = SurrealDbConfig::memory();
        let mut storage = SurrealDbStorage::new(config).await.unwrap();

        // Store some entities
        for i in 0..3 {
            let entity = Entity {
                id: EntityId::new(format!("list_entity_{}", i)),
                name: format!("Entity {}", i),
                entity_type: "test".to_string(),
                confidence: 0.9,
                mentions: vec![],
                embedding: None,
            };
            storage.store_entity(entity).await.unwrap();
        }

        let ids = storage.list_entities().await.unwrap();
        assert_eq!(ids.len(), 3);
    }

    #[tokio::test]
    async fn test_health_check() {
        let config = SurrealDbConfig::memory();
        let storage = SurrealDbStorage::new(config).await.unwrap();

        let healthy = storage.health_check().await.unwrap();
        assert!(healthy);
    }

    #[tokio::test]
    async fn test_retrieve_nonexistent() {
        let config = SurrealDbConfig::memory();
        let storage = SurrealDbStorage::new(config).await.unwrap();

        let entity = storage.retrieve_entity("nonexistent").await.unwrap();
        assert!(entity.is_none());

        let document = storage.retrieve_document("nonexistent").await.unwrap();
        assert!(document.is_none());

        let chunk = storage.retrieve_chunk("nonexistent").await.unwrap();
        assert!(chunk.is_none());
    }

    #[tokio::test]
    async fn test_upsert_semantics() {
        let config = SurrealDbConfig::memory();
        let mut storage = SurrealDbStorage::new(config).await.unwrap();

        // Store initial entity
        let entity = Entity {
            id: EntityId::new("upsert_test".to_string()),
            name: "Original Name".to_string(),
            entity_type: "test".to_string(),
            confidence: 0.5,
            mentions: vec![],
            embedding: None,
        };
        storage.store_entity(entity).await.unwrap();

        // Update with same ID
        let updated_entity = Entity {
            id: EntityId::new("upsert_test".to_string()),
            name: "Updated Name".to_string(),
            entity_type: "test".to_string(),
            confidence: 0.9,
            mentions: vec![],
            embedding: None,
        };
        storage.store_entity(updated_entity).await.unwrap();

        // Verify update
        let retrieved = storage.retrieve_entity("upsert_test").await.unwrap();
        assert!(retrieved.is_some());
        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.name, "Updated Name");
        assert!((retrieved.confidence - 0.9).abs() < f32::EPSILON);
    }
}
