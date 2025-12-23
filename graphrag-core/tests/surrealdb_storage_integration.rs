//! Integration Tests for SurrealDB Storage
//!
//! These tests verify the complete SurrealDB storage workflow including:
//! - Full CRUD operations across entity types
//! - Data consistency and persistence
//! - Concurrent access patterns
//! - Transaction behavior with batch operations
//!
//! Test Type: Integration Tests (SurrealDB in-memory backend)
//! Requirements: surrealdb-storage feature enabled

#![cfg(feature = "surrealdb-storage")]

use std::collections::HashMap;

use graphrag_core::core::{
    traits::{AsyncStorage, AsyncVectorStore},
    ChunkId, ChunkMetadata, Document, DocumentId, Entity, EntityId, TextChunk,
};
use graphrag_core::storage::surrealdb::{
    DistanceMetric, SurrealDbConfig, SurrealDbStorage, SurrealDbVectorConfig, SurrealDbVectorStore,
};
use indexmap::IndexMap;

/// Helper to create a test entity with the given ID
fn create_test_entity(id: &str, name: &str, entity_type: &str) -> Entity {
    Entity {
        id: EntityId::new(id.to_string()),
        name: name.to_string(),
        entity_type: entity_type.to_string(),
        confidence: 0.95,
        mentions: vec![],
        embedding: None,
    }
}

/// Helper to create a test document with the given ID
fn create_test_document(id: &str, title: &str, content: &str) -> Document {
    Document {
        id: DocumentId::new(id.to_string()),
        title: title.to_string(),
        content: content.to_string(),
        metadata: IndexMap::new(),
        chunks: vec![],
    }
}

/// Helper to create a test chunk with the given ID
fn create_test_chunk(id: &str, doc_id: &str, content: &str) -> TextChunk {
    TextChunk {
        id: ChunkId::new(id.to_string()),
        document_id: DocumentId::new(doc_id.to_string()),
        content: content.to_string(),
        start_offset: 0,
        end_offset: content.len(),
        embedding: None,
        entities: vec![],
        metadata: ChunkMetadata::default(),
    }
}

/// Test 1: Complete entity lifecycle
///
/// Verifies the full CRUD workflow for entities:
/// - Create entity
/// - Read entity back
/// - Update entity (upsert)
/// - Verify update applied
/// - Delete via overwrite (no explicit delete in trait)
#[tokio::test]
async fn test_entity_lifecycle() {
    let config = SurrealDbConfig::memory();
    let mut storage = SurrealDbStorage::new(config).await.unwrap();

    // Create
    let entity = create_test_entity("lifecycle_entity", "Test Entity", "person");
    let id = storage.store_entity(entity).await.unwrap();
    assert_eq!(id, "lifecycle_entity");

    // Read
    let retrieved = storage.retrieve_entity(&id).await.unwrap();
    assert!(retrieved.is_some());
    let retrieved = retrieved.unwrap();
    assert_eq!(retrieved.name, "Test Entity");
    assert_eq!(retrieved.entity_type, "person");

    // Update (upsert with same ID)
    let updated_entity = Entity {
        id: EntityId::new("lifecycle_entity".to_string()),
        name: "Updated Entity".to_string(),
        entity_type: "organization".to_string(),
        confidence: 0.99,
        mentions: vec![],
        embedding: Some(vec![0.1, 0.2, 0.3]),
    };
    storage.store_entity(updated_entity).await.unwrap();

    // Verify update
    let retrieved = storage.retrieve_entity(&id).await.unwrap().unwrap();
    assert_eq!(retrieved.name, "Updated Entity");
    assert_eq!(retrieved.entity_type, "organization");
    assert!((retrieved.confidence - 0.99).abs() < f32::EPSILON);
    assert!(retrieved.embedding.is_some());
    assert_eq!(retrieved.embedding.unwrap().len(), 3);
}

/// Test 2: Complete document lifecycle
///
/// Verifies the full CRUD workflow for documents including metadata.
#[tokio::test]
async fn test_document_lifecycle() {
    let config = SurrealDbConfig::memory();
    let mut storage = SurrealDbStorage::new(config).await.unwrap();

    // Create document with metadata
    let mut doc = create_test_document(
        "lifecycle_doc",
        "Test Document",
        "This is the document content for testing.",
    );
    doc.metadata
        .insert("author".to_string(), "Test Author".to_string());
    doc.metadata
        .insert("date".to_string(), "2024-01-01".to_string());

    let id = storage.store_document(doc).await.unwrap();
    assert_eq!(id, "lifecycle_doc");

    // Read and verify metadata preserved
    let retrieved = storage.retrieve_document(&id).await.unwrap().unwrap();
    assert_eq!(retrieved.title, "Test Document");
    assert_eq!(retrieved.metadata.get("author").unwrap(), "Test Author");
    assert_eq!(retrieved.metadata.get("date").unwrap(), "2024-01-01");

    // Update document
    let updated_doc = Document {
        id: DocumentId::new("lifecycle_doc".to_string()),
        title: "Updated Document".to_string(),
        content: "Updated content.".to_string(),
        metadata: IndexMap::new(),
        chunks: vec![],
    };
    storage.store_document(updated_doc).await.unwrap();

    // Verify update (metadata should be cleared)
    let retrieved = storage.retrieve_document(&id).await.unwrap().unwrap();
    assert_eq!(retrieved.title, "Updated Document");
    assert!(retrieved.metadata.is_empty());
}

/// Test 3: Complete chunk lifecycle with metadata
///
/// Verifies the full CRUD workflow for chunks including ChunkMetadata.
#[tokio::test]
async fn test_chunk_lifecycle() {
    let config = SurrealDbConfig::memory();
    let mut storage = SurrealDbStorage::new(config).await.unwrap();

    // Create chunk with rich metadata
    let mut chunk = create_test_chunk("lifecycle_chunk", "doc1", "This is chunk content.");
    chunk.metadata = ChunkMetadata::new()
        .with_chapter("Chapter 1".to_string())
        .with_section("Section 1.1".to_string())
        .with_keywords(vec!["test".to_string(), "chunk".to_string()])
        .with_position(0.25);
    chunk.entities = vec![
        EntityId::new("entity1".to_string()),
        EntityId::new("entity2".to_string()),
    ];

    let id = storage.store_chunk(chunk).await.unwrap();
    assert_eq!(id, "lifecycle_chunk");

    // Read and verify metadata preserved
    let retrieved = storage.retrieve_chunk(&id).await.unwrap().unwrap();
    assert_eq!(retrieved.content, "This is chunk content.");
    assert_eq!(retrieved.metadata.chapter, Some("Chapter 1".to_string()));
    assert_eq!(retrieved.metadata.section, Some("Section 1.1".to_string()));
    assert_eq!(retrieved.metadata.keywords.len(), 2);
    assert_eq!(retrieved.metadata.position_in_document, Some(0.25));
    assert_eq!(retrieved.entities.len(), 2);
}

/// Test 4: Cross-reference integrity
///
/// Verifies that relationships between documents, chunks, and entities
/// are maintained correctly through storage operations.
#[tokio::test]
async fn test_cross_reference_integrity() {
    let config = SurrealDbConfig::memory();
    let mut storage = SurrealDbStorage::new(config).await.unwrap();

    // Create a document
    let doc = create_test_document(
        "crossref_doc",
        "Cross Reference Test",
        "Document for testing cross-references.",
    );
    storage.store_document(doc).await.unwrap();

    // Create entities
    let entity1 = create_test_entity("crossref_entity1", "Entity One", "person");
    let entity2 = create_test_entity("crossref_entity2", "Entity Two", "location");
    storage.store_entity(entity1).await.unwrap();
    storage.store_entity(entity2).await.unwrap();

    // Create chunk that references the document and entities
    let mut chunk = create_test_chunk(
        "crossref_chunk",
        "crossref_doc",
        "Mentions Entity One and Entity Two.",
    );
    chunk.entities = vec![
        EntityId::new("crossref_entity1".to_string()),
        EntityId::new("crossref_entity2".to_string()),
    ];
    storage.store_chunk(chunk).await.unwrap();

    // Retrieve and verify all references intact
    let retrieved_chunk = storage
        .retrieve_chunk("crossref_chunk")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(retrieved_chunk.document_id.0, "crossref_doc");
    assert_eq!(retrieved_chunk.entities.len(), 2);
    assert!(retrieved_chunk
        .entities
        .iter()
        .any(|e| e.0 == "crossref_entity1"));
    assert!(retrieved_chunk
        .entities
        .iter()
        .any(|e| e.0 == "crossref_entity2"));

    // Verify we can still retrieve the referenced entities
    let e1 = storage
        .retrieve_entity("crossref_entity1")
        .await
        .unwrap()
        .unwrap();
    let e2 = storage
        .retrieve_entity("crossref_entity2")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(e1.name, "Entity One");
    assert_eq!(e2.name, "Entity Two");
}

/// Test 5: Batch operations with transaction semantics
///
/// Verifies that batch entity storage works correctly and
/// provides atomic behavior.
#[tokio::test]
async fn test_batch_operations() {
    let config = SurrealDbConfig::memory();
    let mut storage = SurrealDbStorage::new(config).await.unwrap();

    // Create batch of entities
    let entities: Vec<Entity> = (0..10)
        .map(|i| create_test_entity(&format!("batch_{}", i), &format!("Entity {}", i), "test"))
        .collect();

    let ids = storage.store_entities_batch(entities).await.unwrap();
    assert_eq!(ids.len(), 10);

    // Verify all entities stored
    for i in 0..10 {
        let id = format!("batch_{}", i);
        let entity = storage.retrieve_entity(&id).await.unwrap();
        assert!(entity.is_some(), "Entity {} should exist", id);
        assert_eq!(entity.unwrap().name, format!("Entity {}", i));
    }

    // Verify list_entities returns all
    let all_ids = storage.list_entities().await.unwrap();
    assert_eq!(all_ids.len(), 10);
}

/// Test 6: Large batch performance
///
/// Verifies that larger batches are handled correctly.
#[tokio::test]
async fn test_large_batch() {
    let config = SurrealDbConfig::memory();
    let mut storage = SurrealDbStorage::new(config).await.unwrap();

    // Create larger batch
    let entities: Vec<Entity> = (0..100)
        .map(|i| {
            Entity {
                id: EntityId::new(format!("large_batch_{}", i)),
                name: format!("Entity {}", i),
                entity_type: "test".to_string(),
                confidence: 0.9,
                mentions: vec![],
                embedding: Some(vec![0.1; 384]), // Typical embedding size
            }
        })
        .collect();

    let ids = storage.store_entities_batch(entities).await.unwrap();
    assert_eq!(ids.len(), 100);

    // Sample verification
    let entity = storage
        .retrieve_entity("large_batch_50")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(entity.name, "Entity 50");
    assert!(entity.embedding.is_some());
    assert_eq!(entity.embedding.unwrap().len(), 384);
}

/// Test 7: Concurrent access simulation
///
/// Simulates concurrent read/write access patterns.
#[tokio::test]
async fn test_concurrent_access() {
    let config = SurrealDbConfig::memory();
    let mut storage = SurrealDbStorage::new(config).await.unwrap();

    // Pre-populate some data
    for i in 0..5 {
        let entity = create_test_entity(
            &format!("concurrent_{}", i),
            &format!("Entity {}", i),
            "test",
        );
        storage.store_entity(entity).await.unwrap();
    }

    // Simulate concurrent reads (sequential in this test, but verifies no corruption)
    for _ in 0..10 {
        for i in 0..5 {
            let id = format!("concurrent_{}", i);
            let entity = storage.retrieve_entity(&id).await.unwrap();
            assert!(entity.is_some());
        }
    }

    // Interleaved read/write
    for i in 5..10 {
        // Write new entity
        let entity = create_test_entity(
            &format!("concurrent_{}", i),
            &format!("Entity {}", i),
            "test",
        );
        storage.store_entity(entity).await.unwrap();

        // Read back immediately
        let retrieved = storage
            .retrieve_entity(&format!("concurrent_{}", i))
            .await
            .unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().name, format!("Entity {}", i));
    }

    // Final count
    let all_ids = storage.list_entities().await.unwrap();
    assert_eq!(all_ids.len(), 10);
}

/// Test 8: Health check reliability
///
/// Verifies health check works consistently.
#[tokio::test]
async fn test_health_check_reliability() {
    let config = SurrealDbConfig::memory();
    let storage = SurrealDbStorage::new(config).await.unwrap();

    // Multiple health checks should all succeed
    for _ in 0..5 {
        let healthy = storage.health_check().await.unwrap();
        assert!(healthy);
    }
}

/// Test 9: Empty storage behavior
///
/// Verifies correct behavior when storage is empty.
#[tokio::test]
async fn test_empty_storage_behavior() {
    let config = SurrealDbConfig::memory();
    let storage = SurrealDbStorage::new(config).await.unwrap();

    // Retrieving non-existent items should return None, not error
    let entity = storage.retrieve_entity("nonexistent").await.unwrap();
    assert!(entity.is_none());

    let document = storage.retrieve_document("nonexistent").await.unwrap();
    assert!(document.is_none());

    let chunk = storage.retrieve_chunk("nonexistent").await.unwrap();
    assert!(chunk.is_none());

    // List should return empty vec
    let entities = storage.list_entities().await.unwrap();
    assert!(entities.is_empty());
}

/// Test 10: Special characters in IDs and content
///
/// Verifies handling of special characters and unicode.
#[tokio::test]
async fn test_special_characters() {
    let config = SurrealDbConfig::memory();
    let mut storage = SurrealDbStorage::new(config).await.unwrap();

    // Entity with unicode name
    let entity = Entity {
        id: EntityId::new("unicode_entity".to_string()),
        name: "日本語テスト Entity 🚀".to_string(),
        entity_type: "test_type_with_underscore".to_string(),
        confidence: 0.95,
        mentions: vec![],
        embedding: None,
    };
    storage.store_entity(entity).await.unwrap();

    let retrieved = storage
        .retrieve_entity("unicode_entity")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(retrieved.name, "日本語テスト Entity 🚀");

    // Document with special content
    let doc = Document {
        id: DocumentId::new("special_doc".to_string()),
        title: "Test \"Quotes\" & <Brackets>".to_string(),
        content: "Content with\nnewlines\tand\ttabs\n\nAnd émojis 🎉".to_string(),
        metadata: IndexMap::new(),
        chunks: vec![],
    };
    storage.store_document(doc).await.unwrap();

    let retrieved = storage
        .retrieve_document("special_doc")
        .await
        .unwrap()
        .unwrap();
    assert!(retrieved.title.contains("\"Quotes\""));
    assert!(retrieved.content.contains("émojis 🎉"));
}

/// Test 11: Flush operation
///
/// Verifies flush completes without error (no-op for SurrealDB).
#[tokio::test]
async fn test_flush_operation() {
    let config = SurrealDbConfig::memory();
    let mut storage = SurrealDbStorage::new(config).await.unwrap();

    // Store some data
    let entity = create_test_entity("flush_test", "Test", "test");
    storage.store_entity(entity).await.unwrap();

    // Flush should succeed
    storage.flush().await.unwrap();

    // Data should still be accessible
    let retrieved = storage.retrieve_entity("flush_test").await.unwrap();
    assert!(retrieved.is_some());
}

/// Test 12: Schema auto-initialization
///
/// Verifies that auto_init_schema works correctly.
#[tokio::test]
async fn test_schema_auto_init() {
    // With auto_init_schema = true (default)
    let config = SurrealDbConfig::memory();
    assert!(config.auto_init_schema);

    let mut storage = SurrealDbStorage::new(config).await.unwrap();

    // Should be able to store without manual schema setup
    let entity = create_test_entity("schema_test", "Test", "test");
    let result = storage.store_entity(entity).await;
    assert!(result.is_ok());
}

/// Test 13: Multiple storage instances (isolated databases)
///
/// Verifies that different storage instances with different
/// database names are isolated.
#[tokio::test]
async fn test_isolated_databases() {
    // Create two separate databases
    let config1 = SurrealDbConfig::memory()
        .with_namespace("test".to_string())
        .with_database("db1".to_string());
    let config2 = SurrealDbConfig::memory()
        .with_namespace("test".to_string())
        .with_database("db2".to_string());

    let mut storage1 = SurrealDbStorage::new(config1).await.unwrap();
    let mut storage2 = SurrealDbStorage::new(config2).await.unwrap();

    // Store entity in storage1
    let entity1 = create_test_entity("isolated_entity", "Entity in DB1", "test");
    storage1.store_entity(entity1).await.unwrap();

    // Store different entity with same ID in storage2
    let entity2 = create_test_entity("isolated_entity", "Entity in DB2", "test");
    storage2.store_entity(entity2).await.unwrap();

    // Verify isolation
    let retrieved1 = storage1
        .retrieve_entity("isolated_entity")
        .await
        .unwrap()
        .unwrap();
    let retrieved2 = storage2
        .retrieve_entity("isolated_entity")
        .await
        .unwrap()
        .unwrap();

    assert_eq!(retrieved1.name, "Entity in DB1");
    assert_eq!(retrieved2.name, "Entity in DB2");
}

// =============================================================================
// Vector Store Integration Tests
// =============================================================================

/// Helper to create a test SurrealDbVectorStore with in-memory database
async fn create_test_vector_store(dimension: usize) -> SurrealDbVectorStore {
    let db_config = SurrealDbConfig::memory();
    let vector_config = SurrealDbVectorConfig::with_dimension(dimension);
    SurrealDbVectorStore::new(db_config, vector_config)
        .await
        .unwrap()
}

/// Helper to create a test SurrealDbVectorStore with custom config
async fn create_test_vector_store_with_config(
    dimension: usize,
    metric: DistanceMetric,
    table_name: &str,
) -> SurrealDbVectorStore {
    let db_config = SurrealDbConfig::memory();
    let vector_config = SurrealDbVectorConfig::with_dimension(dimension)
        .with_metric(metric)
        .with_table_name(table_name.to_string());
    SurrealDbVectorStore::new(db_config, vector_config)
        .await
        .unwrap()
}

/// Test 14: Vector store basic add and search
///
/// Verifies adding a vector and finding it via search.
#[tokio::test]
async fn test_vector_store_add_and_search() {
    let mut store = create_test_vector_store(4).await;

    // Add a vector with metadata
    let embedding = vec![0.1, 0.2, 0.3, 0.4];
    let metadata = Some(HashMap::from([
        ("source".to_string(), "test".to_string()),
        ("type".to_string(), "integration".to_string()),
    ]));

    store
        .add_vector("vec1".to_string(), embedding.clone(), metadata)
        .await
        .unwrap();

    // Search for it - should find exact match
    let results = store.search(&embedding, 1).await.unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, "vec1");
    assert!(results[0].distance < 0.01); // Very small distance for exact match

    // Verify metadata is returned
    let result_metadata = results[0].metadata.as_ref().unwrap();
    assert_eq!(result_metadata.get("source").unwrap(), "test");
    assert_eq!(result_metadata.get("type").unwrap(), "integration");
}

/// Test 15: Vector store similarity search with cosine
///
/// Verifies similarity search returns correct results ordered by similarity.
#[tokio::test]
async fn test_vector_store_similarity_search_cosine() {
    let mut store = create_test_vector_store(4).await;

    // Add vectors
    store
        .add_vector(
            "vec_a".to_string(),
            vec![1.0, 0.0, 0.0, 0.0],
            Some(HashMap::from([("label".to_string(), "A".to_string())])),
        )
        .await
        .unwrap();
    store
        .add_vector(
            "vec_b".to_string(),
            vec![0.9, 0.1, 0.0, 0.0],
            Some(HashMap::from([("label".to_string(), "B".to_string())])),
        )
        .await
        .unwrap();
    store
        .add_vector(
            "vec_c".to_string(),
            vec![0.0, 1.0, 0.0, 0.0],
            Some(HashMap::from([("label".to_string(), "C".to_string())])),
        )
        .await
        .unwrap();
    store
        .add_vector(
            "vec_d".to_string(),
            vec![0.0, 0.0, 1.0, 0.0],
            Some(HashMap::from([("label".to_string(), "D".to_string())])),
        )
        .await
        .unwrap();

    // Search for vector similar to [1.0, 0.0, 0.0, 0.0]
    let query = vec![1.0, 0.0, 0.0, 0.0];
    let results = store.search(&query, 3).await.unwrap();

    assert_eq!(results.len(), 3);

    // First result should be vec_a (exact match, distance ~0)
    assert_eq!(results[0].id, "vec_a");
    assert!(results[0].distance < 0.01);

    // Second result should be vec_b (very similar, small distance)
    assert_eq!(results[1].id, "vec_b");
    assert!(results[1].distance < 0.2);

    // Third result should be vec_c or vec_d (orthogonal, high distance)
    assert!(results[2].distance > 0.5);
}

/// Test 16: Vector store with Euclidean distance
///
/// Verifies that Euclidean distance metric works correctly.
#[tokio::test]
async fn test_vector_store_euclidean_distance() {
    let mut store =
        create_test_vector_store_with_config(3, DistanceMetric::Euclidean, "vector_euclidean")
            .await;

    // Add vectors
    store
        .add_vector("origin".to_string(), vec![0.0, 0.0, 0.0], None)
        .await
        .unwrap();
    store
        .add_vector("near".to_string(), vec![0.1, 0.1, 0.1], None)
        .await
        .unwrap();
    store
        .add_vector("far".to_string(), vec![1.0, 1.0, 1.0], None)
        .await
        .unwrap();

    // Search from origin - nearest should be origin itself, then near, then far
    let query = vec![0.0, 0.0, 0.0];
    let results = store.search(&query, 3).await.unwrap();

    assert_eq!(results.len(), 3);
    assert_eq!(results[0].id, "origin"); // Distance 0
    assert_eq!(results[1].id, "near"); // Distance ~0.173
    assert_eq!(results[2].id, "far"); // Distance ~1.732

    // Verify distances are increasing
    assert!(results[0].distance < results[1].distance);
    assert!(results[1].distance < results[2].distance);
}

/// Test 17: Vector store batch operations
///
/// Verifies adding multiple vectors in batch.
#[tokio::test]
async fn test_vector_store_batch_add() {
    let mut store = create_test_vector_store(4).await;

    // Build batch
    let batch: Vec<(String, Vec<f32>, Option<HashMap<String, String>>)> = (0..10)
        .map(|i| {
            let id = format!("batch_vec_{}", i);
            let embedding = vec![i as f32 * 0.1, 0.5, 0.5, 0.5];
            let metadata = Some(HashMap::from([("index".to_string(), i.to_string())]));
            (id, embedding, metadata)
        })
        .collect();

    // Add batch
    store.add_vectors_batch(batch).await.unwrap();

    // Verify count
    let count = store.len().await;
    assert_eq!(count, 10);

    // Verify vectors are searchable
    let query = vec![0.5, 0.5, 0.5, 0.5];
    let results = store.search(&query, 10).await.unwrap();
    assert_eq!(results.len(), 10);
}

/// Test 18: Vector store update (upsert)
///
/// Verifies that storing a vector with the same ID updates it.
#[tokio::test]
async fn test_vector_store_update() {
    let mut store = create_test_vector_store(4).await;

    // Add initial vector
    store
        .add_vector(
            "update_test".to_string(),
            vec![0.1, 0.2, 0.3, 0.4],
            Some(HashMap::from([("version".to_string(), "1".to_string())])),
        )
        .await
        .unwrap();

    // Verify initial state
    let results = store.search(&[0.1, 0.2, 0.3, 0.4], 1).await.unwrap();
    assert_eq!(results[0].id, "update_test");
    assert_eq!(
        results[0]
            .metadata
            .as_ref()
            .unwrap()
            .get("version")
            .unwrap(),
        "1"
    );

    // Update with new values
    store
        .add_vector(
            "update_test".to_string(),
            vec![0.5, 0.6, 0.7, 0.8],
            Some(HashMap::from([("version".to_string(), "2".to_string())])),
        )
        .await
        .unwrap();

    // Verify update - search for new embedding should find it
    let results = store.search(&[0.5, 0.6, 0.7, 0.8], 1).await.unwrap();
    assert_eq!(results[0].id, "update_test");
    assert_eq!(
        results[0]
            .metadata
            .as_ref()
            .unwrap()
            .get("version")
            .unwrap(),
        "2"
    );

    // Count should still be 1
    assert_eq!(store.len().await, 1);
}

/// Test 19: Vector store remove
///
/// Verifies removing a vector from the store.
#[tokio::test]
async fn test_vector_store_remove() {
    let mut store = create_test_vector_store(4).await;

    // Add vectors
    store
        .add_vector("remove_1".to_string(), vec![0.1, 0.2, 0.3, 0.4], None)
        .await
        .unwrap();
    store
        .add_vector("remove_2".to_string(), vec![0.5, 0.6, 0.7, 0.8], None)
        .await
        .unwrap();

    assert_eq!(store.len().await, 2);

    // Remove one
    let removed = store.remove_vector("remove_1").await.unwrap();
    assert!(removed);

    // Verify removal
    assert_eq!(store.len().await, 1);

    // Search for removed vector's embedding shouldn't find "remove_1"
    let results = store.search(&[0.1, 0.2, 0.3, 0.4], 10).await.unwrap();
    assert!(!results.iter().any(|r| r.id == "remove_1"));

    // Other vector should still exist
    let results = store.search(&[0.5, 0.6, 0.7, 0.8], 1).await.unwrap();
    assert_eq!(results[0].id, "remove_2");

    // Removing non-existent should return false
    let removed = store.remove_vector("nonexistent").await.unwrap();
    assert!(!removed);
}

/// Test 20: Vector store remove batch
///
/// Verifies removing multiple vectors via batch.
#[tokio::test]
async fn test_vector_store_remove_batch() {
    let mut store = create_test_vector_store(4).await;

    // Add vectors
    for i in 0..5 {
        store
            .add_vector(format!("remove_batch_{}", i), vec![0.1 * i as f32; 4], None)
            .await
            .unwrap();
    }

    assert_eq!(store.len().await, 5);

    // Remove some
    let ids_to_remove: Vec<&str> = vec!["remove_batch_0", "remove_batch_2", "remove_batch_4"];
    let results = store.remove_vectors_batch(&ids_to_remove).await.unwrap();

    assert_eq!(results.len(), 3);
    assert!(results.iter().all(|&removed| removed));

    // Verify count
    assert_eq!(store.len().await, 2);
}

/// Test 21: Vector store empty behavior
///
/// Verifies correct behavior when store is empty.
#[tokio::test]
async fn test_vector_store_empty_behavior() {
    let store = create_test_vector_store(4).await;

    // Empty checks
    assert_eq!(store.len().await, 0);

    // Search on empty store
    let results = store.search(&[0.1, 0.2, 0.3, 0.4], 5).await.unwrap();
    assert!(results.is_empty());
}

/// Test 22: Vector store with large embeddings
///
/// Verifies handling of typical embedding sizes (384 dimensions).
#[tokio::test]
async fn test_vector_store_large_embeddings() {
    // Test with 384 dimensions (common for small models)
    let mut store = create_test_vector_store(384).await;

    let embedding: Vec<f32> = (0..384).map(|i| i as f32 / 384.0).collect();
    let metadata = Some(HashMap::from([("model".to_string(), "small".to_string())]));

    store
        .add_vector("large_embed_1".to_string(), embedding.clone(), metadata)
        .await
        .unwrap();

    // Search should find it
    let results = store.search(&embedding, 1).await.unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, "large_embed_1");
    assert!(results[0].distance < 0.01); // Exact match
}

/// Test 23: Vector store search with top_k limits
///
/// Verifies that search respects the top_k parameter.
#[tokio::test]
async fn test_vector_store_search_topk() {
    let mut store = create_test_vector_store(4).await;

    // Add 20 vectors
    for i in 0..20 {
        store
            .add_vector(
                format!("topk_{}", i),
                vec![i as f32 * 0.05, 0.5, 0.5, 0.5],
                None,
            )
            .await
            .unwrap();
    }

    // Search with different top_k values
    let query = vec![0.5, 0.5, 0.5, 0.5];

    let results_5 = store.search(&query, 5).await.unwrap();
    assert_eq!(results_5.len(), 5);

    let results_10 = store.search(&query, 10).await.unwrap();
    assert_eq!(results_10.len(), 10);

    let results_all = store.search(&query, 100).await.unwrap();
    assert_eq!(results_all.len(), 20); // Can't return more than exist
}

/// Test 24: Vector store metadata in search results
///
/// Verifies that metadata is correctly returned in search results.
#[tokio::test]
async fn test_vector_store_metadata_in_search() {
    let mut store = create_test_vector_store(4).await;

    // Add vectors with different metadata
    store
        .add_vector(
            "meta_a".to_string(),
            vec![1.0, 0.0, 0.0, 0.0],
            Some(HashMap::from([
                ("category".to_string(), "science".to_string()),
                ("priority".to_string(), "high".to_string()),
            ])),
        )
        .await
        .unwrap();
    store
        .add_vector(
            "meta_b".to_string(),
            vec![0.9, 0.1, 0.0, 0.0],
            Some(HashMap::from([
                ("category".to_string(), "tech".to_string()),
                ("priority".to_string(), "low".to_string()),
            ])),
        )
        .await
        .unwrap();

    // Search and verify metadata is in results
    let query = vec![1.0, 0.0, 0.0, 0.0];
    let results = store.search(&query, 2).await.unwrap();

    // First result should be meta_a with its metadata
    assert_eq!(results[0].id, "meta_a");
    let metadata = results[0].metadata.as_ref().unwrap();
    assert_eq!(metadata.get("category").unwrap(), "science");
    assert_eq!(metadata.get("priority").unwrap(), "high");
}

/// Test 25: Vector store with different distance metrics
///
/// Verifies that different distance metrics produce valid rankings.
#[tokio::test]
async fn test_vector_store_distance_metrics() {
    // Setup vectors
    let v1 = vec![1.0, 0.0, 0.0];
    let v2 = vec![0.5, 0.5, 0.0];
    let v3 = vec![0.33, 0.33, 0.33];

    // Cosine similarity store
    let mut cosine_store =
        create_test_vector_store_with_config(3, DistanceMetric::Cosine, "metric_cosine").await;
    cosine_store
        .add_vector("v1".to_string(), v1.clone(), None)
        .await
        .unwrap();
    cosine_store
        .add_vector("v2".to_string(), v2.clone(), None)
        .await
        .unwrap();
    cosine_store
        .add_vector("v3".to_string(), v3.clone(), None)
        .await
        .unwrap();

    // Manhattan distance store
    let mut manhattan_store =
        create_test_vector_store_with_config(3, DistanceMetric::Manhattan, "metric_manhattan")
            .await;
    manhattan_store
        .add_vector("v1".to_string(), v1.clone(), None)
        .await
        .unwrap();
    manhattan_store
        .add_vector("v2".to_string(), v2.clone(), None)
        .await
        .unwrap();
    manhattan_store
        .add_vector("v3".to_string(), v3.clone(), None)
        .await
        .unwrap();

    // Query with [1, 0, 0]
    let query = vec![1.0, 0.0, 0.0];

    let cosine_results = cosine_store.search(&query, 3).await.unwrap();
    let manhattan_results = manhattan_store.search(&query, 3).await.unwrap();

    // Both should return v1 first (exact match)
    assert_eq!(cosine_results[0].id, "v1");
    assert_eq!(manhattan_results[0].id, "v1");

    // Both should return all 3 results
    assert_eq!(cosine_results.len(), 3);
    assert_eq!(manhattan_results.len(), 3);
}

/// Test 26: Vector store isolated by table name
///
/// Verifies that different table names create isolated stores.
#[tokio::test]
async fn test_vector_store_table_isolation() {
    let db_config = SurrealDbConfig::memory();

    // Create two stores with different table names
    let config1 = SurrealDbVectorConfig::with_dimension(4).with_table_name("vectors_1".to_string());
    let config2 = SurrealDbVectorConfig::with_dimension(4).with_table_name("vectors_2".to_string());

    let mut store1 = SurrealDbVectorStore::new(db_config.clone(), config1)
        .await
        .unwrap();
    let mut store2 = SurrealDbVectorStore::new(db_config, config2).await.unwrap();

    // Add to store1
    store1
        .add_vector("shared_id".to_string(), vec![1.0, 0.0, 0.0, 0.0], None)
        .await
        .unwrap();

    // Add different vector with same ID to store2
    store2
        .add_vector("shared_id".to_string(), vec![0.0, 1.0, 0.0, 0.0], None)
        .await
        .unwrap();

    // Verify isolation via search
    let results1 = store1.search(&[1.0, 0.0, 0.0, 0.0], 1).await.unwrap();
    let results2 = store2.search(&[0.0, 1.0, 0.0, 0.0], 1).await.unwrap();

    assert_eq!(results1[0].id, "shared_id");
    assert!(results1[0].distance < 0.01); // Exact match for store1's embedding

    assert_eq!(results2[0].id, "shared_id");
    assert!(results2[0].distance < 0.01); // Exact match for store2's embedding

    // Each store should have count of 1
    assert_eq!(store1.len().await, 1);
    assert_eq!(store2.len().await, 1);
}

/// Test 27: Vector store search with threshold
///
/// Verifies search_with_threshold filters results correctly.
#[tokio::test]
async fn test_vector_store_search_with_threshold() {
    let mut store = create_test_vector_store(4).await;

    // Add vectors at varying distances
    store
        .add_vector("exact".to_string(), vec![1.0, 0.0, 0.0, 0.0], None)
        .await
        .unwrap();
    store
        .add_vector("close".to_string(), vec![0.9, 0.1, 0.0, 0.0], None)
        .await
        .unwrap();
    store
        .add_vector("far".to_string(), vec![0.0, 1.0, 0.0, 0.0], None)
        .await
        .unwrap();

    let query = vec![1.0, 0.0, 0.0, 0.0];

    // Search with tight threshold - should only get exact and close
    let results = store.search_with_threshold(&query, 10, 0.2).await.unwrap();
    assert!(results.len() >= 1 && results.len() <= 2);
    assert!(results.iter().all(|r| r.distance <= 0.2));

    // Search with loose threshold - should get all
    let results = store.search_with_threshold(&query, 10, 2.0).await.unwrap();
    assert_eq!(results.len(), 3);
}
