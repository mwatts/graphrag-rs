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

use graphrag_core::core::{
    traits::AsyncStorage, ChunkId, ChunkMetadata, Document, DocumentId, Entity, EntityId, TextChunk,
};
use graphrag_core::storage::surrealdb::{SurrealDbConfig, SurrealDbStorage};
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
