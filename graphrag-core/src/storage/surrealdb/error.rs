//! SurrealDB-specific error types
//!
//! This module provides error types for SurrealDB storage operations,
//! with conversions to the core GraphRAGError type.

use thiserror::Error;

/// Errors specific to SurrealDB storage operations
///
/// These errors provide detailed information about what went wrong
/// during SurrealDB operations, while still being convertible to
/// the generic `GraphRAGError` for trait implementations.
#[derive(Debug, Error)]
pub enum SurrealDbStorageError {
    /// Failed to establish connection to SurrealDB
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),

    /// Authentication with SurrealDB failed
    #[error("Authentication failed: {0}")]
    AuthenticationFailed(String),

    /// Failed to initialize database schema
    #[error("Schema initialization failed: {0}")]
    SchemaInitFailed(String),

    /// Query execution failed
    #[error("Query execution failed: {0}")]
    QueryFailed(String),

    /// Serialization or deserialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Record was not found
    #[error("Record not found: {table}:{id}")]
    NotFound {
        /// The table that was queried
        table: String,
        /// The ID that was not found
        id: String,
    },

    /// Attempted to create a duplicate record
    #[error("Duplicate record: {table}:{id}")]
    DuplicateRecord {
        /// The table where the duplicate was attempted
        table: String,
        /// The duplicate ID
        id: String,
    },

    /// Transaction failed
    #[error("Transaction failed: {0}")]
    TransactionFailed(String),

    /// Namespace or database selection failed
    #[error("Failed to select namespace/database: {0}")]
    SelectionFailed(String),
}

impl From<surrealdb::Error> for SurrealDbStorageError {
    fn from(err: surrealdb::Error) -> Self {
        // Map SurrealDB errors to our error types
        let msg = err.to_string();

        // Try to categorize based on error message patterns
        if msg.contains("authentication") || msg.contains("credentials") {
            Self::AuthenticationFailed(msg)
        } else if msg.contains("connection") || msg.contains("connect") {
            Self::ConnectionFailed(msg)
        } else if msg.contains("not found") || msg.contains("does not exist") {
            Self::QueryFailed(msg)
        } else if msg.contains("duplicate") || msg.contains("already exists") {
            Self::QueryFailed(msg)
        } else {
            Self::QueryFailed(msg)
        }
    }
}

impl From<SurrealDbStorageError> for crate::core::GraphRAGError {
    fn from(err: SurrealDbStorageError) -> Self {
        crate::core::GraphRAGError::Storage {
            message: err.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = SurrealDbStorageError::ConnectionFailed("timeout".to_string());
        assert_eq!(err.to_string(), "Connection failed: timeout");

        let err = SurrealDbStorageError::NotFound {
            table: "entity".to_string(),
            id: "test123".to_string(),
        };
        assert_eq!(err.to_string(), "Record not found: entity:test123");

        let err = SurrealDbStorageError::DuplicateRecord {
            table: "document".to_string(),
            id: "doc1".to_string(),
        };
        assert_eq!(err.to_string(), "Duplicate record: document:doc1");
    }

    #[test]
    fn test_error_conversion_to_graphrag_error() {
        let err = SurrealDbStorageError::QueryFailed("test error".to_string());
        let graphrag_err: crate::core::GraphRAGError = err.into();

        match graphrag_err {
            crate::core::GraphRAGError::Storage { message } => {
                assert!(message.contains("test error"));
            },
            _ => panic!("Expected Storage error variant"),
        }
    }
}
