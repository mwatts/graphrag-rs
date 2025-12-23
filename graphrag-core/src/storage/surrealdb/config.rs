//! SurrealDB configuration types
//!
//! This module provides configuration structures for connecting to SurrealDB
//! with support for multiple backends: in-memory, RocksDB, and remote connections.

use serde::{Deserialize, Serialize};

/// SurrealDB connection configuration
///
/// Supports multiple connection modes:
/// - In-memory (`mem://`) for development and testing
/// - RocksDB (`rocksdb://path`) for persistent local storage
/// - WebSocket (`ws://host:port`) for remote connections
/// - HTTP (`http://host:port`) for stateless remote connections
///
/// # Examples
///
/// ```rust,ignore
/// use graphrag_core::storage::surrealdb::SurrealDbConfig;
///
/// // In-memory for development
/// let config = SurrealDbConfig::memory();
///
/// // Persistent local storage
/// let config = SurrealDbConfig::rocksdb("./data/graphrag");
///
/// // Remote WebSocket connection
/// let config = SurrealDbConfig::websocket("localhost", 8000);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurrealDbConfig {
    /// Connection endpoint
    ///
    /// Format depends on the backend:
    /// - `mem://` for in-memory (development)
    /// - `rocksdb://path/to/db` for persistent local
    /// - `ws://host:port` for remote WebSocket
    /// - `http://host:port` for remote HTTP
    pub endpoint: String,

    /// Namespace for data isolation
    ///
    /// SurrealDB uses namespaces to separate different applications or tenants.
    pub namespace: String,

    /// Database name within namespace
    ///
    /// Multiple databases can exist within a single namespace.
    pub database: String,

    /// Optional authentication credentials
    ///
    /// Required for remote connections with authentication enabled.
    pub credentials: Option<SurrealDbCredentials>,

    /// Whether to initialize schema on connect
    ///
    /// When `true`, the storage will create required tables and indexes
    /// on first connection. Set to `false` if schema is managed externally.
    pub auto_init_schema: bool,
}

/// Authentication credentials for SurrealDB
///
/// Used for root-level authentication to remote SurrealDB instances.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurrealDbCredentials {
    /// Username for authentication
    pub username: String,

    /// Password for authentication
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
    ///
    /// Data is not persisted and will be lost when the process exits.
    /// Ideal for unit tests and development.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let config = SurrealDbConfig::memory();
    /// ```
    pub fn memory() -> Self {
        Self::default()
    }

    /// Create configuration for persistent local storage using RocksDB
    ///
    /// Data is persisted to disk at the specified path.
    /// Suitable for single-node production deployments.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the database directory
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let config = SurrealDbConfig::rocksdb("./data/graphrag");
    /// ```
    pub fn rocksdb(path: impl Into<String>) -> Self {
        Self {
            endpoint: format!("rocksdb://{}", path.into()),
            ..Self::default()
        }
    }

    /// Create configuration for remote WebSocket connection
    ///
    /// Connects to a remote SurrealDB server via WebSocket.
    /// Supports bidirectional communication and live queries.
    ///
    /// # Arguments
    ///
    /// * `host` - Hostname or IP address of the server
    /// * `port` - Port number (default SurrealDB port is 8000)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let config = SurrealDbConfig::websocket("localhost", 8000);
    /// ```
    pub fn websocket(host: impl Into<String>, port: u16) -> Self {
        Self {
            endpoint: format!("ws://{}:{}", host.into(), port),
            ..Self::default()
        }
    }

    /// Create configuration for remote HTTP connection
    ///
    /// Connects to a remote SurrealDB server via HTTP.
    /// Stateless connection suitable for simpler deployments.
    ///
    /// # Arguments
    ///
    /// * `host` - Hostname or IP address of the server
    /// * `port` - Port number (default SurrealDB port is 8000)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let config = SurrealDbConfig::http("localhost", 8000);
    /// ```
    pub fn http(host: impl Into<String>, port: u16) -> Self {
        Self {
            endpoint: format!("http://{}:{}", host.into(), port),
            ..Self::default()
        }
    }

    /// Set the namespace for this configuration
    ///
    /// # Arguments
    ///
    /// * `namespace` - The namespace name
    pub fn with_namespace(mut self, namespace: impl Into<String>) -> Self {
        self.namespace = namespace.into();
        self
    }

    /// Set the database for this configuration
    ///
    /// # Arguments
    ///
    /// * `database` - The database name
    pub fn with_database(mut self, database: impl Into<String>) -> Self {
        self.database = database.into();
        self
    }

    /// Set authentication credentials
    ///
    /// # Arguments
    ///
    /// * `username` - Username for authentication
    /// * `password` - Password for authentication
    pub fn with_credentials(
        mut self,
        username: impl Into<String>,
        password: impl Into<String>,
    ) -> Self {
        self.credentials = Some(SurrealDbCredentials {
            username: username.into(),
            password: password.into(),
        });
        self
    }

    /// Disable automatic schema initialization
    ///
    /// Use this when the schema is managed externally or already exists.
    pub fn without_auto_schema(mut self) -> Self {
        self.auto_init_schema = false;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = SurrealDbConfig::default();
        assert_eq!(config.endpoint, "mem://");
        assert_eq!(config.namespace, "graphrag");
        assert_eq!(config.database, "default");
        assert!(config.credentials.is_none());
        assert!(config.auto_init_schema);
    }

    #[test]
    fn test_memory_config() {
        let config = SurrealDbConfig::memory();
        assert_eq!(config.endpoint, "mem://");
    }

    #[test]
    fn test_rocksdb_config() {
        let config = SurrealDbConfig::rocksdb("./data/test");
        assert_eq!(config.endpoint, "rocksdb://./data/test");
    }

    #[test]
    fn test_websocket_config() {
        let config = SurrealDbConfig::websocket("localhost", 8000);
        assert_eq!(config.endpoint, "ws://localhost:8000");
    }

    #[test]
    fn test_http_config() {
        let config = SurrealDbConfig::http("example.com", 9000);
        assert_eq!(config.endpoint, "http://example.com:9000");
    }

    #[test]
    fn test_builder_methods() {
        let config = SurrealDbConfig::memory()
            .with_namespace("myapp")
            .with_database("production")
            .with_credentials("admin", "secret")
            .without_auto_schema();

        assert_eq!(config.namespace, "myapp");
        assert_eq!(config.database, "production");
        assert!(config.credentials.is_some());
        let creds = config.credentials.unwrap();
        assert_eq!(creds.username, "admin");
        assert_eq!(creds.password, "secret");
        assert!(!config.auto_init_schema);
    }
}
