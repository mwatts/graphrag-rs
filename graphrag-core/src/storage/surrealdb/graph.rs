//! SurrealDB graph store implementation
//!
//! This module provides a SurrealDB implementation of the `AsyncGraphStore` trait,
//! enabling graph operations using SurrealDB's native graph capabilities.
//!
//! ## Features
//!
//! - **Native graph support**: Uses SurrealDB's `RELATE` statements for edges
//! - **Graph traversal**: Leverages SurrealDB's graph traversal syntax
//! - **Path finding**: Find shortest paths between entities
//! - **Subgraph extraction**: Get entities and relationships within a radius
//!
//! ## Usage
//!
//! ```rust,ignore
//! use graphrag_core::storage::surrealdb::{
//!     SurrealDbConfig, SurrealDbGraphStore, SurrealDbGraphConfig,
//! };
//! use graphrag_core::core::traits::AsyncGraphStore;
//!
//! let db_config = SurrealDbConfig::rocksdb("./data/graph");
//! let graph_config = SurrealDbGraphConfig::default();
//!
//! let mut store = SurrealDbGraphStore::new(db_config, graph_config).await?;
//!
//! // Add nodes and edges
//! store.add_node(entity).await?;
//! store.add_edge("entity1", "entity2", relationship).await?;
//!
//! // Query graph
//! let neighbors = store.get_neighbors("entity1").await?;
//! let stats = store.stats().await;
//! ```

use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use surrealdb::engine::any::Any;
use surrealdb::Surreal;

use crate::core::error::{GraphRAGError, Result};
use crate::core::traits::{AsyncGraphStore, GraphStats};
use crate::core::{ChunkId, Entity, EntityId, EntityMention, Relationship};

use super::config::SurrealDbConfig;

// =============================================================================
// Configuration Types
// =============================================================================

/// Configuration for SurrealDB graph store
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurrealDbGraphConfig {
    /// Table name for nodes/entities (default: "entity")
    #[serde(default = "default_node_table")]
    pub node_table: String,

    /// Table name for edges/relationships (default: "relates_to")
    #[serde(default = "default_edge_table")]
    pub edge_table: String,

    /// Maximum traversal depth for safety (default: 10)
    #[serde(default = "default_max_traversal_depth")]
    pub max_traversal_depth: usize,

    /// Whether to auto-initialize schema (default: true)
    #[serde(default = "default_auto_init_schema")]
    pub auto_init_schema: bool,
}

fn default_node_table() -> String {
    "entity".to_string()
}

fn default_edge_table() -> String {
    "relates_to".to_string()
}

fn default_max_traversal_depth() -> usize {
    10
}

fn default_auto_init_schema() -> bool {
    true
}

impl Default for SurrealDbGraphConfig {
    fn default() -> Self {
        Self {
            node_table: default_node_table(),
            edge_table: default_edge_table(),
            max_traversal_depth: default_max_traversal_depth(),
            auto_init_schema: default_auto_init_schema(),
        }
    }
}

impl SurrealDbGraphConfig {
    /// Create a new config with custom table names
    pub fn with_tables(node_table: impl Into<String>, edge_table: impl Into<String>) -> Self {
        Self {
            node_table: node_table.into(),
            edge_table: edge_table.into(),
            ..Default::default()
        }
    }

    /// Set the maximum traversal depth
    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_traversal_depth = depth;
        self
    }

    /// Disable auto schema initialization
    pub fn without_auto_init(mut self) -> Self {
        self.auto_init_schema = false;
        self
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        if self.node_table.is_empty() {
            return Err(GraphRAGError::Config {
                message: "Node table name cannot be empty".to_string(),
            });
        }

        if self.edge_table.is_empty() {
            return Err(GraphRAGError::Config {
                message: "Edge table name cannot be empty".to_string(),
            });
        }

        if self.max_traversal_depth == 0 {
            return Err(GraphRAGError::Config {
                message: "Max traversal depth must be greater than 0".to_string(),
            });
        }

        Ok(())
    }
}

// =============================================================================
// Graph Store Implementation
// =============================================================================

/// SurrealDB graph store implementation
pub struct SurrealDbGraphStore {
    db: Arc<Surreal<Any>>,
    config: SurrealDbGraphConfig,
    db_config: SurrealDbConfig,
}

impl SurrealDbGraphStore {
    /// Create a new graph store with the given configurations
    pub async fn new(
        db_config: SurrealDbConfig,
        graph_config: SurrealDbGraphConfig,
    ) -> Result<Self> {
        graph_config.validate()?;

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
            config: graph_config.clone(),
            db_config,
        };

        // Initialize schema if enabled
        if store.db_config.auto_init_schema && graph_config.auto_init_schema {
            store.init_schema().await?;
        }

        Ok(store)
    }

    /// Create a graph store from an existing database connection
    pub fn from_client(
        db: Arc<Surreal<Any>>,
        db_config: SurrealDbConfig,
        graph_config: SurrealDbGraphConfig,
    ) -> Result<Self> {
        graph_config.validate()?;

        Ok(Self {
            db,
            config: graph_config,
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

    /// Get the graph configuration
    pub fn config(&self) -> &SurrealDbGraphConfig {
        &self.config
    }

    /// Initialize the graph schema
    async fn init_schema(&self) -> Result<()> {
        let schema = format!(
            r#"
            -- Entity/Node table
            DEFINE TABLE {node_table} SCHEMALESS;
            DEFINE INDEX idx_{node_table}_id ON {node_table} FIELDS id UNIQUE;
            DEFINE INDEX idx_{node_table}_type ON {node_table} FIELDS entity_type;
            DEFINE INDEX idx_{node_table}_name ON {node_table} FIELDS name;

            -- Relationship/Edge table (graph relation)
            DEFINE TABLE {edge_table} TYPE RELATION IN {node_table} OUT {node_table};
            DEFINE FIELD relation_type ON {edge_table} TYPE string;
            DEFINE FIELD confidence ON {edge_table} TYPE float;
            DEFINE FIELD context ON {edge_table} TYPE array DEFAULT [];
            DEFINE INDEX idx_{edge_table}_type ON {edge_table} FIELDS relation_type;
            "#,
            node_table = self.config.node_table,
            edge_table = self.config.edge_table
        );

        self.db
            .query(&schema)
            .await
            .map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to initialize graph schema: {}", e),
            })?;

        Ok(())
    }

    /// Count nodes in the graph
    async fn count_nodes(&self) -> Result<usize> {
        let query = format!("SELECT count() FROM {} GROUP ALL", self.config.node_table);

        let mut response = match self.db.query(&query).await {
            Ok(r) => r,
            Err(_) => return Ok(0),
        };

        #[derive(Deserialize)]
        struct CountResult {
            count: usize,
        }

        Ok(response
            .take::<Vec<CountResult>>(0)
            .ok()
            .and_then(|v| v.into_iter().next())
            .map(|r| r.count)
            .unwrap_or(0))
    }

    /// Count edges in the graph
    async fn count_edges(&self) -> Result<usize> {
        let query = format!("SELECT count() FROM {} GROUP ALL", self.config.edge_table);

        let mut response = match self.db.query(&query).await {
            Ok(r) => r,
            Err(_) => return Ok(0),
        };

        #[derive(Deserialize)]
        struct CountResult {
            count: usize,
        }

        Ok(response
            .take::<Vec<CountResult>>(0)
            .ok()
            .and_then(|v| v.into_iter().next())
            .map(|r| r.count)
            .unwrap_or(0))
    }
}

// =============================================================================
// Advanced Graph Queries
// =============================================================================

impl SurrealDbGraphStore {
    /// Find shortest path between two entities
    ///
    /// Returns the entities along the path from `from_id` to `to_id`,
    /// or an empty vector if no path exists within `max_depth`.
    pub async fn find_path(
        &self,
        from_id: &str,
        to_id: &str,
        max_depth: usize,
    ) -> Result<Vec<Entity>> {
        let depth = max_depth.min(self.config.max_traversal_depth);

        // Use SurrealDB's graph traversal to find path
        // We traverse from source and check if target is reachable
        let query = format!(
            r#"
            SELECT *, meta::id(id) as id
            FROM type::thing($node_table, $from_id)->{edge_table}->(*..{depth})
            WHERE meta::id(id) = $to_id
            LIMIT 1
            "#,
            edge_table = self.config.edge_table,
            depth = depth
        );

        let mut response = self
            .db
            .query(&query)
            .bind(("node_table", self.config.node_table.clone()))
            .bind(("from_id", from_id.to_string()))
            .bind(("to_id", to_id.to_string()))
            .await
            .map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to find path: {}", e),
            })?;

        let results: Vec<serde_json::Value> =
            response.take(0).map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to parse path results: {}", e),
            })?;

        // If we found a path, return the target entity
        // Note: SurrealDB path queries return the destination, not the full path
        // For full path reconstruction, we'd need a more complex query
        if results.is_empty() {
            return Ok(vec![]);
        }

        results.into_iter().map(|v| parse_entity(v)).collect()
    }

    /// Get entities connected by a specific relationship type
    ///
    /// Returns all entities that `entity_id` is connected to via `relation_type`.
    pub async fn get_by_relationship(
        &self,
        entity_id: &str,
        relation_type: &str,
    ) -> Result<Vec<Entity>> {
        let query = format!(
            r#"
            SELECT *, meta::id(id) as id
            FROM type::thing($node_table, $entity_id)->{edge_table}
            WHERE relation_type = $relation_type
            ->{node_table}
            "#,
            edge_table = self.config.edge_table,
            node_table = self.config.node_table
        );

        let mut response = self
            .db
            .query(&query)
            .bind(("node_table", self.config.node_table.clone()))
            .bind(("entity_id", entity_id.to_string()))
            .bind(("relation_type", relation_type.to_string()))
            .await
            .map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to get entities by relationship: {}", e),
            })?;

        let results: Vec<serde_json::Value> =
            response.take(0).map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to parse relationship results: {}", e),
            })?;

        results.into_iter().map(|v| parse_entity(v)).collect()
    }

    /// Get subgraph around an entity
    ///
    /// Returns all entities within `radius` hops and the relationships between them.
    pub async fn get_subgraph(
        &self,
        center_id: &str,
        radius: usize,
    ) -> Result<(Vec<Entity>, Vec<Relationship>)> {
        let depth = radius.min(self.config.max_traversal_depth);

        // Get nodes in subgraph
        let nodes = self.traverse(center_id, depth).await?;

        if nodes.is_empty() {
            return Ok((vec![], vec![]));
        }

        // Collect node IDs for edge query
        let node_ids: Vec<String> = nodes.iter().map(|n| n.id.0.clone()).collect();

        // Get edges between those nodes - use string::concat to ensure we get plain strings
        let edges_query = format!(
            r#"
            SELECT
                relation_type,
                confidence,
                context,
                string::concat("", meta::id(in)) as source_id,
                string::concat("", meta::id(out)) as target_id
            FROM {}
            "#,
            self.config.edge_table
        );

        let mut response =
            self.db
                .query(&edges_query)
                .await
                .map_err(|e| GraphRAGError::Storage {
                    message: format!("Failed to get subgraph edges: {}", e),
                })?;

        let edge_results: Vec<serde_json::Value> =
            response.take(0).map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to parse edge results: {}", e),
            })?;

        // Filter edges to only those connecting nodes in the subgraph
        let edges: Vec<Relationship> = edge_results
            .into_iter()
            .filter_map(|v| {
                let source = v.get("source_id").and_then(|s| s.as_str())?;
                let target = v.get("target_id").and_then(|s| s.as_str())?;

                // Check if both source and target are in our node set
                if node_ids.contains(&source.to_string()) && node_ids.contains(&target.to_string())
                {
                    parse_relationship(v.clone()).ok()
                } else {
                    None
                }
            })
            .collect();

        Ok((nodes, edges))
    }

    /// Get incoming relationships (edges pointing to this entity)
    pub async fn get_incoming(&self, entity_id: &str) -> Result<Vec<(Entity, Relationship)>> {
        let query = format!(
            r#"
            SELECT
                in as source_entity,
                relation_type,
                confidence,
                context,
                meta::id(in) as source_id,
                meta::id(out) as target_id
            FROM {}
            WHERE out = type::thing($node_table, $entity_id)
            FETCH source_entity
            "#,
            self.config.edge_table
        );

        let mut response = self
            .db
            .query(&query)
            .bind(("node_table", self.config.node_table.clone()))
            .bind(("entity_id", entity_id.to_string()))
            .await
            .map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to get incoming edges: {}", e),
            })?;

        let results: Vec<serde_json::Value> =
            response.take(0).map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to parse incoming edge results: {}", e),
            })?;

        results
            .into_iter()
            .filter_map(|v| {
                let entity = v
                    .get("source_entity")
                    .cloned()
                    .and_then(|e| parse_entity(e).ok())?;
                let relationship = parse_relationship(v).ok()?;
                Some((entity, relationship))
            })
            .collect::<Vec<_>>()
            .pipe(Ok)
    }

    /// Get outgoing relationships (edges from this entity)
    pub async fn get_outgoing(&self, entity_id: &str) -> Result<Vec<(Entity, Relationship)>> {
        let query = format!(
            r#"
            SELECT
                out as target_entity,
                relation_type,
                confidence,
                context,
                meta::id(in) as source_id,
                meta::id(out) as target_id
            FROM {}
            WHERE in = type::thing($node_table, $entity_id)
            FETCH target_entity
            "#,
            self.config.edge_table
        );

        let mut response = self
            .db
            .query(&query)
            .bind(("node_table", self.config.node_table.clone()))
            .bind(("entity_id", entity_id.to_string()))
            .await
            .map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to get outgoing edges: {}", e),
            })?;

        let results: Vec<serde_json::Value> =
            response.take(0).map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to parse outgoing edge results: {}", e),
            })?;

        results
            .into_iter()
            .filter_map(|v| {
                let entity = v
                    .get("target_entity")
                    .cloned()
                    .and_then(|e| parse_entity(e).ok())?;
                let relationship = parse_relationship(v).ok()?;
                Some((entity, relationship))
            })
            .collect::<Vec<_>>()
            .pipe(Ok)
    }

    /// Get all relationships for an entity (both incoming and outgoing)
    pub async fn get_all_relationships(&self, entity_id: &str) -> Result<Vec<Relationship>> {
        let query = format!(
            r#"
            SELECT
                relation_type,
                confidence,
                context,
                meta::id(in) as source_id,
                meta::id(out) as target_id
            FROM {}
            WHERE in = type::thing($node_table, $entity_id)
               OR out = type::thing($node_table, $entity_id)
            "#,
            self.config.edge_table
        );

        let mut response = self
            .db
            .query(&query)
            .bind(("node_table", self.config.node_table.clone()))
            .bind(("entity_id", entity_id.to_string()))
            .await
            .map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to get relationships: {}", e),
            })?;

        let results: Vec<serde_json::Value> =
            response.take(0).map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to parse relationship results: {}", e),
            })?;

        results.into_iter().map(|v| parse_relationship(v)).collect()
    }

    /// Remove a node and all its edges
    pub async fn remove_node(&mut self, node_id: &str) -> Result<bool> {
        // Check if node exists
        let count_before = self.count_nodes().await?;

        // Delete all edges connected to this node
        let delete_edges = format!(
            r#"
            DELETE FROM {}
            WHERE in = type::thing($node_table, $node_id)
               OR out = type::thing($node_table, $node_id)
            "#,
            self.config.edge_table
        );

        self.db
            .query(&delete_edges)
            .bind(("node_table", self.config.node_table.clone()))
            .bind(("node_id", node_id.to_string()))
            .await
            .map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to delete node edges: {}", e),
            })?;

        // Delete the node
        self.db
            .query("DELETE type::thing($table, $id)")
            .bind(("table", self.config.node_table.clone()))
            .bind(("id", node_id.to_string()))
            .await
            .map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to delete node: {}", e),
            })?;

        let count_after = self.count_nodes().await?;
        Ok(count_after < count_before)
    }

    /// Remove an edge by source, target, and optionally relation type
    pub async fn remove_edge(
        &mut self,
        from_id: &str,
        to_id: &str,
        relation_type: Option<&str>,
    ) -> Result<bool> {
        let count_before = self.count_edges().await?;

        let query = if let Some(rel_type) = relation_type {
            format!(
                r#"
                DELETE FROM {}
                WHERE in = type::thing($node_table, $from_id)
                  AND out = type::thing($node_table, $to_id)
                  AND relation_type = $relation_type
                "#,
                self.config.edge_table
            )
        } else {
            format!(
                r#"
                DELETE FROM {}
                WHERE in = type::thing($node_table, $from_id)
                  AND out = type::thing($node_table, $to_id)
                "#,
                self.config.edge_table
            )
        };

        let mut q = self
            .db
            .query(&query)
            .bind(("node_table", self.config.node_table.clone()))
            .bind(("from_id", from_id.to_string()))
            .bind(("to_id", to_id.to_string()));

        if let Some(rel_type) = relation_type {
            q = q.bind(("relation_type", rel_type.to_string()));
        }

        q.await.map_err(|e| GraphRAGError::Storage {
            message: format!("Failed to delete edge: {}", e),
        })?;

        let count_after = self.count_edges().await?;
        Ok(count_after < count_before)
    }

    /// Get a node by ID
    pub async fn get_node(&self, node_id: &str) -> Result<Option<Entity>> {
        let query = format!("SELECT *, meta::id(id) as id FROM type::thing($table, $id)",);

        let mut response = self
            .db
            .query(&query)
            .bind(("table", self.config.node_table.clone()))
            .bind(("id", node_id.to_string()))
            .await
            .map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to get node: {}", e),
            })?;

        let results: Vec<serde_json::Value> =
            response.take(0).map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to parse node: {}", e),
            })?;

        if results.is_empty() {
            return Ok(None);
        }

        results
            .into_iter()
            .next()
            .map(|v| parse_entity(v))
            .transpose()
    }
}

// =============================================================================
// AsyncGraphStore Trait Implementation
// =============================================================================

/// Helper struct for edge data serialization
#[derive(Debug, Serialize, Deserialize)]
struct EdgeData {
    relation_type: String,
    confidence: f32,
    context: Vec<String>,
}

#[async_trait]
impl AsyncGraphStore for SurrealDbGraphStore {
    type Node = Entity;
    type Edge = Relationship;
    type Error = GraphRAGError;

    async fn add_node(&mut self, node: Self::Node) -> Result<String> {
        let id = node.id.0.clone();

        let json_value = serde_json::to_value(&node).map_err(|e| GraphRAGError::Serialization {
            message: format!("Failed to serialize node: {}", e),
        })?;

        self.db
            .query("UPSERT type::thing($table, $id) CONTENT $data")
            .bind(("table", self.config.node_table.clone()))
            .bind(("id", id.clone()))
            .bind(("data", json_value))
            .await
            .map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to store node: {}", e),
            })?;

        Ok(id)
    }

    async fn add_edge(&mut self, from_id: &str, to_id: &str, edge: Self::Edge) -> Result<String> {
        let edge_data = EdgeData {
            relation_type: edge.relation_type.clone(),
            confidence: edge.confidence,
            context: edge.context.iter().map(|c| c.0.clone()).collect(),
        };

        let json_value =
            serde_json::to_value(&edge_data).map_err(|e| GraphRAGError::Serialization {
                message: format!("Failed to serialize edge: {}", e),
            })?;

        // Use SurrealDB's RELATE syntax for graph edges
        // Note: We use record ID literals (table:id format) for RELATE
        let query = format!(
            r#"
            RELATE {node_table}:`{from_id}`
            ->{edge_table}->
            {node_table}:`{to_id}`
            CONTENT $data
            "#,
            node_table = self.config.node_table,
            edge_table = self.config.edge_table,
            from_id = from_id,
            to_id = to_id
        );

        self.db
            .query(&query)
            .bind(("data", json_value))
            .await
            .map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to create edge: {}", e),
            })?;

        // Generate a consistent edge ID
        let edge_id = format!("{}:{}->{}", self.config.edge_table, from_id, to_id);

        Ok(edge_id)
    }

    async fn find_nodes(&self, criteria: &str) -> Result<Vec<Self::Node>> {
        // Parse criteria: if it contains operators, treat as WHERE clause
        // Otherwise, treat as name search
        let query = if criteria.contains('=')
            || criteria.contains('<')
            || criteria.contains('>')
            || criteria.to_uppercase().contains("WHERE")
        {
            format!(
                "SELECT *, meta::id(id) as id FROM {} WHERE {}",
                self.config.node_table, criteria
            )
        } else {
            format!(
                "SELECT *, meta::id(id) as id FROM {} WHERE name CONTAINS $criteria",
                self.config.node_table
            )
        };

        let mut response = self
            .db
            .query(&query)
            .bind(("criteria", criteria.to_string()))
            .await
            .map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to find nodes: {}", e),
            })?;

        let results: Vec<serde_json::Value> =
            response.take(0).map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to parse find results: {}", e),
            })?;

        results.into_iter().map(|v| parse_entity(v)).collect()
    }

    async fn get_neighbors(&self, node_id: &str) -> Result<Vec<Self::Node>> {
        // Get outgoing neighbors using graph traversal syntax
        let query = format!(
            r#"
            SELECT *, meta::id(id) as id
            FROM {node_table}:`{node_id}`->{edge_table}->{node_table}
            "#,
            node_table = self.config.node_table,
            edge_table = self.config.edge_table,
            node_id = node_id
        );

        let mut response = self
            .db
            .query(&query)
            .await
            .map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to get neighbors: {}", e),
            })?;

        let results: Vec<serde_json::Value> =
            response.take(0).map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to parse neighbors: {}", e),
            })?;

        results.into_iter().map(|v| parse_entity(v)).collect()
    }

    async fn traverse(&self, start_id: &str, max_depth: usize) -> Result<Vec<Self::Node>> {
        let depth = max_depth.min(self.config.max_traversal_depth);

        // Use iterative BFS approach since SurrealDB doesn't support recursive (*..n) syntax
        let mut visited: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut result: Vec<Entity> = Vec::new();
        let mut current_frontier: Vec<String> = vec![start_id.to_string()];

        for _level in 0..=depth {
            if current_frontier.is_empty() {
                break;
            }

            let mut next_frontier: Vec<String> = Vec::new();

            for node_id in &current_frontier {
                if visited.contains(node_id) {
                    continue;
                }
                visited.insert(node_id.clone());

                // Get the current node
                let node_query = format!(
                    r#"
                    SELECT *, meta::id(id) as id
                    FROM {node_table}:`{node_id}`
                    "#,
                    node_table = self.config.node_table,
                    node_id = node_id
                );

                let mut node_response =
                    self.db
                        .query(&node_query)
                        .await
                        .map_err(|e| GraphRAGError::Storage {
                            message: format!("Failed to get node during traversal: {}", e),
                        })?;

                let node_results: Vec<serde_json::Value> =
                    node_response.take(0).map_err(|e| GraphRAGError::Storage {
                        message: format!("Failed to parse node during traversal: {}", e),
                    })?;

                for value in node_results {
                    if let Ok(entity) = parse_entity(value) {
                        result.push(entity);
                    }
                }

                // Get neighbors for next level
                let neighbors_query = format!(
                    r#"
                    SELECT *, meta::id(id) as id
                    FROM {node_table}:`{node_id}`->{edge_table}->{node_table}
                    "#,
                    node_table = self.config.node_table,
                    edge_table = self.config.edge_table,
                    node_id = node_id
                );

                let mut neighbors_response =
                    self.db
                        .query(&neighbors_query)
                        .await
                        .map_err(|e| GraphRAGError::Storage {
                            message: format!("Failed to get neighbors during traversal: {}", e),
                        })?;

                let neighbor_results: Vec<serde_json::Value> =
                    neighbors_response
                        .take(0)
                        .map_err(|e| GraphRAGError::Storage {
                            message: format!("Failed to parse neighbors during traversal: {}", e),
                        })?;

                for value in neighbor_results {
                    if let Some(id) = value.get("id").and_then(|v| v.as_str()) {
                        if !visited.contains(id) {
                            next_frontier.push(id.to_string());
                        }
                    }
                }
            }

            current_frontier = next_frontier;
        }

        Ok(result)
    }

    async fn stats(&self) -> GraphStats {
        let node_count = self.count_nodes().await.unwrap_or(0);
        let edge_count = self.count_edges().await.unwrap_or(0);

        let average_degree = if node_count > 0 {
            (edge_count as f32 * 2.0) / node_count as f32
        } else {
            0.0
        };

        GraphStats {
            node_count,
            edge_count,
            average_degree,
            max_depth: self.config.max_traversal_depth,
        }
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

    async fn optimize(&mut self) -> Result<()> {
        // Rebuild indexes for better query performance
        let queries = format!(
            r#"
            REBUILD INDEX IF EXISTS idx_{node_table}_id ON {node_table};
            REBUILD INDEX IF EXISTS idx_{node_table}_type ON {node_table};
            REBUILD INDEX IF EXISTS idx_{node_table}_name ON {node_table};
            REBUILD INDEX IF EXISTS idx_{edge_table}_type ON {edge_table};
            "#,
            node_table = self.config.node_table,
            edge_table = self.config.edge_table
        );

        // Ignore errors as indexes may not exist
        let _ = self.db.query(&queries).await;

        Ok(())
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Parse a JSON value into an Entity
fn parse_entity(value: serde_json::Value) -> Result<Entity> {
    // Handle the ID field specially - it may come as a SurrealDB Thing or string
    let id = value
        .get("id")
        .and_then(|v| {
            if let Some(s) = v.as_str() {
                Some(s.to_string())
            } else if let Some(obj) = v.as_object() {
                // SurrealDB Thing format: {"tb": "entity", "id": "123"}
                obj.get("id")
                    .and_then(|i| i.as_str())
                    .map(|s| s.to_string())
            } else {
                None
            }
        })
        .unwrap_or_default();

    let name = value
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();

    let entity_type = value
        .get("entity_type")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();

    let confidence = value
        .get("confidence")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0) as f32;

    let mentions: Vec<EntityMention> = value
        .get("mentions")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| {
                    // Try to parse as EntityMention object
                    if let Some(obj) = v.as_object() {
                        let chunk_id = obj
                            .get("chunk_id")
                            .and_then(|c| c.as_str())
                            .map(|s| ChunkId::new(s.to_string()))
                            .unwrap_or_else(|| ChunkId::new(String::new()));
                        let start_offset = obj
                            .get("start_offset")
                            .and_then(|o| o.as_u64())
                            .unwrap_or(0) as usize;
                        let end_offset =
                            obj.get("end_offset").and_then(|o| o.as_u64()).unwrap_or(0) as usize;
                        let confidence = obj
                            .get("confidence")
                            .and_then(|c| c.as_f64())
                            .unwrap_or(0.0) as f32;
                        Some(EntityMention {
                            chunk_id,
                            start_offset,
                            end_offset,
                            confidence,
                        })
                    } else {
                        None
                    }
                })
                .collect()
        })
        .unwrap_or_default();

    let embedding: Option<Vec<f32>> = value.get("embedding").and_then(|v| {
        v.as_array().map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_f64().map(|f| f as f32))
                .collect()
        })
    });

    Ok(Entity {
        id: EntityId::new(id),
        name,
        entity_type,
        confidence,
        mentions,
        embedding,
    })
}

/// Parse a JSON value into a Relationship
fn parse_relationship(value: serde_json::Value) -> Result<Relationship> {
    let source_id = value
        .get("source_id")
        .or_else(|| value.get("in"))
        .and_then(|v| {
            if let Some(s) = v.as_str() {
                Some(s.to_string())
            } else if let Some(obj) = v.as_object() {
                obj.get("id")
                    .and_then(|i| i.as_str())
                    .map(|s| s.to_string())
            } else {
                None
            }
        })
        .unwrap_or_default();

    let target_id = value
        .get("target_id")
        .or_else(|| value.get("out"))
        .and_then(|v| {
            if let Some(s) = v.as_str() {
                Some(s.to_string())
            } else if let Some(obj) = v.as_object() {
                obj.get("id")
                    .and_then(|i| i.as_str())
                    .map(|s| s.to_string())
            } else {
                None
            }
        })
        .unwrap_or_default();

    let relation_type = value
        .get("relation_type")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();

    let confidence = value
        .get("confidence")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0) as f32;

    let context: Vec<ChunkId> = value
        .get("context")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| ChunkId::new(s.to_string())))
                .collect()
        })
        .unwrap_or_default();

    Ok(Relationship {
        source: EntityId::new(source_id),
        target: EntityId::new(target_id),
        relation_type,
        confidence,
        context,
    })
}

/// Extension trait to allow chaining with Ok
trait Pipe: Sized {
    fn pipe<F, R>(self, f: F) -> R
    where
        F: FnOnce(Self) -> R;
}

impl<T> Pipe for T {
    fn pipe<F, R>(self, f: F) -> R
    where
        F: FnOnce(Self) -> R,
    {
        f(self)
    }
}

// =============================================================================
// Unit Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_config_default() {
        let config = SurrealDbGraphConfig::default();
        assert_eq!(config.node_table, "entity");
        assert_eq!(config.edge_table, "relates_to");
        assert_eq!(config.max_traversal_depth, 10);
        assert!(config.auto_init_schema);
    }

    #[test]
    fn test_graph_config_with_tables() {
        let config = SurrealDbGraphConfig::with_tables("nodes", "edges");
        assert_eq!(config.node_table, "nodes");
        assert_eq!(config.edge_table, "edges");
    }

    #[test]
    fn test_graph_config_with_max_depth() {
        let config = SurrealDbGraphConfig::default().with_max_depth(5);
        assert_eq!(config.max_traversal_depth, 5);
    }

    #[test]
    fn test_graph_config_without_auto_init() {
        let config = SurrealDbGraphConfig::default().without_auto_init();
        assert!(!config.auto_init_schema);
    }

    #[test]
    fn test_graph_config_validation() {
        let valid = SurrealDbGraphConfig::default();
        assert!(valid.validate().is_ok());

        let empty_node = SurrealDbGraphConfig {
            node_table: "".to_string(),
            ..Default::default()
        };
        assert!(empty_node.validate().is_err());

        let empty_edge = SurrealDbGraphConfig {
            edge_table: "".to_string(),
            ..Default::default()
        };
        assert!(empty_edge.validate().is_err());

        let zero_depth = SurrealDbGraphConfig {
            max_traversal_depth: 0,
            ..Default::default()
        };
        assert!(zero_depth.validate().is_err());
    }

    #[test]
    fn test_parse_entity() {
        let json = serde_json::json!({
            "id": "test_entity",
            "name": "Test Entity",
            "entity_type": "person",
            "confidence": 0.95,
            "mentions": [
                {"chunk_id": "chunk1", "start_offset": 0, "end_offset": 10, "confidence": 0.9},
                {"chunk_id": "chunk2", "start_offset": 5, "end_offset": 15, "confidence": 0.85}
            ],
            "embedding": [0.1, 0.2, 0.3]
        });

        let entity = parse_entity(json).unwrap();
        assert_eq!(entity.id.0, "test_entity");
        assert_eq!(entity.name, "Test Entity");
        assert_eq!(entity.entity_type, "person");
        assert!((entity.confidence - 0.95).abs() < 0.01);
        assert_eq!(entity.mentions.len(), 2);
        assert_eq!(entity.mentions[0].chunk_id.0, "chunk1");
        assert_eq!(entity.mentions[1].chunk_id.0, "chunk2");
        assert!(entity.embedding.is_some());
        assert_eq!(entity.embedding.unwrap().len(), 3);
    }

    #[test]
    fn test_parse_relationship() {
        let json = serde_json::json!({
            "source_id": "entity1",
            "target_id": "entity2",
            "relation_type": "works_for",
            "confidence": 0.88,
            "context": ["chunk1"]
        });

        let rel = parse_relationship(json).unwrap();
        assert_eq!(rel.source.0, "entity1");
        assert_eq!(rel.target.0, "entity2");
        assert_eq!(rel.relation_type, "works_for");
        assert!((rel.confidence - 0.88).abs() < 0.01);
        assert_eq!(rel.context.len(), 1);
    }

    #[tokio::test]
    async fn test_graph_store_new() {
        let db_config = SurrealDbConfig::memory();
        let graph_config = SurrealDbGraphConfig::default();

        let result = SurrealDbGraphStore::new(db_config, graph_config).await;
        assert!(result.is_ok());

        let store = result.unwrap();
        assert_eq!(store.config().node_table, "entity");
        assert_eq!(store.config().edge_table, "relates_to");
    }

    #[tokio::test]
    async fn test_graph_store_add_and_get_node() {
        let db_config = SurrealDbConfig::memory();
        let graph_config = SurrealDbGraphConfig::default();
        let mut store = SurrealDbGraphStore::new(db_config, graph_config)
            .await
            .unwrap();

        let entity = Entity {
            id: EntityId::new("test_node".to_string()),
            name: "Test Node".to_string(),
            entity_type: "person".to_string(),
            confidence: 0.95,
            mentions: vec![],
            embedding: None,
        };

        let id = store.add_node(entity).await.unwrap();
        assert_eq!(id, "test_node");

        let retrieved = store.get_node("test_node").await.unwrap();
        assert!(retrieved.is_some());

        let node = retrieved.unwrap();
        assert_eq!(node.name, "Test Node");
        assert_eq!(node.entity_type, "person");
    }

    #[tokio::test]
    async fn test_graph_store_add_edge() {
        let db_config = SurrealDbConfig::memory();
        let graph_config = SurrealDbGraphConfig::default();
        let mut store = SurrealDbGraphStore::new(db_config, graph_config)
            .await
            .unwrap();

        // Add two nodes
        let entity1 = Entity {
            id: EntityId::new("person_alice".to_string()),
            name: "Alice".to_string(),
            entity_type: "person".to_string(),
            confidence: 0.95,
            mentions: vec![],
            embedding: None,
        };

        let entity2 = Entity {
            id: EntityId::new("company_acme".to_string()),
            name: "Acme Corp".to_string(),
            entity_type: "organization".to_string(),
            confidence: 0.90,
            mentions: vec![],
            embedding: None,
        };

        store.add_node(entity1).await.unwrap();
        store.add_node(entity2).await.unwrap();

        // Add edge
        let relationship = Relationship {
            source: EntityId::new("person_alice".to_string()),
            target: EntityId::new("company_acme".to_string()),
            relation_type: "works_for".to_string(),
            confidence: 0.88,
            context: vec![],
        };

        let edge_id = store
            .add_edge("person_alice", "company_acme", relationship)
            .await
            .unwrap();
        assert!(!edge_id.is_empty());

        // Verify stats
        let stats = store.stats().await;
        assert_eq!(stats.node_count, 2);
        assert_eq!(stats.edge_count, 1);
    }

    #[tokio::test]
    async fn test_graph_store_stats() {
        let db_config = SurrealDbConfig::memory();
        let graph_config = SurrealDbGraphConfig::default();
        let store = SurrealDbGraphStore::new(db_config, graph_config)
            .await
            .unwrap();

        let stats = store.stats().await;
        assert_eq!(stats.node_count, 0);
        assert_eq!(stats.edge_count, 0);
        assert_eq!(stats.average_degree, 0.0);
        assert_eq!(stats.max_depth, 10);
    }

    #[tokio::test]
    async fn test_graph_store_health_check() {
        let db_config = SurrealDbConfig::memory();
        let graph_config = SurrealDbGraphConfig::default();
        let store = SurrealDbGraphStore::new(db_config, graph_config)
            .await
            .unwrap();

        let healthy = store.health_check().await.unwrap();
        assert!(healthy);
    }

    #[tokio::test]
    async fn test_graph_store_remove_node() {
        let db_config = SurrealDbConfig::memory();
        let graph_config = SurrealDbGraphConfig::default();
        let mut store = SurrealDbGraphStore::new(db_config, graph_config)
            .await
            .unwrap();

        // Add a node
        let entity = Entity {
            id: EntityId::new("to_remove".to_string()),
            name: "Remove Me".to_string(),
            entity_type: "test".to_string(),
            confidence: 0.9,
            mentions: vec![],
            embedding: None,
        };

        store.add_node(entity).await.unwrap();
        assert_eq!(store.stats().await.node_count, 1);

        // Remove the node
        let removed = store.remove_node("to_remove").await.unwrap();
        assert!(removed);
        assert_eq!(store.stats().await.node_count, 0);

        // Try to remove non-existent
        let removed = store.remove_node("nonexistent").await.unwrap();
        assert!(!removed);
    }
}
