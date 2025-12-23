# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Issue Tracking

This project uses **bd** (beads) for issue tracking. Run `bd onboard` to get started.

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --status in_progress  # Claim work
bd close <id>         # Complete work
bd sync               # Sync with git
```

## Session Completion (Landing the Plane)

When ending a work session, complete ALL steps. Work is NOT complete until `git push` succeeds.

1. File issues for remaining work
2. Run quality gates (if code changed): tests, linters, builds
3. Update issue status
4. **PUSH TO REMOTE** (MANDATORY):
   ```bash
   git pull --rebase
   bd sync
   git push
   git status  # MUST show "up to date with origin"
   ```
5. Verify all changes committed AND pushed

**CRITICAL**: Never stop before pushing. If push fails, resolve and retry until it succeeds.

## Build Commands

```bash
# Build all workspace crates
cargo build --release

# Build specific crates
cargo build -p graphrag-core
cargo build -p graphrag-server --features "qdrant,ollama"
cargo build -p graphrag-cli

# Run tests
cargo test                           # All tests
cargo test -p graphrag-core          # Core library tests
cargo test -p graphrag-core -- test_name  # Single test

# Code quality
cargo clippy --all-targets
cargo fmt --check

# Run the TUI CLI
cargo run -p graphrag-cli

# Run the server
cargo run -p graphrag-server --features "qdrant,ollama"

# WASM build (requires trunk)
cd graphrag-wasm && trunk serve --open
```

## Architecture

### Workspace Structure

```
graphrag-rs/
├── graphrag-core/     # Portable core library (native + WASM compatible)
├── graphrag-wasm/     # WASM bindings and browser integrations
├── graphrag-server/   # REST API server (Actix-web + Apistos OpenAPI)
├── graphrag-cli/      # TUI application (Ratatui-based)
└── examples/          # Demo applications
```

### graphrag-core Modules

The core library (`graphrag-core/src/`) contains:
- `graph/` - Knowledge graph construction with petgraph
- `entity/` - Entity extraction and management
- `embeddings/` - Embedding providers (Ollama, HuggingFace, OpenAI, etc.)
- `retrieval/` - Hybrid retrieval strategies (semantic, keyword, BM25, graph)
- `lightrag/` - Dual-level retrieval (6000x token reduction)
- `rograg/` - Query decomposition
- `query/` - Query processing pipeline
- `caching/` - LLM response caching with moka
- `text/` - Chunking strategies (hierarchical, code-aware with tree-sitter)
- `config/` - TOML/JSON5 configuration parsing
- `ollama/` - Ollama LLM integration

### Feature Flags

Key feature combinations:
- **persistent-storage** and **neural-embeddings** are mutually exclusive
- For production with vector storage: use `persistent-storage`
- For ML experiments: use `neural-embeddings`
- WASM builds: use `wasm` feature (excludes tokio, uses getrandom/js)

Common feature sets:
```toml
# Server with all features
graphrag-core = { features = ["caching", "lightrag", "pagerank", "ollama"] }

# CLI with full functionality  
graphrag-core = { features = ["async", "pagerank", "lightrag", "leiden", "caching", "parallel-processing", "ollama", "rograg", "cross-encoder", "incremental", "json5-support", "vector-hnsw"] }
```

### Dependency Graph

```
graphrag-server → graphrag-core
graphrag-cli    → graphrag-core  
graphrag-wasm   → graphrag-core (with wasm feature)
```

### Server Architecture

- **Framework**: Actix-web 4.9 with Apistos for automatic OpenAPI 3.0.3 docs
- **Vector Store**: Qdrant (production) or in-memory (development)
- **Embeddings**: Ollama (GPU) or hash-based fallback
- **Endpoints**: `/api/documents`, `/api/query`, `/api/entities`, `/health`

### CLI Architecture

- **Framework**: Ratatui TUI with crossterm backend
- **Modes**: Normal mode, Query mode, Command mode
- **Direct Integration**: Uses graphrag-core directly (no HTTP)
