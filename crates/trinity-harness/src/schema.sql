-- Trinity Harness Database Schema

-- Code units table: stores parsed code elements
CREATE TABLE IF NOT EXISTS code_units (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT NOT NULL,
    language TEXT NOT NULL,
    unit_type TEXT NOT NULL,
    name TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Edges table: stores relationships between code units
CREATE TABLE IF NOT EXISTS edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL,
    target_id INTEGER NOT NULL,
    edge_type TEXT NOT NULL,
    FOREIGN KEY (source_id) REFERENCES code_units(id),
    FOREIGN KEY (target_id) REFERENCES code_units(id)
);

-- Indices for common queries
CREATE INDEX IF NOT EXISTS idx_code_units_file ON code_units(file_path);
CREATE INDEX IF NOT EXISTS idx_code_units_language ON code_units(language);
CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
