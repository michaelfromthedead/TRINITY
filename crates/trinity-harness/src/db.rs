//! Database connection and schema management using rusqlite.

use rusqlite::{Connection, Result};

/// Database connection wrapper for the harness.
pub struct HarnessDb {
    conn: Connection,
}

impl HarnessDb {
    /// Open a database connection at the given path.
    pub fn open(path: &str) -> Result<Self> {
        let conn = Connection::open(path)?;
        conn.execute_batch(&format!(
            r#"
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;
            PRAGMA cache_size = {};
        "#,
            crate::constants::SQLITE_CACHE_SIZE
        ))?;
        Self::init_schema(&conn)?;
        Ok(Self { conn })
    }

    /// Open an in-memory database.
    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        Self::init_schema(&conn)?;
        Ok(Self { conn })
    }

    fn init_schema(conn: &Connection) -> Result<()> {
        conn.execute_batch(include_str!("schema.sql"))?;
        Ok(())
    }

    /// Get a reference to the underlying connection.
    pub fn connection(&self) -> &Connection {
        &self.conn
    }
}
