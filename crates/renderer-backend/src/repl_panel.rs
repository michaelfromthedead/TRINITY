//! REPL (Read-Eval-Print-Loop) Panel
//!
//! This module provides a REPL interface for the editor with:
//!
//! - **REPLHistory**: SQLite-persisted command history with search and navigation
//! - **MultilineInput**: Multi-line text editing with cursor movement and auto-indent
//! - **SyntaxHighlighter**: Python syntax highlighting (keywords, strings, numbers, comments)
//! - **OutputBlock**: Formatted output display (text, errors, tables, code)
//! - **REPLPanel**: Main UI component integrating all features
//!
//! # Example
//!
//! ```rust,ignore
//! use renderer_backend::repl_panel::{REPLPanel, REPLConfig};
//!
//! let mut panel = REPLPanel::with_config(REPLConfig::default());
//! panel.execute("print('Hello, world!')");
//!
//! // Render the panel
//! panel.render(&mut ui_context);
//! ```

use crate::egui_adapter::UIContext;
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// Token and TokenKind for Syntax Highlighting
// ---------------------------------------------------------------------------

/// Kind of token for syntax highlighting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenKind {
    /// Plain text (identifiers, operators, etc.)
    Plain,
    /// Python keyword (def, class, if, else, etc.)
    Keyword,
    /// Python builtin function (print, len, range, etc.)
    Builtin,
    /// String literal (single, double, triple quoted)
    String,
    /// Numeric literal (int, float, hex, binary)
    Number,
    /// Comment (# to end of line)
    Comment,
    /// Decorator (@...)
    Decorator,
    /// Operator (=, +, -, *, /, etc.)
    Operator,
    /// Delimiter (parentheses, brackets, braces)
    Delimiter,
}

/// A token with its text span and kind.
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    /// The text content of the token.
    pub text: String,
    /// The kind of token.
    pub kind: TokenKind,
    /// Start column (0-indexed).
    pub start: usize,
    /// End column (exclusive).
    pub end: usize,
}

impl Token {
    /// Create a new token.
    pub fn new(text: impl Into<String>, kind: TokenKind, start: usize, end: usize) -> Self {
        Self {
            text: text.into(),
            kind,
            start,
            end,
        }
    }
}

// ---------------------------------------------------------------------------
// SyntaxHighlighter
// ---------------------------------------------------------------------------

/// Python keywords for syntax highlighting.
const PYTHON_KEYWORDS: &[&str] = &[
    "False", "None", "True", "and", "as", "assert", "async", "await",
    "break", "class", "continue", "def", "del", "elif", "else", "except",
    "finally", "for", "from", "global", "if", "import", "in", "is",
    "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try",
    "while", "with", "yield",
];

/// Python builtin functions.
const PYTHON_BUILTINS: &[&str] = &[
    "abs", "all", "any", "ascii", "bin", "bool", "breakpoint", "bytearray",
    "bytes", "callable", "chr", "classmethod", "compile", "complex",
    "delattr", "dict", "dir", "divmod", "enumerate", "eval", "exec",
    "filter", "float", "format", "frozenset", "getattr", "globals",
    "hasattr", "hash", "help", "hex", "id", "input", "int", "isinstance",
    "issubclass", "iter", "len", "list", "locals", "map", "max",
    "memoryview", "min", "next", "object", "oct", "open", "ord", "pow",
    "print", "property", "range", "repr", "reversed", "round", "set",
    "setattr", "slice", "sorted", "staticmethod", "str", "sum", "super",
    "tuple", "type", "vars", "zip",
];

/// Syntax highlighter for Python code.
#[derive(Debug, Clone)]
pub struct SyntaxHighlighter {
    /// Cached keyword lookup.
    keywords: std::collections::HashSet<&'static str>,
    /// Cached builtin lookup.
    builtins: std::collections::HashSet<&'static str>,
}

impl Default for SyntaxHighlighter {
    fn default() -> Self {
        Self::new()
    }
}

impl SyntaxHighlighter {
    /// Create a new syntax highlighter.
    pub fn new() -> Self {
        Self {
            keywords: PYTHON_KEYWORDS.iter().copied().collect(),
            builtins: PYTHON_BUILTINS.iter().copied().collect(),
        }
    }

    /// Check if a string is a Python keyword.
    pub fn is_keyword(&self, s: &str) -> bool {
        self.keywords.contains(s)
    }

    /// Check if a string is a Python builtin.
    pub fn is_builtin(&self, s: &str) -> bool {
        self.builtins.contains(s)
    }

    /// Tokenize a line of Python code.
    pub fn tokenize_line(&self, line: &str) -> Vec<Token> {
        let mut tokens = Vec::new();
        let chars: Vec<char> = line.chars().collect();
        let len = chars.len();
        let mut i = 0;

        while i < len {
            let start = i;
            let ch = chars[i];

            // Comment
            if ch == '#' {
                let text: String = chars[i..].iter().collect();
                tokens.push(Token::new(text, TokenKind::Comment, start, len));
                break;
            }

            // String literals
            if ch == '"' || ch == '\'' {
                let (text, end) = self.scan_string(&chars, i);
                tokens.push(Token::new(text, TokenKind::String, start, end));
                i = end;
                continue;
            }

            // Numeric literals
            if ch.is_ascii_digit() || (ch == '.' && i + 1 < len && chars[i + 1].is_ascii_digit()) {
                let (text, end) = self.scan_number(&chars, i);
                tokens.push(Token::new(text, TokenKind::Number, start, end));
                i = end;
                continue;
            }

            // Decorator
            if ch == '@' && (start == 0 || chars[..start].iter().all(|c| c.is_whitespace())) {
                let (text, end) = self.scan_identifier(&chars, i + 1);
                let full_text = format!("@{}", text);
                tokens.push(Token::new(full_text, TokenKind::Decorator, start, end));
                i = end;
                continue;
            }

            // Identifiers and keywords
            if ch.is_alphabetic() || ch == '_' {
                let (text, end) = self.scan_identifier(&chars, i);
                let kind = if self.is_keyword(&text) {
                    TokenKind::Keyword
                } else if self.is_builtin(&text) {
                    TokenKind::Builtin
                } else {
                    TokenKind::Plain
                };
                tokens.push(Token::new(text, kind, start, end));
                i = end;
                continue;
            }

            // Operators
            if "+-*/%=<>!&|^~".contains(ch) {
                let (text, end) = self.scan_operator(&chars, i);
                tokens.push(Token::new(text, TokenKind::Operator, start, end));
                i = end;
                continue;
            }

            // Delimiters
            if "()[]{},:;.".contains(ch) {
                let text = ch.to_string();
                tokens.push(Token::new(text, TokenKind::Delimiter, start, i + 1));
                i += 1;
                continue;
            }

            // Whitespace and other
            if ch.is_whitespace() {
                // Skip whitespace (don't emit tokens for it)
                i += 1;
                continue;
            }

            // Unknown character
            tokens.push(Token::new(ch.to_string(), TokenKind::Plain, start, i + 1));
            i += 1;
        }

        tokens
    }

    /// Scan a string literal starting at position `start`.
    fn scan_string(&self, chars: &[char], start: usize) -> (String, usize) {
        let quote = chars[start];
        let len = chars.len();
        let mut i = start + 1;

        // Check for triple quotes
        let triple = i + 1 < len && chars[i] == quote && chars[i + 1] == quote;
        if triple {
            i += 2;
            // Scan until closing triple quote
            while i + 2 < len {
                if chars[i] == quote && chars[i + 1] == quote && chars[i + 2] == quote {
                    i += 3;
                    break;
                }
                if chars[i] == '\\' && i + 1 < len {
                    i += 2;
                } else {
                    i += 1;
                }
            }
            // If we reach end without finding closing, consume rest
            if i + 2 >= len && !(i >= 3 && chars[i-1] == quote && chars[i-2] == quote && chars[i-3] == quote) {
                i = len;
            }
        } else {
            // Single or double quote
            while i < len && chars[i] != quote {
                if chars[i] == '\\' && i + 1 < len {
                    i += 2;
                } else {
                    i += 1;
                }
            }
            if i < len && chars[i] == quote {
                i += 1;
            }
        }

        let text: String = chars[start..i].iter().collect();
        (text, i)
    }

    /// Scan a numeric literal starting at position `start`.
    fn scan_number(&self, chars: &[char], start: usize) -> (String, usize) {
        let len = chars.len();
        let mut i = start;

        // Handle hex, octal, binary prefixes
        if i < len && chars[i] == '0' && i + 1 < len {
            let next = chars[i + 1].to_ascii_lowercase();
            if next == 'x' || next == 'o' || next == 'b' {
                i += 2;
                while i < len && (chars[i].is_ascii_hexdigit() || chars[i] == '_') {
                    i += 1;
                }
                let text: String = chars[start..i].iter().collect();
                return (text, i);
            }
        }

        // Regular number
        while i < len && (chars[i].is_ascii_digit() || chars[i] == '_') {
            i += 1;
        }

        // Decimal part
        if i < len && chars[i] == '.' {
            i += 1;
            while i < len && (chars[i].is_ascii_digit() || chars[i] == '_') {
                i += 1;
            }
        }

        // Exponent part
        if i < len && (chars[i] == 'e' || chars[i] == 'E') {
            i += 1;
            if i < len && (chars[i] == '+' || chars[i] == '-') {
                i += 1;
            }
            while i < len && (chars[i].is_ascii_digit() || chars[i] == '_') {
                i += 1;
            }
        }

        // Complex number suffix
        if i < len && (chars[i] == 'j' || chars[i] == 'J') {
            i += 1;
        }

        let text: String = chars[start..i].iter().collect();
        (text, i)
    }

    /// Scan an identifier starting at position `start`.
    fn scan_identifier(&self, chars: &[char], start: usize) -> (String, usize) {
        let len = chars.len();
        let mut i = start;

        while i < len && (chars[i].is_alphanumeric() || chars[i] == '_') {
            i += 1;
        }

        let text: String = chars[start..i].iter().collect();
        (text, i)
    }

    /// Scan an operator starting at position `start`.
    fn scan_operator(&self, chars: &[char], start: usize) -> (String, usize) {
        let len = chars.len();
        let mut i = start + 1;

        // Multi-character operators
        if i < len {
            let two_char: String = chars[start..i + 1].iter().collect();
            if ["==", "!=", "<=", ">=", "//", "**", "<<", ">>", "+=", "-=",
                "*=", "/=", "%=", "&=", "|=", "^=", "->", "//", "**"].contains(&two_char.as_str()) {
                i += 1;
                // Three-character operators
                if i < len {
                    let three_char: String = chars[start..i + 1].iter().collect();
                    if ["//=", "**=", "<<=", ">>="].contains(&three_char.as_str()) {
                        i += 1;
                    }
                }
            }
        }

        let text: String = chars[start..i].iter().collect();
        (text, i)
    }

    /// Highlight a full block of Python code (multiple lines).
    pub fn highlight(&self, code: &str) -> Vec<Vec<Token>> {
        code.lines().map(|line| self.tokenize_line(line)).collect()
    }
}

// ---------------------------------------------------------------------------
// REPLHistory
// ---------------------------------------------------------------------------

/// A single history entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HistoryEntry {
    /// Unique ID of the entry.
    pub id: i64,
    /// The command text.
    pub command: String,
    /// Unix timestamp when executed.
    pub timestamp: i64,
    /// Number of times this exact command was executed.
    pub execution_count: u32,
}

impl HistoryEntry {
    /// Create a new history entry with the current timestamp.
    pub fn new(id: i64, command: impl Into<String>) -> Self {
        Self {
            id,
            command: command.into(),
            timestamp: current_timestamp() as i64,
            execution_count: 1,
        }
    }
}

/// SQLite-persisted command history.
pub struct REPLHistory {
    /// Database connection.
    conn: Connection,
    /// Current navigation position (0 = most recent).
    nav_position: Option<usize>,
    /// Cached entries for navigation.
    cache: Vec<HistoryEntry>,
    /// Maximum number of entries to keep.
    max_entries: usize,
}

impl REPLHistory {
    /// Create a new in-memory history.
    pub fn new_in_memory() -> Result<Self, rusqlite::Error> {
        let conn = Connection::open_in_memory()?;
        Self::init(conn, 1000)
    }

    /// Create a new history backed by a file.
    pub fn new(path: impl AsRef<Path>) -> Result<Self, rusqlite::Error> {
        let conn = Connection::open(path)?;
        Self::init(conn, 1000)
    }

    /// Create with custom max entries.
    pub fn with_max_entries(path: impl AsRef<Path>, max_entries: usize) -> Result<Self, rusqlite::Error> {
        let conn = Connection::open(path)?;
        Self::init(conn, max_entries)
    }

    fn init(conn: Connection, max_entries: usize) -> Result<Self, rusqlite::Error> {
        conn.execute(
            "CREATE TABLE IF NOT EXISTS repl_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                command TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                execution_count INTEGER DEFAULT 1
            )",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_history_timestamp ON repl_history(timestamp DESC)",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_history_command ON repl_history(command)",
            [],
        )?;

        let mut history = Self {
            conn,
            nav_position: None,
            cache: Vec::new(),
            max_entries,
        };
        history.refresh_cache()?;
        Ok(history)
    }

    /// Refresh the in-memory cache from the database.
    fn refresh_cache(&mut self) -> Result<(), rusqlite::Error> {
        let mut stmt = self.conn.prepare(
            "SELECT id, command, timestamp, execution_count FROM repl_history ORDER BY timestamp DESC, id DESC LIMIT ?",
        )?;
        let entries = stmt.query_map(params![self.max_entries], |row| {
            Ok(HistoryEntry {
                id: row.get(0)?,
                command: row.get(1)?,
                timestamp: row.get(2)?,
                execution_count: row.get(3)?,
            })
        })?;

        self.cache.clear();
        for entry in entries {
            self.cache.push(entry?);
        }
        Ok(())
    }

    /// Add a command to history.
    pub fn add(&mut self, command: impl Into<String>) -> Result<i64, rusqlite::Error> {
        let command = command.into();
        if command.trim().is_empty() {
            return Ok(-1);
        }

        let timestamp = current_timestamp() as i64;

        // Check if this exact command already exists
        let existing: Option<(i64, u32)> = self.conn.query_row(
            "SELECT id, execution_count FROM repl_history WHERE command = ? ORDER BY timestamp DESC LIMIT 1",
            params![&command],
            |row| Ok((row.get(0)?, row.get(1)?)),
        ).optional()?;

        let id = if let Some((existing_id, count)) = existing {
            // Update existing entry
            self.conn.execute(
                "UPDATE repl_history SET timestamp = ?, execution_count = ? WHERE id = ?",
                params![timestamp, count + 1, existing_id],
            )?;
            existing_id
        } else {
            // Insert new entry
            self.conn.execute(
                "INSERT INTO repl_history (command, timestamp, execution_count) VALUES (?, ?, 1)",
                params![&command, timestamp],
            )?;
            self.conn.last_insert_rowid()
        };

        // Enforce max entries limit
        self.enforce_limit()?;
        self.refresh_cache()?;
        self.nav_position = None;

        Ok(id)
    }

    /// Enforce the maximum entries limit by removing oldest entries.
    fn enforce_limit(&mut self) -> Result<usize, rusqlite::Error> {
        let count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM repl_history",
            [],
            |row| row.get(0),
        )?;

        if count as usize > self.max_entries {
            let to_delete = count as usize - self.max_entries;
            self.conn.execute(
                "DELETE FROM repl_history WHERE id IN (SELECT id FROM repl_history ORDER BY timestamp ASC LIMIT ?)",
                params![to_delete],
            )?;
            Ok(to_delete)
        } else {
            Ok(0)
        }
    }

    /// Search history for commands containing the query string.
    pub fn search(&self, query: &str) -> Vec<HistoryEntry> {
        if query.is_empty() {
            return self.cache.clone();
        }

        let pattern = format!("%{}%", query);
        let mut stmt = self.conn.prepare(
            "SELECT id, command, timestamp, execution_count FROM repl_history WHERE command LIKE ? ORDER BY timestamp DESC, id DESC",
        ).unwrap();

        let entries = stmt.query_map(params![pattern], |row| {
            Ok(HistoryEntry {
                id: row.get(0)?,
                command: row.get(1)?,
                timestamp: row.get(2)?,
                execution_count: row.get(3)?,
            })
        }).unwrap();

        entries.filter_map(|e| e.ok()).collect()
    }

    /// Search history using prefix matching.
    pub fn search_prefix(&self, prefix: &str) -> Vec<HistoryEntry> {
        if prefix.is_empty() {
            return self.cache.clone();
        }

        let pattern = format!("{}%", prefix);
        let mut stmt = self.conn.prepare(
            "SELECT id, command, timestamp, execution_count FROM repl_history WHERE command LIKE ? ORDER BY timestamp DESC, id DESC",
        ).unwrap();

        let entries = stmt.query_map(params![pattern], |row| {
            Ok(HistoryEntry {
                id: row.get(0)?,
                command: row.get(1)?,
                timestamp: row.get(2)?,
                execution_count: row.get(3)?,
            })
        }).unwrap();

        entries.filter_map(|e| e.ok()).collect()
    }

    /// Clear all history.
    pub fn clear(&mut self) -> Result<usize, rusqlite::Error> {
        let count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM repl_history",
            [],
            |row| row.get(0),
        )?;
        self.conn.execute("DELETE FROM repl_history", [])?;
        self.cache.clear();
        self.nav_position = None;
        Ok(count as usize)
    }

    /// Get the number of entries in history.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Check if history is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Navigate to the previous (older) command.
    pub fn navigate_prev(&mut self) -> Option<&str> {
        if self.cache.is_empty() {
            return None;
        }

        let new_pos = match self.nav_position {
            None => 0,
            Some(pos) => {
                if pos + 1 < self.cache.len() {
                    pos + 1
                } else {
                    pos
                }
            }
        };

        self.nav_position = Some(new_pos);
        self.cache.get(new_pos).map(|e| e.command.as_str())
    }

    /// Navigate to the next (newer) command.
    pub fn navigate_next(&mut self) -> Option<&str> {
        match self.nav_position {
            None => None,
            Some(0) => {
                self.nav_position = None;
                None
            }
            Some(pos) => {
                self.nav_position = Some(pos - 1);
                self.cache.get(pos - 1).map(|e| e.command.as_str())
            }
        }
    }

    /// Reset navigation position.
    pub fn reset_navigation(&mut self) {
        self.nav_position = None;
    }

    /// Get current navigation position.
    pub fn navigation_position(&self) -> Option<usize> {
        self.nav_position
    }

    /// Get all entries (for display).
    pub fn entries(&self) -> &[HistoryEntry] {
        &self.cache
    }

    /// Get a specific entry by index (0 = most recent).
    pub fn get(&self, index: usize) -> Option<&HistoryEntry> {
        self.cache.get(index)
    }

    /// Delete a specific entry by ID.
    pub fn delete(&mut self, id: i64) -> Result<bool, rusqlite::Error> {
        let affected = self.conn.execute(
            "DELETE FROM repl_history WHERE id = ?",
            params![id],
        )?;
        self.refresh_cache()?;
        Ok(affected > 0)
    }
}

// ---------------------------------------------------------------------------
// MultilineInput
// ---------------------------------------------------------------------------

/// Cursor position in multiline text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CursorPosition {
    /// Line number (0-indexed).
    pub line: usize,
    /// Column (0-indexed, measured in characters).
    pub column: usize,
}

impl CursorPosition {
    /// Create a new cursor position.
    pub fn new(line: usize, column: usize) -> Self {
        Self { line, column }
    }
}

/// Multi-line text input with cursor movement, auto-indent, and completeness detection.
#[derive(Debug, Clone)]
pub struct MultilineInput {
    /// Lines of text.
    lines: Vec<String>,
    /// Current cursor position.
    cursor: CursorPosition,
    /// Indent string (spaces or tab).
    pub indent_str: String,
    /// Whether to auto-indent.
    pub auto_indent: bool,
}

impl Default for MultilineInput {
    fn default() -> Self {
        Self::new()
    }
}

impl MultilineInput {
    /// Create a new empty multiline input.
    pub fn new() -> Self {
        Self {
            lines: vec![String::new()],
            cursor: CursorPosition::default(),
            indent_str: "    ".to_string(),
            auto_indent: true,
        }
    }

    /// Create with custom indent string.
    pub fn with_indent(indent: impl Into<String>) -> Self {
        Self {
            lines: vec![String::new()],
            cursor: CursorPosition::default(),
            indent_str: indent.into(),
            auto_indent: true,
        }
    }

    /// Get the current text.
    pub fn text(&self) -> String {
        self.lines.join("\n")
    }

    /// Set the text content.
    pub fn set_text(&mut self, text: impl AsRef<str>) {
        self.lines = text.as_ref().lines().map(String::from).collect();
        if self.lines.is_empty() {
            self.lines.push(String::new());
        }
        // Clamp cursor to valid position
        self.clamp_cursor();
    }

    /// Clear all text.
    pub fn clear(&mut self) {
        self.lines = vec![String::new()];
        self.cursor = CursorPosition::default();
    }

    /// Get the number of lines.
    pub fn line_count(&self) -> usize {
        self.lines.len()
    }

    /// Get a specific line.
    pub fn line(&self, index: usize) -> Option<&str> {
        self.lines.get(index).map(|s| s.as_str())
    }

    /// Get all lines.
    pub fn lines(&self) -> &[String] {
        &self.lines
    }

    /// Get the current cursor position.
    pub fn cursor(&self) -> CursorPosition {
        self.cursor
    }

    /// Set cursor position.
    pub fn set_cursor(&mut self, pos: CursorPosition) {
        self.cursor = pos;
        self.clamp_cursor();
    }

    /// Clamp cursor to valid range.
    fn clamp_cursor(&mut self) {
        if self.cursor.line >= self.lines.len() {
            self.cursor.line = self.lines.len().saturating_sub(1);
        }
        let line_len = self.lines.get(self.cursor.line).map(|l| l.chars().count()).unwrap_or(0);
        if self.cursor.column > line_len {
            self.cursor.column = line_len;
        }
    }

    /// Get the current line.
    pub fn current_line(&self) -> &str {
        self.lines.get(self.cursor.line).map(|s| s.as_str()).unwrap_or("")
    }

    /// Insert a character at cursor position.
    pub fn insert_char(&mut self, ch: char) {
        if ch == '\n' {
            self.insert_newline();
            return;
        }

        if let Some(line) = self.lines.get_mut(self.cursor.line) {
            let byte_pos = char_to_byte_index(line, self.cursor.column);
            line.insert(byte_pos, ch);
            self.cursor.column += 1;
        }
    }

    /// Insert a string at cursor position.
    pub fn insert_str(&mut self, s: &str) {
        for ch in s.chars() {
            self.insert_char(ch);
        }
    }

    /// Insert a newline with auto-indent.
    pub fn insert_newline(&mut self) {
        let indent = if self.auto_indent {
            self.calculate_indent()
        } else {
            String::new()
        };

        if let Some(line) = self.lines.get_mut(self.cursor.line) {
            let byte_pos = char_to_byte_index(line, self.cursor.column);
            let remainder = line[byte_pos..].to_string();
            line.truncate(byte_pos);

            let new_line = format!("{}{}", indent, remainder);
            self.lines.insert(self.cursor.line + 1, new_line);
            self.cursor.line += 1;
            self.cursor.column = indent.chars().count();
        }
    }

    /// Calculate indent for a new line based on the current line.
    fn calculate_indent(&self) -> String {
        let current = self.current_line();

        // Get existing indent
        let base_indent: String = current.chars().take_while(|c| c.is_whitespace()).collect();

        // Check if line ends with colon (increase indent)
        let trimmed = current.trim_end();
        if trimmed.ends_with(':') {
            return format!("{}{}", base_indent, self.indent_str);
        }

        // Check for dedent keywords
        let first_word = trimmed.split_whitespace().next().unwrap_or("");
        if ["return", "break", "continue", "pass", "raise"].contains(&first_word) {
            // Keep same indent (dedent on following empty line)
            return base_indent;
        }

        base_indent
    }

    /// Delete character before cursor (backspace).
    pub fn delete_char_before(&mut self) {
        if self.cursor.column > 0 {
            if let Some(line) = self.lines.get_mut(self.cursor.line) {
                let byte_pos = char_to_byte_index(line, self.cursor.column - 1);
                let next_byte_pos = char_to_byte_index(line, self.cursor.column);
                line.replace_range(byte_pos..next_byte_pos, "");
                self.cursor.column -= 1;
            }
        } else if self.cursor.line > 0 {
            // Merge with previous line
            let current_line = self.lines.remove(self.cursor.line);
            self.cursor.line -= 1;
            if let Some(prev_line) = self.lines.get_mut(self.cursor.line) {
                self.cursor.column = prev_line.chars().count();
                prev_line.push_str(&current_line);
            }
        }
    }

    /// Delete character at cursor (delete key).
    pub fn delete_char_at(&mut self) {
        if let Some(line) = self.lines.get_mut(self.cursor.line) {
            let char_count = line.chars().count();
            if self.cursor.column < char_count {
                let byte_pos = char_to_byte_index(line, self.cursor.column);
                let next_byte_pos = char_to_byte_index(line, self.cursor.column + 1);
                line.replace_range(byte_pos..next_byte_pos, "");
            } else if self.cursor.line + 1 < self.lines.len() {
                // Merge with next line
                let next_line = self.lines.remove(self.cursor.line + 1);
                if let Some(current_line) = self.lines.get_mut(self.cursor.line) {
                    current_line.push_str(&next_line);
                }
            }
        }
    }

    /// Move cursor left.
    pub fn move_left(&mut self) {
        if self.cursor.column > 0 {
            self.cursor.column -= 1;
        } else if self.cursor.line > 0 {
            self.cursor.line -= 1;
            self.cursor.column = self.lines.get(self.cursor.line)
                .map(|l| l.chars().count())
                .unwrap_or(0);
        }
    }

    /// Move cursor right.
    pub fn move_right(&mut self) {
        let line_len = self.lines.get(self.cursor.line)
            .map(|l| l.chars().count())
            .unwrap_or(0);

        if self.cursor.column < line_len {
            self.cursor.column += 1;
        } else if self.cursor.line + 1 < self.lines.len() {
            self.cursor.line += 1;
            self.cursor.column = 0;
        }
    }

    /// Move cursor up.
    pub fn move_up(&mut self) {
        if self.cursor.line > 0 {
            self.cursor.line -= 1;
            self.clamp_cursor();
        }
    }

    /// Move cursor down.
    pub fn move_down(&mut self) {
        if self.cursor.line + 1 < self.lines.len() {
            self.cursor.line += 1;
            self.clamp_cursor();
        }
    }

    /// Move cursor to start of line.
    pub fn move_to_line_start(&mut self) {
        self.cursor.column = 0;
    }

    /// Move cursor to end of line.
    pub fn move_to_line_end(&mut self) {
        self.cursor.column = self.lines.get(self.cursor.line)
            .map(|l| l.chars().count())
            .unwrap_or(0);
    }

    /// Move cursor to start of text.
    pub fn move_to_start(&mut self) {
        self.cursor.line = 0;
        self.cursor.column = 0;
    }

    /// Move cursor to end of text.
    pub fn move_to_end(&mut self) {
        self.cursor.line = self.lines.len().saturating_sub(1);
        self.cursor.column = self.lines.last()
            .map(|l| l.chars().count())
            .unwrap_or(0);
    }

    /// Move cursor to next word.
    pub fn move_word_right(&mut self) {
        let line = self.current_line().to_string();
        let chars: Vec<char> = line.chars().collect();

        // Skip current word
        while self.cursor.column < chars.len() && chars[self.cursor.column].is_alphanumeric() {
            self.cursor.column += 1;
        }
        // Skip whitespace
        while self.cursor.column < chars.len() && !chars[self.cursor.column].is_alphanumeric() {
            self.cursor.column += 1;
        }

        // If at end of line, move to next line
        if self.cursor.column >= chars.len() && self.cursor.line + 1 < self.lines.len() {
            self.cursor.line += 1;
            self.cursor.column = 0;
        }
    }

    /// Move cursor to previous word.
    pub fn move_word_left(&mut self) {
        if self.cursor.column == 0 {
            if self.cursor.line > 0 {
                self.cursor.line -= 1;
                self.cursor.column = self.lines[self.cursor.line].chars().count();
            }
            return;
        }

        let line = self.current_line().to_string();
        let chars: Vec<char> = line.chars().collect();

        // Move back one
        self.cursor.column = self.cursor.column.saturating_sub(1);

        // Skip whitespace
        while self.cursor.column > 0 && !chars[self.cursor.column].is_alphanumeric() {
            self.cursor.column -= 1;
        }
        // Skip word
        while self.cursor.column > 0 && chars[self.cursor.column - 1].is_alphanumeric() {
            self.cursor.column -= 1;
        }
    }

    /// Check if the input is complete (valid Python block).
    ///
    /// Returns `true` if:
    /// - The input is empty
    /// - The input ends with a blank line after a complete statement
    /// - The input is a single-line complete statement
    pub fn is_complete(&self) -> bool {
        let text = self.text();
        let trimmed = text.trim();

        if trimmed.is_empty() {
            return true;
        }

        // Check for incomplete multi-line constructs
        if self.has_unclosed_brackets(&text) {
            return false;
        }

        if self.has_unclosed_string(&text) {
            return false;
        }

        // Check if ends with colon (expecting body)
        if trimmed.ends_with(':') {
            return false;
        }

        // Check for continuation (backslash at end of line)
        for line in text.lines() {
            if line.trim_end().ends_with('\\') {
                return false;
            }
        }

        // Multi-line block that's indented - needs empty line to complete
        if self.lines.len() > 1 {
            let last_line = self.lines.last().unwrap();
            let prev_line = self.lines.get(self.lines.len() - 2).unwrap();

            // If last line is empty and previous line has content, it's complete
            if last_line.trim().is_empty() && !prev_line.trim().is_empty() {
                return true;
            }

            // If the last line is indented, we're still in a block
            if !last_line.trim().is_empty() {
                let last_indent: String = last_line.chars().take_while(|c| c.is_whitespace()).collect();
                if !last_indent.is_empty() {
                    return false;
                }
            }
        }

        true
    }

    /// Check for unclosed brackets/parens/braces.
    fn has_unclosed_brackets(&self, text: &str) -> bool {
        let mut depth_paren = 0i32;
        let mut depth_bracket = 0i32;
        let mut depth_brace = 0i32;
        let mut in_string = false;
        let mut string_char = ' ';
        let mut chars = text.chars().peekable();

        while let Some(ch) = chars.next() {
            if in_string {
                if ch == '\\' {
                    chars.next(); // Skip escaped char
                } else if ch == string_char {
                    in_string = false;
                }
            } else if ch == '#' {
                // Skip to end of line
                while chars.peek().map(|&c| c != '\n').unwrap_or(false) {
                    chars.next();
                }
            } else if ch == '"' || ch == '\'' {
                in_string = true;
                string_char = ch;
            } else {
                match ch {
                    '(' => depth_paren += 1,
                    ')' => depth_paren -= 1,
                    '[' => depth_bracket += 1,
                    ']' => depth_bracket -= 1,
                    '{' => depth_brace += 1,
                    '}' => depth_brace -= 1,
                    _ => {}
                }
            }
        }

        depth_paren != 0 || depth_bracket != 0 || depth_brace != 0
    }

    /// Check for unclosed string literal.
    fn has_unclosed_string(&self, text: &str) -> bool {
        let mut in_string = false;
        let mut string_char = ' ';
        let mut triple_quote = false;
        let mut chars = text.chars().peekable();

        while let Some(ch) = chars.next() {
            if in_string {
                if ch == '\\' {
                    chars.next();
                } else if triple_quote {
                    if ch == string_char {
                        if chars.peek() == Some(&string_char) {
                            chars.next();
                            if chars.peek() == Some(&string_char) {
                                chars.next();
                                in_string = false;
                                triple_quote = false;
                            }
                        }
                    }
                } else if ch == string_char {
                    in_string = false;
                } else if ch == '\n' {
                    // Newline in non-triple-quoted string is invalid
                    // but we don't treat this as unclosed
                }
            } else if ch == '#' {
                while chars.peek().map(|&c| c != '\n').unwrap_or(false) {
                    chars.next();
                }
            } else if ch == '"' || ch == '\'' {
                string_char = ch;
                if chars.peek() == Some(&ch) {
                    chars.next();
                    if chars.peek() == Some(&ch) {
                        chars.next();
                        triple_quote = true;
                        in_string = true;
                    } else {
                        // Empty string ""
                    }
                } else {
                    in_string = true;
                }
            }
        }

        in_string
    }
}

/// Convert character index to byte index in a string.
fn char_to_byte_index(s: &str, char_idx: usize) -> usize {
    s.char_indices()
        .nth(char_idx)
        .map(|(i, _)| i)
        .unwrap_or(s.len())
}

// ---------------------------------------------------------------------------
// OutputBlock
// ---------------------------------------------------------------------------

/// Kind of output block in the REPL.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OutputBlock {
    /// Plain text output.
    Text(String),
    /// Error output (exception, traceback).
    Error {
        /// Error type (e.g., "TypeError").
        error_type: String,
        /// Error message.
        message: String,
        /// Optional traceback lines.
        traceback: Option<Vec<String>>,
    },
    /// Tabular data.
    Table {
        /// Column headers.
        headers: Vec<String>,
        /// Rows of data.
        rows: Vec<Vec<String>>,
    },
    /// Code block (syntax highlighted).
    Code {
        /// The code content.
        code: String,
        /// Language hint (e.g., "python").
        language: String,
    },
    /// Image output (base64 encoded).
    Image {
        /// Base64-encoded image data.
        data: String,
        /// MIME type.
        mime_type: String,
        /// Optional alt text.
        alt: Option<String>,
    },
    /// HTML content.
    Html(String),
}

impl OutputBlock {
    /// Create a text output block.
    pub fn text(s: impl Into<String>) -> Self {
        Self::Text(s.into())
    }

    /// Create an error output block.
    pub fn error(error_type: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Error {
            error_type: error_type.into(),
            message: message.into(),
            traceback: None,
        }
    }

    /// Create an error with traceback.
    pub fn error_with_traceback(
        error_type: impl Into<String>,
        message: impl Into<String>,
        traceback: Vec<String>,
    ) -> Self {
        Self::Error {
            error_type: error_type.into(),
            message: message.into(),
            traceback: Some(traceback),
        }
    }

    /// Create a table output block.
    pub fn table(headers: Vec<String>, rows: Vec<Vec<String>>) -> Self {
        Self::Table { headers, rows }
    }

    /// Create a code output block.
    pub fn code(code: impl Into<String>, language: impl Into<String>) -> Self {
        Self::Code {
            code: code.into(),
            language: language.into(),
        }
    }

    /// Get the display text for this block.
    pub fn display_text(&self) -> String {
        match self {
            OutputBlock::Text(s) => s.clone(),
            OutputBlock::Error { error_type, message, traceback } => {
                let mut s = format!("{}: {}", error_type, message);
                if let Some(tb) = traceback {
                    s.push('\n');
                    s.push_str(&tb.join("\n"));
                }
                s
            }
            OutputBlock::Table { headers, rows } => {
                let mut s = headers.join(" | ");
                s.push('\n');
                s.push_str(&"-".repeat(s.len()));
                for row in rows {
                    s.push('\n');
                    s.push_str(&row.join(" | "));
                }
                s
            }
            OutputBlock::Code { code, .. } => code.clone(),
            OutputBlock::Image { alt, .. } => alt.clone().unwrap_or_else(|| "[image]".to_string()),
            OutputBlock::Html(html) => html.clone(),
        }
    }

    /// Check if this is an error block.
    pub fn is_error(&self) -> bool {
        matches!(self, OutputBlock::Error { .. })
    }
}

// ---------------------------------------------------------------------------
// REPLCell
// ---------------------------------------------------------------------------

/// A single REPL cell (input + output pair).
#[derive(Debug, Clone)]
pub struct REPLCell {
    /// Unique cell ID.
    pub id: u64,
    /// Input code.
    pub input: String,
    /// Output blocks.
    pub outputs: Vec<OutputBlock>,
    /// Execution timestamp.
    pub timestamp: i64,
    /// Execution duration in milliseconds.
    pub duration_ms: Option<u64>,
    /// Whether the cell is expanded.
    pub expanded: bool,
}

impl REPLCell {
    /// Create a new cell.
    pub fn new(id: u64, input: impl Into<String>) -> Self {
        Self {
            id,
            input: input.into(),
            outputs: Vec::new(),
            timestamp: current_timestamp() as i64,
            duration_ms: None,
            expanded: true,
        }
    }

    /// Add an output block.
    pub fn add_output(&mut self, output: OutputBlock) {
        self.outputs.push(output);
    }

    /// Check if the cell has errors.
    pub fn has_error(&self) -> bool {
        self.outputs.iter().any(|o| o.is_error())
    }
}

// ---------------------------------------------------------------------------
// REPLConfig
// ---------------------------------------------------------------------------

/// Configuration for the REPL panel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct REPLConfig {
    /// Maximum number of cells to display.
    pub max_cells: usize,
    /// Maximum number of history entries.
    pub max_history: usize,
    /// Whether to enable syntax highlighting.
    pub syntax_highlighting: bool,
    /// Whether to enable auto-indent.
    pub auto_indent: bool,
    /// Indent string.
    pub indent_str: String,
    /// Whether to show timestamps.
    pub show_timestamps: bool,
    /// Whether to show execution duration.
    pub show_duration: bool,
    /// Font size for code.
    pub font_size: f32,
}

impl Default for REPLConfig {
    fn default() -> Self {
        Self {
            max_cells: 100,
            max_history: 1000,
            syntax_highlighting: true,
            auto_indent: true,
            indent_str: "    ".to_string(),
            show_timestamps: true,
            show_duration: true,
            font_size: 14.0,
        }
    }
}

// ---------------------------------------------------------------------------
// REPLPanel
// ---------------------------------------------------------------------------

/// Actions that can be triggered by the REPL panel.
#[derive(Debug, Clone, PartialEq)]
pub enum REPLAction {
    /// No action.
    None,
    /// Execute the current input.
    Execute(String),
    /// Clear all cells.
    ClearCells,
    /// Clear history.
    ClearHistory,
    /// Copy cell output to clipboard.
    CopyOutput(u64),
    /// Collapse/expand cell.
    ToggleCell(u64),
    /// Delete a cell.
    DeleteCell(u64),
    /// Rerun a cell.
    RerunCell(u64),
}

/// The main REPL panel component.
pub struct REPLPanel {
    /// Configuration.
    config: REPLConfig,
    /// Command history.
    history: REPLHistory,
    /// Current input.
    input: MultilineInput,
    /// Syntax highlighter.
    highlighter: SyntaxHighlighter,
    /// Executed cells.
    cells: VecDeque<REPLCell>,
    /// Next cell ID.
    next_cell_id: u64,
    /// Whether history search is active.
    history_search_active: bool,
    /// Current history search query.
    history_search_query: String,
    /// Filtered history results.
    history_search_results: Vec<HistoryEntry>,
    /// Whether input is focused.
    input_focused: bool,
}

impl REPLPanel {
    /// Create a new REPL panel with in-memory history.
    pub fn new() -> Self {
        Self::with_config(REPLConfig::default())
    }

    /// Create a new REPL panel with custom configuration.
    pub fn with_config(config: REPLConfig) -> Self {
        let history = REPLHistory::new_in_memory().expect("Failed to create history");
        let mut input = MultilineInput::new();
        input.auto_indent = config.auto_indent;
        input.indent_str = config.indent_str.clone();

        Self {
            config,
            history,
            input,
            highlighter: SyntaxHighlighter::new(),
            cells: VecDeque::new(),
            next_cell_id: 1,
            history_search_active: false,
            history_search_query: String::new(),
            history_search_results: Vec::new(),
            input_focused: true,
        }
    }

    /// Create with a file-backed history.
    pub fn with_history_file(path: impl AsRef<std::path::Path>, config: REPLConfig) -> Result<Self, rusqlite::Error> {
        let history = REPLHistory::with_max_entries(path, config.max_history)?;
        let mut input = MultilineInput::new();
        input.auto_indent = config.auto_indent;
        input.indent_str = config.indent_str.clone();

        Ok(Self {
            config,
            history,
            input,
            highlighter: SyntaxHighlighter::new(),
            cells: VecDeque::new(),
            next_cell_id: 1,
            history_search_active: false,
            history_search_query: String::new(),
            history_search_results: Vec::new(),
            input_focused: true,
        })
    }

    /// Get the current input text.
    pub fn input_text(&self) -> String {
        self.input.text()
    }

    /// Set the input text.
    pub fn set_input(&mut self, text: impl AsRef<str>) {
        self.input.set_text(text);
    }

    /// Clear the input.
    pub fn clear_input(&mut self) {
        self.input.clear();
    }

    /// Get the cursor position.
    pub fn cursor(&self) -> CursorPosition {
        self.input.cursor()
    }

    /// Check if the input is complete.
    pub fn is_complete(&self) -> bool {
        self.input.is_complete()
    }

    /// Execute the current input.
    ///
    /// Returns the input text if execution should proceed.
    pub fn execute(&mut self) -> Option<String> {
        let text = self.input.text();
        if text.trim().is_empty() {
            return None;
        }

        // Add to history
        let _ = self.history.add(&text);

        // Create cell
        let cell = REPLCell::new(self.next_cell_id, &text);
        self.next_cell_id += 1;

        self.cells.push_back(cell);

        // Enforce max cells
        while self.cells.len() > self.config.max_cells {
            self.cells.pop_front();
        }

        // Clear input
        self.input.clear();

        Some(text)
    }

    /// Add output to the most recent cell.
    pub fn add_output(&mut self, output: OutputBlock) {
        if let Some(cell) = self.cells.back_mut() {
            cell.add_output(output);
        }
    }

    /// Set execution duration for the most recent cell.
    pub fn set_duration(&mut self, duration_ms: u64) {
        if let Some(cell) = self.cells.back_mut() {
            cell.duration_ms = Some(duration_ms);
        }
    }

    /// Get all cells.
    pub fn cells(&self) -> &VecDeque<REPLCell> {
        &self.cells
    }

    /// Clear all cells.
    pub fn clear_cells(&mut self) {
        self.cells.clear();
    }

    /// Clear history.
    pub fn clear_history(&mut self) -> Result<usize, rusqlite::Error> {
        self.history.clear()
    }

    /// Navigate to previous history entry.
    pub fn history_prev(&mut self) {
        if let Some(cmd) = self.history.navigate_prev() {
            self.input.set_text(cmd);
        }
    }

    /// Navigate to next history entry.
    pub fn history_next(&mut self) {
        if let Some(cmd) = self.history.navigate_next() {
            self.input.set_text(cmd);
        } else {
            self.input.clear();
        }
    }

    /// Start history search.
    pub fn start_history_search(&mut self) {
        self.history_search_active = true;
        self.history_search_query.clear();
        self.history_search_results = self.history.entries().to_vec();
    }

    /// Update history search query.
    pub fn update_history_search(&mut self, query: &str) {
        self.history_search_query = query.to_string();
        self.history_search_results = self.history.search(query);
    }

    /// Select a history search result.
    pub fn select_history_result(&mut self, index: usize) {
        if let Some(entry) = self.history_search_results.get(index) {
            self.input.set_text(&entry.command);
            self.history_search_active = false;
        }
    }

    /// Cancel history search.
    pub fn cancel_history_search(&mut self) {
        self.history_search_active = false;
        self.history_search_query.clear();
        self.history_search_results.clear();
    }

    /// Get history.
    pub fn history(&self) -> &REPLHistory {
        &self.history
    }

    /// Get mutable history.
    pub fn history_mut(&mut self) -> &mut REPLHistory {
        &mut self.history
    }

    /// Get input.
    pub fn input(&self) -> &MultilineInput {
        &self.input
    }

    /// Get mutable input.
    pub fn input_mut(&mut self) -> &mut MultilineInput {
        &mut self.input
    }

    /// Get highlighter.
    pub fn highlighter(&self) -> &SyntaxHighlighter {
        &self.highlighter
    }

    /// Highlight the current input.
    pub fn highlight_input(&self) -> Vec<Vec<Token>> {
        self.highlighter.highlight(&self.input.text())
    }

    /// Render the REPL panel.
    pub fn render<T: UIContext>(&mut self, ctx: &mut T) -> REPLAction {
        let mut action = REPLAction::None;

        ctx.vertical(|v| {
            // Render cells
            let cells_snapshot: Vec<_> = self.cells.iter().cloned().collect();
            for cell in &cells_snapshot {
                Self::render_cell_static(&self.highlighter, &self.config, v, cell);
            }

            // Render input area
            v.horizontal(|h| {
                h.label(">>> ");
                let mut input_text = self.input.text();
                if h.text_area("input", &mut input_text, 4) {
                    self.input.set_text(&input_text);
                }
            });

            // Handle execution
            if v.button("Run") {
                if let Some(code) = self.execute() {
                    action = REPLAction::Execute(code);
                }
            }
        });

        action
    }

    /// Render a single cell (static version to avoid borrow issues).
    fn render_cell_static<T: UIContext>(
        highlighter: &SyntaxHighlighter,
        config: &REPLConfig,
        ctx: &mut T,
        cell: &REPLCell,
    ) {
        ctx.vertical(|v| {
            // Input with syntax highlighting
            v.horizontal(|h| {
                h.label(&format!("In [{}]:", cell.id));
                let tokens = highlighter.highlight(&cell.input);
                let highlighted = tokens.iter()
                    .map(|line| line.iter().map(|t| t.text.clone()).collect::<Vec<_>>().join(""))
                    .collect::<Vec<_>>()
                    .join("\n");
                h.label(&highlighted);
            });

            // Outputs
            for output in &cell.outputs {
                v.horizontal(|h| {
                    match output {
                        OutputBlock::Text(text) => {
                            h.label(&format!("Out[{}]: {}", cell.id, text));
                        }
                        OutputBlock::Error { error_type, message, traceback } => {
                            h.label(&format!("{}: {}", error_type, message));
                            if let Some(tb) = traceback {
                                for line in tb {
                                    h.label(line);
                                }
                            }
                        }
                        OutputBlock::Table { headers, rows } => {
                            h.label(&headers.join(" | "));
                            for row in rows {
                                h.label(&row.join(" | "));
                            }
                        }
                        OutputBlock::Code { code, .. } => {
                            h.label(code);
                        }
                        OutputBlock::Image { alt, .. } => {
                            h.label(alt.as_deref().unwrap_or("[image]"));
                        }
                        OutputBlock::Html(html) => {
                            h.label(html);
                        }
                    }
                });
            }

            // Duration
            if config.show_duration {
                if let Some(ms) = cell.duration_ms {
                    v.label(&format!("({:.2}ms)", ms));
                }
            }
        });
    }

    /// Get config.
    pub fn config(&self) -> &REPLConfig {
        &self.config
    }

    /// Set config.
    pub fn set_config(&mut self, config: REPLConfig) {
        self.input.auto_indent = config.auto_indent;
        self.input.indent_str = config.indent_str.clone();
        self.config = config;
    }
}

impl Default for REPLPanel {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Utility Functions
// ---------------------------------------------------------------------------

/// Get current Unix timestamp.
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ========================
    // SyntaxHighlighter Tests
    // ========================

    #[test]
    fn test_highlighter_keywords() {
        let h = SyntaxHighlighter::new();
        assert!(h.is_keyword("def"));
        assert!(h.is_keyword("class"));
        assert!(h.is_keyword("if"));
        assert!(h.is_keyword("return"));
        assert!(!h.is_keyword("print")); // builtin, not keyword
        assert!(!h.is_keyword("foo"));
    }

    #[test]
    fn test_highlighter_builtins() {
        let h = SyntaxHighlighter::new();
        assert!(h.is_builtin("print"));
        assert!(h.is_builtin("len"));
        assert!(h.is_builtin("range"));
        assert!(!h.is_builtin("def")); // keyword, not builtin
        assert!(!h.is_builtin("foo"));
    }

    #[test]
    fn test_tokenize_simple() {
        let h = SyntaxHighlighter::new();
        let tokens = h.tokenize_line("x = 42");

        let kinds: Vec<_> = tokens.iter().map(|t| t.kind).collect();
        assert!(kinds.contains(&TokenKind::Plain)); // x
        assert!(kinds.contains(&TokenKind::Operator)); // =
        assert!(kinds.contains(&TokenKind::Number)); // 42
    }

    #[test]
    fn test_tokenize_string() {
        let h = SyntaxHighlighter::new();
        let tokens = h.tokenize_line("print('hello')");

        let string_tokens: Vec<_> = tokens.iter().filter(|t| t.kind == TokenKind::String).collect();
        assert_eq!(string_tokens.len(), 1);
        assert_eq!(string_tokens[0].text, "'hello'");
    }

    #[test]
    fn test_tokenize_comment() {
        let h = SyntaxHighlighter::new();
        let tokens = h.tokenize_line("x = 1  # comment");

        let comment_tokens: Vec<_> = tokens.iter().filter(|t| t.kind == TokenKind::Comment).collect();
        assert_eq!(comment_tokens.len(), 1);
        assert!(comment_tokens[0].text.contains("comment"));
    }

    #[test]
    fn test_tokenize_decorator() {
        let h = SyntaxHighlighter::new();
        let tokens = h.tokenize_line("@property");

        let decorator_tokens: Vec<_> = tokens.iter().filter(|t| t.kind == TokenKind::Decorator).collect();
        assert_eq!(decorator_tokens.len(), 1);
        assert_eq!(decorator_tokens[0].text, "@property");
    }

    #[test]
    fn test_tokenize_numbers() {
        let h = SyntaxHighlighter::new();

        // Integer
        let tokens = h.tokenize_line("42");
        assert!(tokens.iter().any(|t| t.kind == TokenKind::Number && t.text == "42"));

        // Float
        let tokens = h.tokenize_line("3.14");
        assert!(tokens.iter().any(|t| t.kind == TokenKind::Number && t.text == "3.14"));

        // Hex
        let tokens = h.tokenize_line("0xff");
        assert!(tokens.iter().any(|t| t.kind == TokenKind::Number && t.text == "0xff"));

        // Binary
        let tokens = h.tokenize_line("0b101");
        assert!(tokens.iter().any(|t| t.kind == TokenKind::Number && t.text == "0b101"));
    }

    #[test]
    fn test_highlight_multiline() {
        let h = SyntaxHighlighter::new();
        let code = "def foo():\n    return 42";
        let lines = h.highlight(code);
        assert_eq!(lines.len(), 2);
    }

    // ========================
    // REPLHistory Tests
    // ========================

    #[test]
    fn test_history_add_and_get() {
        let mut h = REPLHistory::new_in_memory().unwrap();
        let id = h.add("print('hello')").unwrap();
        assert!(id > 0);
        assert_eq!(h.len(), 1);

        let entry = h.get(0).unwrap();
        assert_eq!(entry.command, "print('hello')");
    }

    #[test]
    fn test_history_navigation() {
        let mut h = REPLHistory::new_in_memory().unwrap();
        h.add("cmd1").unwrap();
        h.add("cmd2").unwrap();
        h.add("cmd3").unwrap();

        // Navigate back
        assert_eq!(h.navigate_prev(), Some("cmd3"));
        assert_eq!(h.navigate_prev(), Some("cmd2"));
        assert_eq!(h.navigate_prev(), Some("cmd1"));
        assert_eq!(h.navigate_prev(), Some("cmd1")); // stays at oldest

        // Navigate forward
        assert_eq!(h.navigate_next(), Some("cmd2"));
        assert_eq!(h.navigate_next(), Some("cmd3"));
        assert_eq!(h.navigate_next(), None); // back to current
    }

    #[test]
    fn test_history_search() {
        let mut h = REPLHistory::new_in_memory().unwrap();
        h.add("print('hello')").unwrap();
        h.add("x = 42").unwrap();
        h.add("print('world')").unwrap();

        let results = h.search("print");
        assert_eq!(results.len(), 2);

        let results = h.search("42");
        assert_eq!(results.len(), 1);

        let results = h.search("nonexistent");
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_history_prefix_search() {
        let mut h = REPLHistory::new_in_memory().unwrap();
        h.add("print('hello')").unwrap();
        h.add("x = 42").unwrap();
        h.add("print('world')").unwrap();

        let results = h.search_prefix("print");
        assert_eq!(results.len(), 2);

        let results = h.search_prefix("x");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_history_clear() {
        let mut h = REPLHistory::new_in_memory().unwrap();
        h.add("cmd1").unwrap();
        h.add("cmd2").unwrap();

        let count = h.clear().unwrap();
        assert_eq!(count, 2);
        assert!(h.is_empty());
    }

    #[test]
    fn test_history_empty_command() {
        let mut h = REPLHistory::new_in_memory().unwrap();
        let id = h.add("   ").unwrap();
        assert_eq!(id, -1); // empty commands not added
        assert!(h.is_empty());
    }

    #[test]
    fn test_history_duplicate_command() {
        let mut h = REPLHistory::new_in_memory().unwrap();
        h.add("print('hello')").unwrap();
        h.add("print('hello')").unwrap();

        // Should update execution count, not add duplicate
        assert_eq!(h.len(), 1);
        let entry = h.get(0).unwrap();
        assert_eq!(entry.execution_count, 2);
    }

    // ========================
    // MultilineInput Tests
    // ========================

    #[test]
    fn test_input_basic() {
        let mut input = MultilineInput::new();
        input.insert_str("hello");
        assert_eq!(input.text(), "hello");
        assert_eq!(input.cursor().column, 5);
    }

    #[test]
    fn test_input_newline() {
        let mut input = MultilineInput::new();
        input.insert_str("line1");
        input.insert_newline();
        input.insert_str("line2");

        assert_eq!(input.line_count(), 2);
        assert_eq!(input.text(), "line1\nline2");
    }

    #[test]
    fn test_input_auto_indent() {
        let mut input = MultilineInput::new();
        input.insert_str("def foo():");
        input.insert_newline();

        // Should be indented
        assert_eq!(input.cursor().column, 4);
    }

    #[test]
    fn test_input_cursor_movement() {
        let mut input = MultilineInput::new();
        input.insert_str("hello");

        input.move_left();
        assert_eq!(input.cursor().column, 4);

        input.move_to_line_start();
        assert_eq!(input.cursor().column, 0);

        input.move_to_line_end();
        assert_eq!(input.cursor().column, 5);
    }

    #[test]
    fn test_input_delete_before() {
        let mut input = MultilineInput::new();
        input.insert_str("hello");
        input.delete_char_before();

        assert_eq!(input.text(), "hell");
    }

    #[test]
    fn test_input_delete_at() {
        let mut input = MultilineInput::new();
        input.insert_str("hello");
        input.move_to_line_start();
        input.delete_char_at();

        assert_eq!(input.text(), "ello");
    }

    #[test]
    fn test_input_multiline_cursor() {
        let mut input = MultilineInput::new();
        input.insert_str("line1");
        input.insert_newline();
        input.insert_str("line2");

        assert_eq!(input.cursor().line, 1);

        input.move_up();
        assert_eq!(input.cursor().line, 0);

        input.move_down();
        assert_eq!(input.cursor().line, 1);
    }

    #[test]
    fn test_input_clear() {
        let mut input = MultilineInput::new();
        input.insert_str("hello\nworld");
        input.clear();

        assert_eq!(input.text(), "");
        assert_eq!(input.line_count(), 1);
        assert_eq!(input.cursor().line, 0);
        assert_eq!(input.cursor().column, 0);
    }

    #[test]
    fn test_input_completeness_simple() {
        let mut input = MultilineInput::new();

        // Empty is complete
        assert!(input.is_complete());

        // Simple statement is complete
        input.set_text("x = 42");
        assert!(input.is_complete());
    }

    #[test]
    fn test_input_completeness_colon() {
        let mut input = MultilineInput::new();
        input.set_text("def foo():");
        assert!(!input.is_complete()); // needs body
    }

    #[test]
    fn test_input_completeness_brackets() {
        let mut input = MultilineInput::new();

        input.set_text("x = [1, 2");
        assert!(!input.is_complete()); // unclosed bracket

        input.set_text("x = [1, 2]");
        assert!(input.is_complete());
    }

    // ========================
    // OutputBlock Tests
    // ========================

    #[test]
    fn test_output_text() {
        let output = OutputBlock::text("Hello, world!");
        assert_eq!(output.display_text(), "Hello, world!");
        assert!(!output.is_error());
    }

    #[test]
    fn test_output_error() {
        let output = OutputBlock::error("TypeError", "expected int");
        assert!(output.is_error());
        assert!(output.display_text().contains("TypeError"));
    }

    #[test]
    fn test_output_error_with_traceback() {
        let output = OutputBlock::error_with_traceback(
            "ValueError",
            "invalid",
            vec!["  File \"test.py\", line 1".to_string()],
        );
        assert!(output.is_error());
        let text = output.display_text();
        assert!(text.contains("ValueError"));
        assert!(text.contains("test.py"));
    }

    #[test]
    fn test_output_table() {
        let output = OutputBlock::table(
            vec!["Name".to_string(), "Value".to_string()],
            vec![vec!["x".to_string(), "42".to_string()]],
        );
        let text = output.display_text();
        assert!(text.contains("Name"));
        assert!(text.contains("42"));
    }

    #[test]
    fn test_output_code() {
        let output = OutputBlock::code("def foo(): pass", "python");
        assert_eq!(output.display_text(), "def foo(): pass");
    }

    // ========================
    // REPLPanel Tests
    // ========================

    #[test]
    fn test_panel_creation() {
        let panel = REPLPanel::new();
        assert!(panel.cells().is_empty());
        assert_eq!(panel.input_text(), "");
    }

    #[test]
    fn test_panel_execute() {
        let mut panel = REPLPanel::new();
        panel.set_input("print('hello')");

        let code = panel.execute();
        assert_eq!(code, Some("print('hello')".to_string()));
        assert_eq!(panel.cells().len(), 1);
        assert_eq!(panel.input_text(), "");
    }

    #[test]
    fn test_panel_add_output() {
        let mut panel = REPLPanel::new();
        panel.set_input("x = 42");
        panel.execute();

        panel.add_output(OutputBlock::text("42"));

        let cell = panel.cells().back().unwrap();
        assert_eq!(cell.outputs.len(), 1);
    }

    #[test]
    fn test_panel_history_navigation() {
        let mut panel = REPLPanel::new();

        panel.set_input("cmd1");
        panel.execute();

        panel.set_input("cmd2");
        panel.execute();

        panel.history_prev();
        assert_eq!(panel.input_text(), "cmd2");

        panel.history_prev();
        assert_eq!(panel.input_text(), "cmd1");

        panel.history_next();
        assert_eq!(panel.input_text(), "cmd2");
    }

    #[test]
    fn test_panel_clear_cells() {
        let mut panel = REPLPanel::new();
        panel.set_input("cmd1");
        panel.execute();
        panel.set_input("cmd2");
        panel.execute();

        assert_eq!(panel.cells().len(), 2);

        panel.clear_cells();
        assert!(panel.cells().is_empty());
    }

    #[test]
    fn test_panel_config() {
        let config = REPLConfig {
            max_cells: 50,
            auto_indent: false,
            ..Default::default()
        };
        let panel = REPLPanel::with_config(config);

        assert_eq!(panel.config().max_cells, 50);
        assert!(!panel.config().auto_indent);
    }

    #[test]
    fn test_panel_highlight_input() {
        let mut panel = REPLPanel::new();
        panel.set_input("def foo(): pass");

        let tokens = panel.highlight_input();
        assert_eq!(tokens.len(), 1);

        // Should have keyword token
        let keywords: Vec<_> = tokens[0].iter().filter(|t| t.kind == TokenKind::Keyword).collect();
        assert!(!keywords.is_empty());
    }

    #[test]
    fn test_panel_max_cells_enforcement() {
        let config = REPLConfig {
            max_cells: 3,
            ..Default::default()
        };
        let mut panel = REPLPanel::with_config(config);

        for i in 0..5 {
            panel.set_input(&format!("cmd{}", i));
            panel.execute();
        }

        assert_eq!(panel.cells().len(), 3);
        // Should have kept the most recent
        assert_eq!(panel.cells().back().unwrap().input, "cmd4");
    }
}
