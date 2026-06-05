//! Remote Content Cache Distribution
//!
//! Provides HTTP-accessible remote content caching with local fallback.
//! Features:
//! - Check local first, then remote on miss
//! - Fetch from remote and store locally on hit
//! - Optional compression for transfer (zstd feature-gated, RLE fallback)
//! - Token-based authentication
//! - DeltaSync integration for efficient synchronization

use crate::pipeline::{
    ContentHash, DeltaSyncProtocol, FileBackend, SyncBatch, SyncError, SyncItem,
};
use std::collections::HashSet;
use std::io;

// ---------------------------------------------------------------------------
// RemoteCacheError
// ---------------------------------------------------------------------------

/// Error type for remote cache operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RemoteCacheError {
    /// Network error during remote operation.
    NetworkError(String),
    /// Authentication failed.
    AuthenticationFailed,
    /// Invalid token provided.
    InvalidToken,
    /// Content not found locally or remotely.
    NotFound(ContentHash),
    /// Compression/decompression error.
    CompressionError(String),
    /// Local I/O error.
    IoError(String),
    /// Sync protocol error.
    SyncError(SyncError),
    /// Invalid response from server.
    InvalidResponse(String),
    /// Remote server unavailable.
    ServerUnavailable,
}

impl std::fmt::Display for RemoteCacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NetworkError(msg) => write!(f, "network error: {}", msg),
            Self::AuthenticationFailed => write!(f, "authentication failed"),
            Self::InvalidToken => write!(f, "invalid token"),
            Self::NotFound(hash) => write!(f, "content not found: {}", hash),
            Self::CompressionError(msg) => write!(f, "compression error: {}", msg),
            Self::IoError(msg) => write!(f, "I/O error: {}", msg),
            Self::SyncError(e) => write!(f, "sync error: {}", e),
            Self::InvalidResponse(msg) => write!(f, "invalid response: {}", msg),
            Self::ServerUnavailable => write!(f, "server unavailable"),
        }
    }
}

impl std::error::Error for RemoteCacheError {}

impl From<io::Error> for RemoteCacheError {
    fn from(e: io::Error) -> Self {
        Self::IoError(e.to_string())
    }
}

impl From<SyncError> for RemoteCacheError {
    fn from(e: SyncError) -> Self {
        Self::SyncError(e)
    }
}

// ---------------------------------------------------------------------------
// Compression Module (zstd feature-gated with RLE fallback)
// ---------------------------------------------------------------------------

/// Compression utilities for remote transfer.
pub mod compression {
    /// Compress data for transfer.
    ///
    /// Uses zstd if available, otherwise falls back to simple RLE.
    pub fn compress(data: &[u8]) -> Vec<u8> {
        #[cfg(feature = "zstd")]
        {
            zstd::encode_all(std::io::Cursor::new(data), 3).unwrap_or_else(|_| rle_compress(data))
        }

        #[cfg(not(feature = "zstd"))]
        {
            rle_compress(data)
        }
    }

    /// Decompress data from transfer format.
    ///
    /// Automatically detects format (zstd or RLE).
    pub fn decompress(data: &[u8]) -> Result<Vec<u8>, String> {
        if data.is_empty() {
            return Ok(Vec::new());
        }

        // Check for zstd magic number (0x28 0xB5 0x2F 0xFD)
        #[cfg(feature = "zstd")]
        if data.len() >= 4 && data[0..4] == [0x28, 0xB5, 0x2F, 0xFD] {
            return zstd::decode_all(std::io::Cursor::new(data))
                .map_err(|e| format!("zstd decompress error: {}", e));
        }

        // Fall back to RLE
        rle_decompress(data)
    }

    /// Simple RLE compression for repeated bytes.
    ///
    /// Format: For runs of 4+ same bytes: [0xFE, byte, count_hi, count_lo]
    /// For non-runs: literal bytes (0xFE is escaped as [0xFE, 0xFE, 0x00, 0x01])
    pub fn rle_compress(data: &[u8]) -> Vec<u8> {
        if data.is_empty() {
            return Vec::new();
        }

        let mut result = Vec::with_capacity(data.len());
        let mut i = 0;

        while i < data.len() {
            let byte = data[i];
            let mut run_len = 1;

            // Count run length
            while i + run_len < data.len() && data[i + run_len] == byte && run_len < 65535 {
                run_len += 1;
            }

            if run_len >= 4 {
                // Encode run
                result.push(0xFE);
                result.push(byte);
                result.push((run_len >> 8) as u8);
                result.push((run_len & 0xFF) as u8);
            } else if byte == 0xFE {
                // Escape 0xFE
                for _ in 0..run_len {
                    result.push(0xFE);
                    result.push(0xFE);
                    result.push(0x00);
                    result.push(0x01);
                }
            } else {
                // Literal bytes
                for _ in 0..run_len {
                    result.push(byte);
                }
            }

            i += run_len;
        }

        result
    }

    /// Decompress RLE-compressed data.
    pub fn rle_decompress(data: &[u8]) -> Result<Vec<u8>, String> {
        let mut result = Vec::new();
        let mut i = 0;

        while i < data.len() {
            if data[i] == 0xFE {
                if i + 3 >= data.len() {
                    return Err("truncated RLE sequence".into());
                }

                let byte = data[i + 1];
                let count = ((data[i + 2] as usize) << 8) | (data[i + 3] as usize);

                for _ in 0..count {
                    result.push(byte);
                }
                i += 4;
            } else {
                result.push(data[i]);
                i += 1;
            }
        }

        Ok(result)
    }

    /// Check if compression is worth it for given data size.
    pub fn should_compress(data_len: usize) -> bool {
        // Only compress data larger than 256 bytes
        data_len > 256
    }
}

// ---------------------------------------------------------------------------
// RemoteCacheClient
// ---------------------------------------------------------------------------

/// HTTP request/response for mock server interaction.
#[derive(Debug, Clone)]
pub struct HttpRequest {
    /// HTTP method (GET, PUT, HEAD).
    pub method: String,
    /// Request path (e.g., "/content/{hash}").
    pub path: String,
    /// Request body (for PUT).
    pub body: Option<Vec<u8>>,
    /// Authorization token.
    pub auth_token: Option<String>,
}

/// HTTP response from server.
#[derive(Debug, Clone)]
pub struct HttpResponse {
    /// Status code.
    pub status: u16,
    /// Response body.
    pub body: Vec<u8>,
}

impl HttpResponse {
    /// Create a success response.
    pub fn ok(body: Vec<u8>) -> Self {
        Self { status: 200, body }
    }

    /// Create a not found response.
    pub fn not_found() -> Self {
        Self {
            status: 404,
            body: Vec::new(),
        }
    }

    /// Create an unauthorized response.
    pub fn unauthorized() -> Self {
        Self {
            status: 401,
            body: Vec::new(),
        }
    }

    /// Create a server error response.
    pub fn server_error(msg: &str) -> Self {
        Self {
            status: 500,
            body: msg.as_bytes().to_vec(),
        }
    }
}

/// Client for accessing remote content cache via HTTP.
///
/// Implements a two-tier caching strategy:
/// 1. Check local FileBackend first
/// 2. On miss, fetch from remote server
/// 3. Cache fetched content locally
///
/// # Example
///
/// ```ignore
/// let local_backend = FileBackend::new("/tmp/local_cache")?;
/// let client = RemoteCacheClient::new(
///     "https://cache.example.com",
///     local_backend,
/// );
///
/// // Get content - checks local first, then remote
/// let data = client.get(&hash)?;
/// ```
pub struct RemoteCacheClient {
    /// Base URL for the remote cache server.
    base_url: String,
    /// Optional authentication token.
    auth_token: Option<String>,
    /// Local file backend for caching.
    local_backend: FileBackend,
    /// Whether to compress data for transfer.
    enable_compression: bool,
    /// Mock server for testing (None in production).
    mock_server: Option<RemoteCacheServer>,
}

impl RemoteCacheClient {
    /// Create a new remote cache client.
    pub fn new(base_url: impl Into<String>, local_backend: FileBackend) -> Self {
        Self {
            base_url: base_url.into(),
            auth_token: None,
            local_backend,
            enable_compression: true,
            mock_server: None,
        }
    }

    /// Create a client with authentication.
    pub fn with_auth(
        base_url: impl Into<String>,
        auth_token: impl Into<String>,
        local_backend: FileBackend,
    ) -> Self {
        Self {
            base_url: base_url.into(),
            auth_token: Some(auth_token.into()),
            local_backend,
            enable_compression: true,
            mock_server: None,
        }
    }

    /// Set whether to enable compression for transfers.
    pub fn set_compression(&mut self, enabled: bool) {
        self.enable_compression = enabled;
    }

    /// Set authentication token.
    pub fn set_auth_token(&mut self, token: Option<String>) {
        self.auth_token = token;
    }

    /// Connect to a mock server for testing.
    pub fn with_mock_server(mut self, server: RemoteCacheServer) -> Self {
        self.mock_server = Some(server);
        self
    }

    /// Get content by hash, checking local first then remote.
    ///
    /// On a remote hit, the content is cached locally for future access.
    pub fn get(&mut self, hash: &ContentHash) -> Result<Option<Vec<u8>>, RemoteCacheError> {
        // Check local first
        if let Some(data) = self.local_backend.get(hash)? {
            return Ok(Some(data));
        }

        // Try remote
        let response = self.http_get(&format!("/content/{}", hash))?;

        match response.status {
            200 => {
                // Decompress if needed
                let data = if self.enable_compression && !response.body.is_empty() {
                    compression::decompress(&response.body)
                        .map_err(RemoteCacheError::CompressionError)?
                } else {
                    response.body
                };

                // Store locally for future access
                self.local_backend.put(&data)?;

                Ok(Some(data))
            }
            404 => Ok(None),
            401 => Err(RemoteCacheError::AuthenticationFailed),
            _ => Err(RemoteCacheError::InvalidResponse(format!(
                "unexpected status: {}",
                response.status
            ))),
        }
    }

    /// Store content locally and optionally push to remote.
    ///
    /// If `push_to_remote` is true, also uploads to the remote server.
    pub fn put(
        &mut self,
        data: &[u8],
        push_to_remote: bool,
    ) -> Result<ContentHash, RemoteCacheError> {
        // Store locally first
        let hash = self.local_backend.put(data)?;

        // Optionally push to remote
        if push_to_remote {
            let body = if self.enable_compression && compression::should_compress(data.len()) {
                compression::compress(data)
            } else {
                data.to_vec()
            };

            let response = self.http_put(&format!("/content/{}", hash), body)?;

            match response.status {
                200 | 201 => {} // Success
                401 => return Err(RemoteCacheError::AuthenticationFailed),
                _ => {
                    return Err(RemoteCacheError::InvalidResponse(format!(
                        "push failed with status: {}",
                        response.status
                    )))
                }
            }
        }

        Ok(hash)
    }

    /// Check if content exists locally or remotely.
    pub fn has(&mut self, hash: &ContentHash) -> Result<bool, RemoteCacheError> {
        // Check local first
        if self.local_backend.has(hash) {
            return Ok(true);
        }

        // Check remote
        let response = self.http_head(&format!("/content/{}", hash))?;

        match response.status {
            200 => Ok(true),
            404 => Ok(false),
            401 => Err(RemoteCacheError::AuthenticationFailed),
            _ => Err(RemoteCacheError::InvalidResponse(format!(
                "has check failed with status: {}",
                response.status
            ))),
        }
    }

    /// Synchronize local cache with remote using DeltaSync protocol.
    ///
    /// Downloads missing content and uploads local-only content.
    pub fn sync(&mut self) -> Result<SyncStats, RemoteCacheError> {
        let protocol = DeltaSyncProtocol::new();

        // Get local items
        let local_hashes = self.local_backend.list()?;
        let local_items: Vec<SyncItem> = local_hashes
            .iter()
            .map(|h| {
                let size = self.local_backend.size(h).unwrap_or(Some(0)).unwrap_or(0);
                SyncItem::new(h.to_string(), h.clone(), size)
            })
            .collect();

        // Get remote items via API
        let remote_items = self.fetch_remote_items()?;

        // Compute what we need to download (remote has, local doesn't)
        let local_set: HashSet<_> = local_items.iter().map(|i| &i.hash).collect();
        let remote_set: HashSet<_> = remote_items.iter().map(|i| &i.hash).collect();

        let mut downloaded = 0;
        let mut uploaded = 0;
        let mut bytes_transferred = 0u64;

        // Download missing items
        for remote_item in &remote_items {
            if !local_set.contains(&remote_item.hash) {
                if let Some(data) = self.get(&remote_item.hash)? {
                    downloaded += 1;
                    bytes_transferred += data.len() as u64;
                }
            }
        }

        // Upload local-only items (if we have push access)
        for local_item in &local_items {
            if !remote_set.contains(&local_item.hash) {
                if let Some(data) = self.local_backend.get(&local_item.hash)? {
                    // Try to push - ignore errors for read-only remotes
                    if self.put(&data, true).is_ok() {
                        uploaded += 1;
                        bytes_transferred += data.len() as u64;
                    }
                }
            }
        }

        Ok(SyncStats {
            downloaded,
            uploaded,
            bytes_transferred,
            protocol_version: protocol.version(),
        })
    }

    /// Synchronize using DeltaSync protocol for efficient transfer.
    ///
    /// Uses delta compression to minimize bandwidth.
    pub fn delta_sync(
        &mut self,
        local_items: &[SyncItem],
        remote_items: &[SyncItem],
    ) -> Result<SyncBatch, RemoteCacheError> {
        let protocol = DeltaSyncProtocol::new();

        // Compute batch
        let batch = protocol.compute_batch(local_items, remote_items, |hash| {
            self.local_backend.get(hash).ok().flatten()
        })?;

        Ok(batch)
    }

    /// Fetch list of items available on remote.
    fn fetch_remote_items(&mut self) -> Result<Vec<SyncItem>, RemoteCacheError> {
        let response = self.http_get("/items")?;

        match response.status {
            200 => {
                // Parse response as JSON-like format
                // For simplicity, assume format: hash:size\n per line
                let text = String::from_utf8(response.body)
                    .map_err(|e| RemoteCacheError::InvalidResponse(e.to_string()))?;

                let items: Vec<SyncItem> = text
                    .lines()
                    .filter_map(|line| {
                        let parts: Vec<&str> = line.split(':').collect();
                        if parts.len() == 2 {
                            let hash = parts[0].parse::<ContentHash>().ok()?;
                            let size: u64 = parts[1].parse().ok()?;
                            Some(SyncItem::new(parts[0], hash, size))
                        } else {
                            None
                        }
                    })
                    .collect();

                Ok(items)
            }
            401 => Err(RemoteCacheError::AuthenticationFailed),
            _ => Err(RemoteCacheError::InvalidResponse(format!(
                "fetch items failed with status: {}",
                response.status
            ))),
        }
    }

    /// Perform HTTP GET request (via mock server or real HTTP).
    fn http_get(&mut self, path: &str) -> Result<HttpResponse, RemoteCacheError> {
        let request = HttpRequest {
            method: "GET".to_string(),
            path: path.to_string(),
            body: None,
            auth_token: self.auth_token.clone(),
        };

        self.execute_request(request)
    }

    /// Perform HTTP PUT request.
    fn http_put(&mut self, path: &str, body: Vec<u8>) -> Result<HttpResponse, RemoteCacheError> {
        let request = HttpRequest {
            method: "PUT".to_string(),
            path: path.to_string(),
            body: Some(body),
            auth_token: self.auth_token.clone(),
        };

        self.execute_request(request)
    }

    /// Perform HTTP HEAD request.
    fn http_head(&mut self, path: &str) -> Result<HttpResponse, RemoteCacheError> {
        let request = HttpRequest {
            method: "HEAD".to_string(),
            path: path.to_string(),
            body: None,
            auth_token: self.auth_token.clone(),
        };

        self.execute_request(request)
    }

    /// Execute HTTP request via mock server or stub.
    fn execute_request(&mut self, request: HttpRequest) -> Result<HttpResponse, RemoteCacheError> {
        if let Some(ref mut server) = self.mock_server {
            Ok(server.handle_request(&request))
        } else {
            // Real HTTP would go here - for now, return server unavailable
            Err(RemoteCacheError::ServerUnavailable)
        }
    }
}

// ---------------------------------------------------------------------------
// SyncStats
// ---------------------------------------------------------------------------

/// Statistics from a sync operation.
#[derive(Debug, Clone, Default)]
pub struct SyncStats {
    /// Number of items downloaded from remote.
    pub downloaded: usize,
    /// Number of items uploaded to remote.
    pub uploaded: usize,
    /// Total bytes transferred.
    pub bytes_transferred: u64,
    /// Protocol version used.
    pub protocol_version: u32,
}

// ---------------------------------------------------------------------------
// RemoteCacheServer
// ---------------------------------------------------------------------------

/// Mock HTTP server for remote cache.
///
/// Implements the server-side API for testing:
/// - GET /content/{hash} - retrieve content
/// - PUT /content/{hash} - store content
/// - HEAD /content/{hash} - check existence
/// - GET /items - list all items
///
/// # Example
///
/// ```ignore
/// let backend = FileBackend::new("/tmp/server_cache")?;
/// let mut server = RemoteCacheServer::new(backend);
///
/// // Add authentication token
/// server.add_auth_token("secret-token");
///
/// // Handle requests
/// let response = server.handle_request(&request);
/// ```
pub struct RemoteCacheServer {
    /// Backend storage.
    backend: FileBackend,
    /// Valid authentication tokens.
    auth_tokens: HashSet<String>,
    /// Whether authentication is required.
    require_auth: bool,
    /// Enable compression for responses.
    enable_compression: bool,
}

impl RemoteCacheServer {
    /// Create a new server with the given backend.
    pub fn new(backend: FileBackend) -> Self {
        Self {
            backend,
            auth_tokens: HashSet::new(),
            require_auth: false,
            enable_compression: true,
        }
    }

    /// Create a server that requires authentication.
    pub fn with_auth(backend: FileBackend) -> Self {
        Self {
            backend,
            auth_tokens: HashSet::new(),
            require_auth: true,
            enable_compression: true,
        }
    }

    /// Add a valid authentication token.
    pub fn add_auth_token(&mut self, token: impl Into<String>) {
        self.auth_tokens.insert(token.into());
    }

    /// Remove an authentication token.
    pub fn remove_auth_token(&mut self, token: &str) -> bool {
        self.auth_tokens.remove(token)
    }

    /// Set whether compression is enabled for responses.
    pub fn set_compression(&mut self, enabled: bool) {
        self.enable_compression = enabled;
    }

    /// Check if a token is valid.
    pub fn validate_token(&self, token: &Option<String>) -> bool {
        if !self.require_auth {
            return true;
        }

        match token {
            Some(t) => self.auth_tokens.contains(t),
            None => false,
        }
    }

    /// Handle an HTTP request and return a response.
    pub fn handle_request(&mut self, request: &HttpRequest) -> HttpResponse {
        // Check authentication
        if !self.validate_token(&request.auth_token) {
            return HttpResponse::unauthorized();
        }

        // Route request
        match request.method.as_str() {
            "GET" => self.handle_get(request),
            "PUT" => self.handle_put(request),
            "HEAD" => self.handle_head(request),
            _ => HttpResponse::server_error("method not allowed"),
        }
    }

    /// Handle GET request.
    fn handle_get(&self, request: &HttpRequest) -> HttpResponse {
        if request.path == "/items" {
            return self.handle_list_items();
        }

        // Parse hash from path: /content/{hash}
        let hash = match self.parse_content_path(&request.path) {
            Some(h) => h,
            None => return HttpResponse::not_found(),
        };

        match self.get(&hash) {
            Ok(Some(data)) => {
                let body = if self.enable_compression && compression::should_compress(data.len()) {
                    compression::compress(&data)
                } else {
                    data
                };
                HttpResponse::ok(body)
            }
            Ok(None) => HttpResponse::not_found(),
            Err(_) => HttpResponse::server_error("internal error"),
        }
    }

    /// Handle PUT request.
    fn handle_put(&mut self, request: &HttpRequest) -> HttpResponse {
        let body = match &request.body {
            Some(b) => b,
            None => return HttpResponse::server_error("missing body"),
        };

        // Decompress if needed
        let data = compression::decompress(body).unwrap_or_else(|_| body.clone());

        match self.put(&data) {
            Ok(_) => HttpResponse::ok(Vec::new()),
            Err(_) => HttpResponse::server_error("storage failed"),
        }
    }

    /// Handle HEAD request.
    fn handle_head(&self, request: &HttpRequest) -> HttpResponse {
        let hash = match self.parse_content_path(&request.path) {
            Some(h) => h,
            None => return HttpResponse::not_found(),
        };

        if self.has(&hash) {
            HttpResponse::ok(Vec::new())
        } else {
            HttpResponse::not_found()
        }
    }

    /// Handle list items request.
    fn handle_list_items(&self) -> HttpResponse {
        match self.backend.list() {
            Ok(hashes) => {
                let mut body = String::new();
                for hash in hashes {
                    let size = self.backend.size(&hash).ok().flatten().unwrap_or(0);
                    body.push_str(&format!("{}:{}\n", hash, size));
                }
                HttpResponse::ok(body.into_bytes())
            }
            Err(_) => HttpResponse::server_error("failed to list items"),
        }
    }

    /// Parse content path to extract hash.
    fn parse_content_path(&self, path: &str) -> Option<ContentHash> {
        let prefix = "/content/";
        if !path.starts_with(prefix) {
            return None;
        }

        path[prefix.len()..].parse().ok()
    }

    /// Get content from backend.
    pub fn get(&self, hash: &ContentHash) -> io::Result<Option<Vec<u8>>> {
        self.backend.get(hash)
    }

    /// Store content in backend.
    pub fn put(&self, data: &[u8]) -> io::Result<ContentHash> {
        self.backend.put(data)
    }

    /// Check if content exists.
    pub fn has(&self, hash: &ContentHash) -> bool {
        self.backend.has(hash)
    }

    /// Get the backend for direct access.
    pub fn backend(&self) -> &FileBackend {
        &self.backend
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    // Helper to create test backends
    fn create_temp_backends() -> (TempDir, TempDir, FileBackend, FileBackend) {
        let local_dir = TempDir::new().expect("create local temp dir");
        let remote_dir = TempDir::new().expect("create remote temp dir");
        let local_backend = FileBackend::new(local_dir.path()).expect("create local backend");
        let remote_backend = FileBackend::new(remote_dir.path()).expect("create remote backend");
        (local_dir, remote_dir, local_backend, remote_backend)
    }

    // ========================================================================
    // Test 1: Local hit returns immediately
    // ========================================================================
    #[test]
    fn test_local_hit_returns_immediately() {
        let (_local_dir, _remote_dir, local_backend, remote_backend) = create_temp_backends();

        // Store data locally
        let data = b"local cached data";
        let hash = local_backend.put(data).expect("put local");

        // Create client with empty remote
        let server = RemoteCacheServer::new(remote_backend);
        let mut client = RemoteCacheClient::new("http://test", local_backend).with_mock_server(server);

        // Get should return local data without hitting remote
        let result = client.get(&hash).expect("get");
        assert_eq!(result, Some(data.to_vec()));
    }

    // ========================================================================
    // Test 2: Remote fetch on local miss
    // ========================================================================
    #[test]
    fn test_remote_fetch_on_local_miss() {
        let (_local_dir, _remote_dir, local_backend, remote_backend) = create_temp_backends();

        // Store data on remote only
        let data = b"remote only data";
        let hash = remote_backend.put(data).expect("put remote");

        // Create client
        let server = RemoteCacheServer::new(remote_backend);
        let mut client = RemoteCacheClient::new("http://test", local_backend).with_mock_server(server);

        // Get should fetch from remote
        let result = client.get(&hash).expect("get");
        assert_eq!(result, Some(data.to_vec()));
    }

    // ========================================================================
    // Test 3: Fetched content stored locally
    // ========================================================================
    #[test]
    fn test_fetched_content_stored_locally() {
        let (local_dir, _remote_dir, local_backend, remote_backend) = create_temp_backends();

        // Store data on remote only
        let data = b"content to cache locally";
        let hash = remote_backend.put(data).expect("put remote");

        // Verify not in local
        assert!(!local_backend.has(&hash));

        // Create client and fetch
        let server = RemoteCacheServer::new(remote_backend);
        let mut client = RemoteCacheClient::new("http://test", local_backend).with_mock_server(server);
        let _ = client.get(&hash).expect("get");

        // Now check local backend directly
        let local_backend2 = FileBackend::open(local_dir.path()).expect("reopen local");
        assert!(local_backend2.has(&hash));
        assert_eq!(local_backend2.get(&hash).expect("get local"), Some(data.to_vec()));
    }

    // ========================================================================
    // Test 4: Auth token validation
    // ========================================================================
    #[test]
    fn test_auth_token_validation() {
        let (_local_dir, _remote_dir, local_backend, remote_backend) = create_temp_backends();

        // Store data on remote
        let data = b"protected data";
        let hash = remote_backend.put(data).expect("put remote");

        // Create server with auth required
        let mut server = RemoteCacheServer::with_auth(remote_backend);
        server.add_auth_token("valid-token");

        // Client with valid token
        let mut client = RemoteCacheClient::with_auth("http://test", "valid-token", local_backend)
            .with_mock_server(server);

        // Should succeed
        let result = client.get(&hash).expect("get with valid token");
        assert_eq!(result, Some(data.to_vec()));
    }

    // ========================================================================
    // Test 5: Compression roundtrip
    // ========================================================================
    #[test]
    fn test_compression_roundtrip() {
        // Test with repetitive data (compresses well)
        let data: Vec<u8> = vec![0xAA; 1000];
        let compressed = compression::compress(&data);
        let decompressed = compression::decompress(&compressed).expect("decompress");
        assert_eq!(data, decompressed);

        // Compressed should be smaller
        assert!(compressed.len() < data.len());
    }

    // ========================================================================
    // Test 6: Has() checks both local and remote
    // ========================================================================
    #[test]
    fn test_has_checks_both() {
        let (_local_dir, _remote_dir, local_backend, remote_backend) = create_temp_backends();

        // Store different data on local and remote
        let local_data = b"local data";
        let remote_data = b"remote data";
        let local_hash = local_backend.put(local_data).expect("put local");
        let remote_hash = remote_backend.put(remote_data).expect("put remote");

        let server = RemoteCacheServer::new(remote_backend);
        let mut client = RemoteCacheClient::new("http://test", local_backend).with_mock_server(server);

        // Both should exist
        assert!(client.has(&local_hash).expect("has local"));
        assert!(client.has(&remote_hash).expect("has remote"));

        // Non-existent should not exist
        let fake_hash = ContentHash::from_bytes(b"nonexistent");
        assert!(!client.has(&fake_hash).expect("has fake"));
    }

    // ========================================================================
    // Test 7: Invalid token rejected
    // ========================================================================
    #[test]
    fn test_invalid_token_rejected() {
        let (_local_dir, _remote_dir, local_backend, remote_backend) = create_temp_backends();

        // Store data on remote
        let data = b"protected data";
        let hash = remote_backend.put(data).expect("put remote");

        // Create server with auth required
        let mut server = RemoteCacheServer::with_auth(remote_backend);
        server.add_auth_token("valid-token");

        // Client with invalid token
        let mut client = RemoteCacheClient::with_auth("http://test", "wrong-token", local_backend)
            .with_mock_server(server);

        // Should fail
        let result = client.get(&hash);
        assert!(matches!(result, Err(RemoteCacheError::AuthenticationFailed)));
    }

    // ========================================================================
    // Test 8: Network error handling (mock unavailable)
    // ========================================================================
    #[test]
    fn test_network_error_handling() {
        let (_local_dir, local_backend) = {
            let dir = TempDir::new().expect("create temp dir");
            let backend = FileBackend::new(dir.path()).expect("create backend");
            (dir, backend)
        };

        // Client without mock server (simulates network unavailable)
        let mut client = RemoteCacheClient::new("http://unreachable", local_backend);

        let hash = ContentHash::from_bytes(b"missing");
        let result = client.get(&hash);
        assert!(matches!(result, Err(RemoteCacheError::ServerUnavailable)));
    }

    // ========================================================================
    // Test 9: DeltaSync integration
    // ========================================================================
    #[test]
    fn test_delta_sync_integration() {
        let (_local_dir, _remote_dir, local_backend, remote_backend) = create_temp_backends();

        // Create items - put both in local so delta_sync can compute the batch
        // This tests the delta computation logic
        let data1 = b"item one";
        let data2 = b"item two";
        let hash1 = local_backend.put(data1).expect("put 1");
        let hash2 = local_backend.put(data2).expect("put 2"); // Also in local for delta computation
        remote_backend.put(data2).expect("put 2 remote"); // And on remote

        // Local has item1, remote has item2 - delta should compute operations
        let local_items = vec![SyncItem::new("item1", hash1.clone(), data1.len() as u64)];
        let remote_items = vec![SyncItem::new("item2", hash2.clone(), data2.len() as u64)];

        let server = RemoteCacheServer::new(remote_backend);
        let mut client = RemoteCacheClient::new("http://test", local_backend).with_mock_server(server);

        let batch = client.delta_sync(&local_items, &remote_items).expect("delta sync");

        // Should have operations (1 upsert for item2, 1 remove for item1)
        assert!(!batch.is_empty());
        assert_eq!(batch.operations.len(), 2);
    }

    // ========================================================================
    // Test 10: Put with remote push
    // ========================================================================
    #[test]
    fn test_put_with_remote_push() {
        let (_local_dir, _remote_dir, local_backend, remote_backend) = create_temp_backends();

        let server = RemoteCacheServer::new(remote_backend);
        let mut client = RemoteCacheClient::new("http://test", local_backend).with_mock_server(server);

        let data = b"new content";
        let hash = client.put(data, true).expect("put with push");

        // Should be in both local and remote (via server)
        assert!(client.has(&hash).expect("has"));
    }

    // ========================================================================
    // Test 11: Content not found returns None
    // ========================================================================
    #[test]
    fn test_content_not_found() {
        let (_local_dir, _remote_dir, local_backend, remote_backend) = create_temp_backends();

        let server = RemoteCacheServer::new(remote_backend);
        let mut client = RemoteCacheClient::new("http://test", local_backend).with_mock_server(server);

        let fake_hash = ContentHash::from_bytes(b"nonexistent");
        let result = client.get(&fake_hash).expect("get");
        assert_eq!(result, None);
    }

    // ========================================================================
    // Test 12: RLE compression edge cases
    // ========================================================================
    #[test]
    fn test_rle_compression_edge_cases() {
        // Empty data
        let empty: Vec<u8> = vec![];
        assert_eq!(compression::rle_compress(&empty), empty);
        assert_eq!(compression::rle_decompress(&empty).expect("decompress"), empty);

        // Single byte
        let single = vec![0x42];
        let compressed = compression::rle_compress(&single);
        assert_eq!(compression::rle_decompress(&compressed).expect("decompress"), single);

        // Three bytes (below threshold)
        let three = vec![0xAB; 3];
        let compressed = compression::rle_compress(&three);
        assert_eq!(compression::rle_decompress(&compressed).expect("decompress"), three);

        // Exactly four bytes (at threshold)
        let four = vec![0xCD; 4];
        let compressed = compression::rle_compress(&four);
        assert_eq!(compression::rle_decompress(&compressed).expect("decompress"), four);

        // Mixed data
        let mixed = vec![0x00, 0x00, 0x00, 0x00, 0x11, 0x22, 0x33, 0x33, 0x33, 0x33];
        let compressed = compression::rle_compress(&mixed);
        assert_eq!(compression::rle_decompress(&compressed).expect("decompress"), mixed);
    }

    // ========================================================================
    // Test 13: Escape sequence handling (0xFE byte)
    // ========================================================================
    #[test]
    fn test_rle_escape_sequence() {
        // Data containing the escape byte
        let data = vec![0xFE, 0xFE, 0xFE];
        let compressed = compression::rle_compress(&data);
        let decompressed = compression::rle_decompress(&compressed).expect("decompress");
        assert_eq!(data, decompressed);

        // Mixed with escape byte
        let mixed = vec![0x00, 0xFE, 0x00, 0xFE, 0xFE];
        let compressed = compression::rle_compress(&mixed);
        let decompressed = compression::rle_decompress(&compressed).expect("decompress");
        assert_eq!(mixed, decompressed);
    }

    // ========================================================================
    // Test 14: Server token management
    // ========================================================================
    #[test]
    fn test_server_token_management() {
        let (_remote_dir, remote_backend) = {
            let dir = TempDir::new().expect("create temp dir");
            let backend = FileBackend::new(dir.path()).expect("create backend");
            (dir, backend)
        };

        let mut server = RemoteCacheServer::with_auth(remote_backend);

        // Add tokens
        server.add_auth_token("token1");
        server.add_auth_token("token2");

        // Validate
        assert!(server.validate_token(&Some("token1".to_string())));
        assert!(server.validate_token(&Some("token2".to_string())));
        assert!(!server.validate_token(&Some("invalid".to_string())));
        assert!(!server.validate_token(&None));

        // Remove token
        assert!(server.remove_auth_token("token1"));
        assert!(!server.validate_token(&Some("token1".to_string())));
        assert!(server.validate_token(&Some("token2".to_string())));
    }

    // ========================================================================
    // Test 15: Server handles all HTTP methods
    // ========================================================================
    #[test]
    fn test_server_http_methods() {
        let (_remote_dir, remote_backend) = {
            let dir = TempDir::new().expect("create temp dir");
            let backend = FileBackend::new(dir.path()).expect("create backend");
            (dir, backend)
        };

        let mut server = RemoteCacheServer::new(remote_backend);

        // PUT content
        let data = b"test content";
        let put_request = HttpRequest {
            method: "PUT".to_string(),
            path: "/content/test".to_string(),
            body: Some(data.to_vec()),
            auth_token: None,
        };
        let put_response = server.handle_request(&put_request);
        assert_eq!(put_response.status, 200);

        // Verify content was stored
        let hash = ContentHash::from_bytes(data);

        // HEAD request
        let head_request = HttpRequest {
            method: "HEAD".to_string(),
            path: format!("/content/{}", hash),
            body: None,
            auth_token: None,
        };
        let head_response = server.handle_request(&head_request);
        assert_eq!(head_response.status, 200);

        // GET request
        let get_request = HttpRequest {
            method: "GET".to_string(),
            path: format!("/content/{}", hash),
            body: None,
            auth_token: None,
        };
        let get_response = server.handle_request(&get_request);
        assert_eq!(get_response.status, 200);
    }

    // ========================================================================
    // Test 16: Sync stats tracking
    // ========================================================================
    #[test]
    fn test_sync_stats() {
        let (_local_dir, _remote_dir, local_backend, remote_backend) = create_temp_backends();

        // Add items to remote only
        let data1 = b"sync item 1";
        let data2 = b"sync item 2";
        remote_backend.put(data1).expect("put 1");
        remote_backend.put(data2).expect("put 2");

        let server = RemoteCacheServer::new(remote_backend);
        let mut client = RemoteCacheClient::new("http://test", local_backend).with_mock_server(server);

        let stats = client.sync().expect("sync");

        // Should have downloaded 2 items
        assert_eq!(stats.downloaded, 2);
        assert!(stats.bytes_transferred > 0);
        assert_eq!(stats.protocol_version, 1);
    }

    // ========================================================================
    // Test 17: Compression disabled
    // ========================================================================
    #[test]
    fn test_compression_disabled() {
        let (_local_dir, _remote_dir, local_backend, remote_backend) = create_temp_backends();

        // Store data on remote
        let data = vec![0xAA; 500]; // Repetitive data that would compress well
        let hash = remote_backend.put(&data).expect("put remote");

        // Server with compression disabled
        let mut server = RemoteCacheServer::new(remote_backend);
        server.set_compression(false);

        // Client with compression disabled
        let mut client = RemoteCacheClient::new("http://test", local_backend).with_mock_server(server);
        client.set_compression(false);

        let result = client.get(&hash).expect("get");
        assert_eq!(result, Some(data));
    }

    // ========================================================================
    // Test 18: Large run compression
    // ========================================================================
    #[test]
    fn test_large_run_compression() {
        // Max run length is 65535
        let data: Vec<u8> = vec![0x77; 70000]; // Over max run
        let compressed = compression::rle_compress(&data);
        let decompressed = compression::rle_decompress(&compressed).expect("decompress");
        assert_eq!(data, decompressed);

        // Should be significantly smaller
        assert!(compressed.len() < data.len() / 10);
    }

    // ========================================================================
    // Test 19: Server list items
    // ========================================================================
    #[test]
    fn test_server_list_items() {
        let (_remote_dir, remote_backend) = {
            let dir = TempDir::new().expect("create temp dir");
            let backend = FileBackend::new(dir.path()).expect("create backend");
            (dir, backend)
        };

        // Add items
        remote_backend.put(b"item a").expect("put a");
        remote_backend.put(b"item b").expect("put b");
        remote_backend.put(b"item c").expect("put c");

        let mut server = RemoteCacheServer::new(remote_backend);

        let request = HttpRequest {
            method: "GET".to_string(),
            path: "/items".to_string(),
            body: None,
            auth_token: None,
        };

        let response = server.handle_request(&request);
        assert_eq!(response.status, 200);

        let body = String::from_utf8(response.body).expect("parse body");
        let lines: Vec<&str> = body.lines().collect();
        assert_eq!(lines.len(), 3);
    }

    // ========================================================================
    // Test 20: Error conversion
    // ========================================================================
    #[test]
    fn test_error_conversion() {
        // io::Error conversion
        let io_err = io::Error::new(io::ErrorKind::NotFound, "file not found");
        let cache_err: RemoteCacheError = io_err.into();
        assert!(matches!(cache_err, RemoteCacheError::IoError(_)));

        // SyncError conversion
        let sync_err = SyncError::CheckpointMismatch;
        let cache_err: RemoteCacheError = sync_err.into();
        assert!(matches!(cache_err, RemoteCacheError::SyncError(_)));
    }

    // ========================================================================
    // Test 21: No auth required server accepts all
    // ========================================================================
    #[test]
    fn test_no_auth_required() {
        let (_remote_dir, remote_backend) = {
            let dir = TempDir::new().expect("create temp dir");
            let backend = FileBackend::new(dir.path()).expect("create backend");
            (dir, backend)
        };

        let data = b"public data";
        remote_backend.put(data).expect("put");
        let hash = ContentHash::from_bytes(data);

        // Server without auth requirement
        let mut server = RemoteCacheServer::new(remote_backend);

        // No token provided
        let request = HttpRequest {
            method: "GET".to_string(),
            path: format!("/content/{}", hash),
            body: None,
            auth_token: None,
        };

        let response = server.handle_request(&request);
        assert_eq!(response.status, 200);
    }
}
