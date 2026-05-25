//! Asset loading subsystem with a background thread for glTF mesh loading.
//!
//! Provides an [`AssetLoader`] that dispatches load requests to a worker thread
//! via a channel.  The actual glTF parsing is a placeholder -- in a full
//! implementation this would use the `gltf` and `image` crates.
//!
//! # Example (placeholder)
//!
//! ```ignore
//! use renderer_backend::asset_loader::{AssetLoader, MeshData};
//!
//! let loader = AssetLoader::new();
//! let meshes = loader.load_gltf("models/teapot.gltf").unwrap();
//! ```

use std::sync::mpsc::{self, Sender};
use std::thread::{self, JoinHandle};

// ---------------------------------------------------------------------------
// MeshData
// ---------------------------------------------------------------------------

/// Raw mesh data produced by the asset loader.
///
/// Contains vertex and index data as flat byte vectors, ready for GPU upload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeshData {
    /// Interleaved vertex attribute bytes.
    pub vertex_data: Vec<u8>,
    /// Index buffer bytes (typically u16 or u32 indices).
    pub index_data: Vec<u8>,
    /// Number of vertices in this mesh.
    pub vertex_count: u32,
    /// Number of indices in this mesh.
    pub index_count: u32,
}

impl MeshData {
    /// Create a new `MeshData` from its raw components.
    pub const fn new(
        vertex_data: Vec<u8>,
        index_data: Vec<u8>,
        vertex_count: u32,
        index_count: u32,
    ) -> Self {
        Self {
            vertex_data,
            index_data,
            vertex_count,
            index_count,
        }
    }
}

// ---------------------------------------------------------------------------
// AssetError
// ---------------------------------------------------------------------------

/// Errors that can occur during asset loading.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AssetError {
    /// The requested asset file was not found or the path was empty.
    NotFound,
    /// The asset data could not be parsed.
    ParseError,
    /// The file format is not supported by this loader.
    UnsupportedFormat,
}

impl std::fmt::Display for AssetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AssetError::NotFound => write!(f, "asset not found"),
            AssetError::ParseError => write!(f, "asset parse error"),
            AssetError::UnsupportedFormat => write!(f, "unsupported asset format"),
        }
    }
}

impl std::error::Error for AssetError {}

// ---------------------------------------------------------------------------
// Internal request type
// ---------------------------------------------------------------------------

/// A load request sent from the public API to the background worker thread.
struct LoadRequest {
    /// Path to the asset file.
    path: String,
    /// Channel to send the result back on.
    sender: Sender<Result<Vec<MeshData>, AssetError>>,
}

// ---------------------------------------------------------------------------
// AssetLoader
// ---------------------------------------------------------------------------

/// Asynchronous asset loader with a background worker thread.
///
/// Spawns a single background thread that processes load requests received
/// via an `mpsc` channel.  Each `load_gltf` call sends a request and blocks
/// until the worker responds.
///
/// The loader is dropped cleanly: the channel is closed, the worker thread
/// exits its event loop, and `Drop` joins the thread.
pub struct AssetLoader {
    /// Sender end of the request channel.
    tx: Sender<LoadRequest>,
    /// Worker thread handle; `None` after being taken in `Drop`.
    handle: Option<JoinHandle<()>>,
}

impl AssetLoader {
    /// Create a new `AssetLoader`, spawning the background worker thread.
    pub fn new() -> Self {
        let (tx, rx) = mpsc::channel::<LoadRequest>();

        let handle = thread::Builder::new()
            .name("asset-loader".into())
            .spawn(move || {
                // Process requests until the channel is closed (sender dropped).
                for request in rx {
                    let result = Self::process_load(&request.path);
                    // Ignore send errors -- the receiver may have been dropped.
                    let _ = request.sender.send(result);
                }
            })
            .expect("failed to spawn asset loader thread");

        Self {
            tx,
            handle: Some(handle),
        }
    }

    /// Load a glTF file and return its meshes.
    ///
    /// Dispatches the load to the background thread and blocks until the
    /// result is available.  Currently a placeholder that returns an empty
    /// `Vec` for any valid glTF/glB path.
    ///
    /// # Errors
    ///
    /// Returns [`AssetError::NotFound`] if the path is empty or the channel
    /// fails.  Returns [`AssetError::UnsupportedFormat`] if the file extension
    /// is not `.gltf` or `.glb`.
    pub fn load_gltf(&self, path: &str) -> Result<Vec<MeshData>, AssetError> {
        let (tx, rx) = mpsc::channel();
        self.tx
            .send(LoadRequest {
                path: path.to_string(),
                sender: tx,
            })
            .map_err(|_| AssetError::NotFound)?;
        rx.recv().map_err(|_| AssetError::ParseError)?
    }

    /// Internal: process a single load request on the worker thread.
    ///
    /// Currently a placeholder that validates the file extension and returns
    /// an empty `Vec`.  A real implementation would use the `gltf` crate.
    fn process_load(path: &str) -> Result<Vec<MeshData>, AssetError> {
        if path.is_empty() {
            return Err(AssetError::NotFound);
        }
        let ext = path.rsplit('.').next().unwrap_or("");
        if ext != "gltf" && ext != "glb" {
            return Err(AssetError::UnsupportedFormat);
        }
        // Placeholder: return empty mesh list.
        // A real implementation would decode the glTF buffer views,
        // extract vertex positions/normals/UVs, and flatten primitives
        // into the MeshTable's flat indexing scheme.
        Ok(Vec::new())
    }
}

impl Default for AssetLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for AssetLoader {
    fn drop(&mut self) {
        // Close the channel by replacing `tx` with a dummy sender, signaling
        // the worker thread to exit its `for request in rx` loop.  We must
        // explicitly drop the original sender *before* joining because `tx`
        // is a field of `self` and would otherwise survive until after `drop`
        // returns, keeping the channel open and deadlocking both sides.
        let _ = std::mem::replace(&mut self.tx, mpsc::channel().0);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── MeshData ────────────────────────────────────────────────────────

    #[test]
    fn test_mesh_data_creation() {
        let md = MeshData::new(vec![1, 2, 3], vec![0, 1, 2], 1, 3);
        assert_eq!(md.vertex_data, vec![1, 2, 3]);
        assert_eq!(md.index_data, vec![0, 1, 2]);
        assert_eq!(md.vertex_count, 1);
        assert_eq!(md.index_count, 3);
    }

    #[test]
    fn test_mesh_data_empty() {
        let md = MeshData::new(vec![], vec![], 0, 0);
        assert!(md.vertex_data.is_empty());
        assert!(md.index_data.is_empty());
        assert_eq!(md.vertex_count, 0);
        assert_eq!(md.index_count, 0);
    }

    #[test]
    fn test_mesh_data_clone() {
        let md = MeshData::new(vec![10, 20], vec![30, 40], 2, 2);
        let cloned = md.clone();
        assert_eq!(cloned.vertex_data, md.vertex_data);
        assert_eq!(cloned.index_data, md.index_data);
        assert_eq!(cloned.vertex_count, md.vertex_count);
        assert_eq!(cloned.index_count, md.index_count);
    }

    // ── AssetError ──────────────────────────────────────────────────────

    #[test]
    fn test_asset_error_display() {
        assert_eq!(format!("{}", AssetError::NotFound), "asset not found");
        assert_eq!(format!("{}", AssetError::ParseError), "asset parse error");
        assert_eq!(
            format!("{}", AssetError::UnsupportedFormat),
            "unsupported asset format"
        );
    }

    #[test]
    fn test_asset_error_clone_eq() {
        assert_eq!(AssetError::NotFound, AssetError::NotFound);
        assert_ne!(AssetError::NotFound, AssetError::ParseError);
    }

    #[test]
    fn test_asset_error_is_error() {
        use std::error::Error;
        assert!(AssetError::NotFound.source().is_none());
        assert!(AssetError::ParseError.source().is_none());
        assert!(AssetError::UnsupportedFormat.source().is_none());
    }

    // ── AssetLoader (process_load internals) ────────────────────────────

    #[test]
    fn test_process_load_empty_path() {
        let result = AssetLoader::process_load("");
        assert_eq!(result, Err(AssetError::NotFound));
    }

    #[test]
    fn test_process_load_unsupported_extension() {
        let result = AssetLoader::process_load("model.obj");
        assert_eq!(result, Err(AssetError::UnsupportedFormat));
    }

    #[test]
    fn test_process_load_no_extension() {
        let result = AssetLoader::process_load("teapot");
        assert_eq!(result, Err(AssetError::UnsupportedFormat));
    }

    #[test]
    fn test_process_load_gltf_returns_empty() {
        let result = AssetLoader::process_load("model.gltf");
        assert_eq!(result, Ok(Vec::new()));
    }

    #[test]
    fn test_process_load_glb_returns_empty() {
        let result = AssetLoader::process_load("model.glb");
        assert_eq!(result, Ok(Vec::new()));
    }

    #[test]
    fn test_process_load_path_with_directories() {
        let result = AssetLoader::process_load("/assets/models/teapot.gltf");
        assert_eq!(result, Ok(Vec::new()));
    }

    #[test]
    fn test_process_load_case_sensitive_extension() {
        // The placeholder uses simple extension matching; .GLTF should fail.
        let result = AssetLoader::process_load("model.GLTF");
        assert_eq!(result, Err(AssetError::UnsupportedFormat));
    }

    // ── AssetLoader (live thread) ───────────────────────────────────────

    #[test]
    fn test_asset_loader_creation() {
        let loader = AssetLoader::new();
        // Ensure the thread is alive.
        assert!(loader.tx.send(LoadRequest {
            path: "test.gltf".into(),
            sender: mpsc::channel().0,
        }).is_ok());
    }

    #[test]
    fn test_load_gltf_valid_path() {
        let loader = AssetLoader::new();
        let result = loader.load_gltf("models/teapot.gltf");
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_load_glb_valid_path() {
        let loader = AssetLoader::new();
        let result = loader.load_gltf("models/scene.glb");
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_load_gltf_empty_path() {
        let loader = AssetLoader::new();
        let result = loader.load_gltf("");
        assert_eq!(result, Err(AssetError::NotFound));
    }

    #[test]
    fn test_load_gltf_unsupported_extension() {
        let loader = AssetLoader::new();
        let result = loader.load_gltf("model.obj");
        assert_eq!(result, Err(AssetError::UnsupportedFormat));
    }

    #[test]
    fn test_load_gltf_multiple_calls() {
        let loader = AssetLoader::new();
        let r1 = loader.load_gltf("a.gltf");
        let r2 = loader.load_gltf("b.glb");
        let r3 = loader.load_gltf("");
        assert!(r1.is_ok());
        assert!(r2.is_ok());
        assert_eq!(r3, Err(AssetError::NotFound));
    }

    #[test]
    fn test_asset_loader_default() {
        let loader = AssetLoader::default();
        let result = loader.load_gltf("test.gltf");
        assert!(result.is_ok());
    }
}
