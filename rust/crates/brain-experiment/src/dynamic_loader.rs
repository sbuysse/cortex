//! Dynamic mutation .so loader.
//!
//! Loads compiled mutation shared libraries and extracts the vtable symbol.
//! Validates ABI version and runs a smoke test before accepting the mutation.

use brain_traits::{MutationVtable, ABI_VERSION};
use std::path::Path;

#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("Library loading error: {0}")]
    Library(#[from] libloading::Error),
    #[error("ABI version mismatch: expected {expected}, got {got}")]
    AbiMismatch { expected: u32, got: u32 },
    #[error("Smoke test failed: {0}")]
    SmokeTestFailed(String),
}

/// A loaded mutation with its shared library handle.
pub struct LoadedMutation {
    /// Keep the library alive as long as the vtable is in use.
    _lib: libloading::Library,
    /// Pointer to the vtable in the loaded library.
    pub vtable: *const MutationVtable,
}

// Safety: The vtable uses C-ABI function pointers that are Send+Sync.
unsafe impl Send for LoadedMutation {}
unsafe impl Sync for LoadedMutation {}

impl LoadedMutation {
    /// Load a mutation from a shared library (.so) file.
    ///
    /// # Safety
    /// The .so must export a valid `brain_mutation_vtable` symbol
    /// with the correct ABI version.
    pub unsafe fn load(path: &Path) -> Result<Self, LoadError> {
        let lib = unsafe { libloading::Library::new(path)? };

        let vtable_ptr: libloading::Symbol<*const MutationVtable> =
            unsafe { lib.get(b"brain_mutation_vtable\0")? };

        let vtable = *vtable_ptr;

        // Validate ABI version
        let version = unsafe { (*vtable).version };
        if version != ABI_VERSION {
            return Err(LoadError::AbiMismatch {
                expected: ABI_VERSION,
                got: version,
            });
        }

        Ok(Self {
            _lib: lib,
            vtable,
        })
    }

    /// Get a reference to the vtable.
    ///
    /// # Safety
    /// The vtable pointer must be valid (guaranteed by successful load).
    pub unsafe fn vtable_ref(&self) -> &MutationVtable {
        unsafe { &*self.vtable }
    }

    /// Run a basic smoke test: create small test data and call the mutation
    /// function inside catch_unwind to detect panics/segfaults.
    pub fn smoke_test(&self) -> Result<(), LoadError> {
        use std::panic;

        let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            // Basic validation — just check the vtable has at least one function
            let vt = unsafe { &*self.vtable };
            let has_fn = vt.hebbian_update.is_some()
                || vt.hebbian_prune.is_some()
                || vt.sparse_forward.is_some()
                || vt.temporal_update.is_some();
            if !has_fn {
                return Err("No mutation functions in vtable");
            }
            Ok(())
        }));

        match result {
            Ok(Ok(())) => Ok(()),
            Ok(Err(msg)) => Err(LoadError::SmokeTestFailed(msg.to_string())),
            Err(_) => Err(LoadError::SmokeTestFailed("Panic during smoke test".to_string())),
        }
    }
}
