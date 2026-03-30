//! C-ABI stable trait definitions for dynamic .so loading of mutations.
//!
//! Each compiled mutation exports a `brain_mutation_vtable` symbol
//! that the experiment runner loads at runtime.

use std::os::raw::c_uint;

/// ABI version — increment when vtable layout changes.
pub const ABI_VERSION: u32 = 1;

/// A C-compatible view into a 2D matrix (row-major).
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct MatrixView {
    pub data: *mut f32,
    pub rows: usize,
    pub cols: usize,
    /// Row stride in number of f32 elements (typically == cols for contiguous).
    pub stride: usize,
}

/// A C-compatible view into a 1D vector.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct VectorView {
    pub data: *mut f32,
    pub len: usize,
}

/// Input batch pair for Hebbian update.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct BatchPair {
    pub x_a: MatrixView,
    pub x_b: MatrixView,
    pub batch_size: usize,
}

/// Result from a Hebbian update mutation.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct HebbianUpdateResult {
    /// Whether the update succeeded (0 = success, nonzero = error).
    pub status: c_uint,
}

/// Sparse projection input/output.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct SparseForwardResult {
    pub status: c_uint,
}

/// Full Hebbian state passed to mutation functions.
#[repr(C)]
#[derive(Debug)]
pub struct HebbianState {
    pub m: MatrixView,
    pub consolidation: MatrixView,
    pub effective_lr: MatrixView,
    pub delta_buf: MatrixView,
    pub pi_v: VectorView,
    pub lr: f32,
    pub decay_rate: f32,
    pub max_norm: f32,
    pub update_count: u64,
    pub d: usize,
}

/// Which component this mutation targets.
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum MutationKind {
    HebbianUpdate = 0,
    HebbianPrune = 1,
    SparseForward = 2,
    TemporalUpdate = 3,
    AssociationForward = 4,
}

/// Function pointer types for each mutation target.
pub type HebbianUpdateFn =
    unsafe extern "C" fn(state: *mut HebbianState, batch: *const BatchPair) -> HebbianUpdateResult;

pub type HebbianPruneFn =
    unsafe extern "C" fn(m: *mut MatrixView, threshold: f32) -> c_uint;

pub type SparseForwardFn = unsafe extern "C" fn(
    input: *const MatrixView,
    weights: *const MatrixView,
    bias: *const VectorView,
    output: *mut MatrixView,
    k: usize,
) -> SparseForwardResult;

pub type TemporalUpdateFn = unsafe extern "C" fn(
    trace: *mut VectorView,
    current: *const MatrixView,
    decay: f32,
) -> c_uint;

/// The vtable that each mutation .so exports as `brain_mutation_vtable`.
#[repr(C)]
pub struct MutationVtable {
    pub version: u32,
    pub kind: MutationKind,
    pub hebbian_update: Option<HebbianUpdateFn>,
    pub hebbian_prune: Option<HebbianPruneFn>,
    pub sparse_forward: Option<SparseForwardFn>,
    pub temporal_update: Option<TemporalUpdateFn>,
}

// Safety helpers for constructing MatrixView from ndarray

impl MatrixView {
    /// Create a MatrixView from a raw pointer and dimensions.
    ///
    /// # Safety
    /// Caller must ensure the pointer is valid for `rows * stride` f32 elements.
    pub unsafe fn from_raw(data: *mut f32, rows: usize, cols: usize, stride: usize) -> Self {
        Self {
            data,
            rows,
            cols,
            stride,
        }
    }

    /// Number of elements in the matrix.
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl VectorView {
    /// # Safety
    /// Caller must ensure the pointer is valid for `len` f32 elements.
    pub unsafe fn from_raw(data: *mut f32, len: usize) -> Self {
        Self { data, len }
    }
}

/// Helper to create a MatrixView from an ndarray Array2.
///
/// # Safety
/// The Array2 must outlive the MatrixView.
pub unsafe fn matrix_view_from_ndarray(arr: &mut ndarray::Array2<f32>) -> MatrixView {
    let (rows, cols) = arr.dim();
    let stride = arr.strides()[0] as usize;
    MatrixView {
        data: arr.as_mut_ptr(),
        rows,
        cols,
        stride,
    }
}

/// Helper to create a VectorView from an ndarray Array1.
///
/// # Safety
/// The Array1 must outlive the VectorView.
pub unsafe fn vector_view_from_ndarray(arr: &mut ndarray::Array1<f32>) -> VectorView {
    VectorView {
        data: arr.as_mut_ptr(),
        len: arr.len(),
    }
}
