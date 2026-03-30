fn main() {
    // Link to pthread-enabled OpenBLAS for multi-threaded BLAS
    println!("cargo:rustc-link-lib=openblasp");
    println!("cargo:rustc-link-search=native=/lib64");
}
