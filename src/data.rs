// src/data.rs
// MNIST + text loading.

use crate::tensor::Matrix;
use std::fs;

/// MNIST images: N x 784 (normalized to 0..1)
pub fn load_images(path: &str) -> Matrix {
    let bytes = fs::read(path).expect("could not read image file");
    let magic = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    assert_eq!(magic, 0x00000803, "wrong magic (images)");
    let n = u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    let rows = u32::from_be_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
    let cols = u32::from_be_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) as usize;
    assert_eq!(rows, 28);
    assert_eq!(cols, 28);
    let dim = rows * cols;

    let mut data = Vec::with_capacity(n * dim);
    for i in 0..n * dim {
        let v = bytes[16 + i] as f32 / 255.0;
        data.push(v);
    }
    Matrix::from_vec(n, dim, data)
}

/// MNIST etiketleri: N × 10 one-hot
pub fn load_labels_onehot(path: &str) -> Matrix {
    let bytes = fs::read(path).expect("could not read label file");
    let magic = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    assert_eq!(magic, 0x00000801, "wrong magic (labels)");
    let n = u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;

    let mut data = vec![0.0f32; n * 10];
    for i in 0..n {
        let label = bytes[8 + i] as usize;
        assert!(label < 10);
        data[i * 10 + label] = 1.0;
    }
    Matrix::from_vec(n, 10, data)
}

/// Read labels as plain integers (for accuracy)
pub fn load_labels_raw(path: &str) -> Vec<usize> {
    let bytes = fs::read(path).expect("could not read label file");
    let n = u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    (0..n).map(|i| bytes[8 + i] as usize).collect()
}

/// Slice a row range out of a Matrix (for mini-batches)
pub fn slice_rows(m: &Matrix, start: usize, end: usize) -> Matrix {
    assert!(end <= m.rows);
    let cols = m.cols;
    let mut out = Vec::with_capacity((end - start) * cols);
    for i in start..end {
        for j in 0..cols {
            out.push(m.get(i, j));
        }
    }
    Matrix::from_vec(end - start, cols, out)
}

// ============================================================
// TEXT LOADING (for the transformer)
// ============================================================

/// Load a text file.
pub fn load_text(path: &str) -> String {
    fs::read_to_string(path).expect("could not read text file")
}

/// Split text into fixed-length chunks (for transformer training).
/// Returns (input, target) pairs. Every pair is `seq_len` long.
/// target = input shifted by 1.
pub fn make_sequences(
    ids: &[usize],
    seq_len: usize,
    stride: usize,
) -> Vec<(Vec<usize>, Vec<usize>)> {
    let mut out = Vec::new();
    let mut start = 0;
    while start + seq_len < ids.len() {
        let input = ids[start..start + seq_len].to_vec();
        let target = ids[start + 1..start + seq_len + 1].to_vec();
        out.push((input, target));
        start += stride;
    }
    out
}
