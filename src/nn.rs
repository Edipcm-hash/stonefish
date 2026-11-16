// src/nn.rs
use crate::autograd::Tensor;
use crate::tensor::Matrix;
use std::fs::File;
use std::io::{Read, Write, BufWriter, BufReader};

pub struct Linear {
    pub weight: Tensor, // in × out
    pub bias: Tensor,   // 1 × out
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, seed: &mut u64) -> Self {
        let scale = (2.0 / in_features as f32).sqrt();
        let w = Matrix::random(in_features, out_features, scale, seed);
        let b = Matrix::zeros(1, out_features);
        Self {
            weight: Tensor::leaf(w, true),
            bias: Tensor::leaf(b, true),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        x.matmul(&self.weight).add(&self.bias)
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone(), self.bias.clone()]
    }

    // ----- Kaydet / Yükle -----

    pub fn write_to<W: Write>(&self, w: &mut W) -> std::io::Result<()> {
        write_matrix(w, &self.weight.data.borrow())?;
        write_matrix(w, &self.bias.data.borrow())?;
        Ok(())
    }

    pub fn read_from<R: Read>(r: &mut R) -> std::io::Result<Self> {
        let weight = read_matrix(r)?;
        let bias = read_matrix(r)?;
        Ok(Self {
            weight: Tensor::leaf(weight, true),
            bias: Tensor::leaf(bias, true),
        })
    }
}

fn write_matrix<W: Write>(w: &mut W, m: &Matrix) -> std::io::Result<()> {
    w.write_all(&(m.rows as u32).to_le_bytes())?;
    w.write_all(&(m.cols as u32).to_le_bytes())?;
    for &v in &m.data {
        w.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

fn read_matrix<R: Read>(r: &mut R) -> std::io::Result<Matrix> {
    let rows = read_u32(r)? as usize;
    let cols = read_u32(r)? as usize;
    let mut data = Vec::with_capacity(rows * cols);
    let mut buf = [0u8; 4];
    for _ in 0..rows * cols {
        r.read_exact(&mut buf)?;
        data.push(f32::from_le_bytes(buf));
    }
    Ok(Matrix::from_vec(rows, cols, data))
}

fn read_u32<R: Read>(r: &mut R) -> std::io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

// ----- Model toplu kaydet/yükle -----

const MAGIC: &[u8; 8] = b"SFISH001";
const VERSION: u32 = 1;

pub fn save_model(path: &str, layers: &[&Linear]) -> std::io::Result<()> {
    let f = File::create(path)?;
    let mut w = BufWriter::new(f);
    w.write_all(MAGIC)?;
    w.write_all(&VERSION.to_le_bytes())?;
    w.write_all(&(layers.len() as u32).to_le_bytes())?;
    for l in layers {
        l.write_to(&mut w)?;
    }
    w.flush()?;
    Ok(())
}

pub fn load_model(path: &str, n_layers: usize) -> std::io::Result<Vec<Linear>> {
    let f = File::open(path)?;
    let mut r = BufReader::new(f);

    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    assert_eq!(&magic, MAGIC, "magic yanlış — bu bir Stonefish modeli değil");

    let version = read_u32(&mut r)?;
    assert_eq!(version, VERSION, "desteklenmeyen versiyon: {}", version);

    let saved_layers = read_u32(&mut r)? as usize;
    assert_eq!(
        saved_layers, n_layers,
        "kaydedilmiş katman sayısı ({}) ile yüklenen ({}) uyuşmuyor",
        saved_layers, n_layers
    );

    let mut out = Vec::with_capacity(n_layers);
    for _ in 0..n_layers {
        out.push(Linear::read_from(&mut r)?);
    }
    Ok(out)
}