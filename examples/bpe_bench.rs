// examples/bpe_bench.rs
// BPE tokenizasyon hızını ölçer: bellek içi encode ve disk cache yazımı ayrı ayrı.

use stonefish_ai::bpe::train_bpe;
use stonefish_ai::corpus::tokenize_file_to_cache;

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or("data/en/bench10.txt".into());
    let bytes = std::fs::read(&path).expect("dosya okunamadı");
    let text = String::from_utf8_lossy(&bytes).into_owned();
    println!(
        "dosya: {path} ({:.1} MB)",
        bytes.len() as f64 / (1 << 20) as f64
    );

    // BPE'yi aynı dosyadan öğren (küçük vocab, hız odaklı)
    let t0 = std::time::Instant::now();
    let bpe = train_bpe(&text, 16_384, 300_000);
    println!(
        "BPE öğrenimi: {:.2} s (vocab={})",
        t0.elapsed().as_secs_f64(),
        bpe.vocab()
    );

    // 1) Bellek içi, önbelleksiz (naif)
    let t0 = std::time::Instant::now();
    let n = bpe.encode(&text).len();
    let dt = t0.elapsed().as_secs_f64();
    println!(
        "bellek içi: {n} token, {:.2} s → {:.2} MB/s (önbelleksiz)",
        dt,
        bytes.len() as f64 / (1 << 20) as f64 / dt
    );

    // 2) Disk cache yazımı (kelime önbellekli yol)
    let t0 = std::time::Instant::now();
    let n2 = tokenize_file_to_cache(&path, "/tmp_bpe_cache.bin", &bpe, 64 << 20).unwrap_or(0);
    let dt2 = t0.elapsed().as_secs_f64();
    println!(
        "disk cache: {n2} token, {:.2} s → {:.2} MB/s",
        dt2,
        bytes.len() as f64 / (1 << 20) as f64 / dt2
    );
    let _ = std::fs::remove_file("/tmp_bpe_cache.bin");
}
