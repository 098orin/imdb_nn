use std::fs::File;
use std::io::{BufRead, BufReader};

pub struct BoWDataset {
    reader: BufReader<File>,
    vocab_size: usize,
}

impl BoWDataset {
    pub fn new(feat_file: &str, vocab_size: usize) -> Self {
        let file = File::open(feat_file).unwrap();
        let reader = BufReader::new(file);
        Self { reader, vocab_size }
    }
}

impl Iterator for BoWDataset {
    type Item = (Vec<f32>, usize); // (BoW vector, label)

    fn next(&mut self) -> Option<Self::Item> {
        let mut line = String::new();
        loop {
            line.clear();
            let bytes = self.reader.read_line(&mut line).ok()?;
            if bytes == 0 {
                return None; // EOF
            }

            let mut tokens = line.split_whitespace();
            if let Some(label_str) = tokens.next() {
                let label: usize = label_str.parse().unwrap();
                let mut bow = vec![0.0; self.vocab_size];

                for tok in tokens {
                    let mut split = tok.split(':');
                    if let (Some(idx_str), Some(val_str)) = (split.next(), split.next()) {
                        let idx: usize = idx_str.parse().unwrap();
                        let val: f32 = val_str.parse().unwrap();
                        bow[idx] = val;
                    }
                }

                return Some((bow, label));
            }
        }
    }
}
