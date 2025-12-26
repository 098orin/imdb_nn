use std::fs;
use std::path::Path;
use walkdir::WalkDir;

#[derive(Debug)]
pub struct IMDbDataset {
    pub train_texts: Vec<String>,
    pub train_labels: Vec<usize>, // 1: positive, 0: negative
    pub test_texts: Vec<String>,
    pub test_labels: Vec<usize>,
}

impl IMDbDataset {
    pub fn load(base_path: &str) -> Self {
        let train_pos = Self::read_dir(Path::new(base_path).join("train/pos"), 1);
        let train_neg = Self::read_dir(Path::new(base_path).join("train/neg"), 0);
        let test_pos = Self::read_dir(Path::new(base_path).join("test/pos"), 1);
        let test_neg = Self::read_dir(Path::new(base_path).join("test/neg"), 0);

        let (train_texts, train_labels) = train_pos.into_iter().chain(train_neg).unzip();
        let (test_texts, test_labels) = test_pos.into_iter().chain(test_neg).unzip();

        Self {
            train_texts,
            train_labels,
            test_texts,
            test_labels,
        }
    }

    fn read_dir(dir: std::path::PathBuf, label: usize) -> Vec<(String, usize)> {
        let mut data = Vec::new();
        for entry in WalkDir::new(dir)
            .min_depth(1)
            .max_depth(1)
            .into_iter()
            .filter_map(Result::ok)
        {
            if entry.file_type().is_file() {
                let text = fs::read_to_string(entry.path()).unwrap_or_else(|_| String::from(""));
                data.push((text, label));
            }
        }
        data
    }
}

use std::fs::File;
use std::io::{BufRead, BufReader};

pub fn load_bow(path: &str, vocab_size: usize) -> (Vec<Vec<f32>>, Vec<usize>) {
    let file = File::open(path).unwrap();
    let reader = BufReader::new(file);

    let mut data = Vec::new();
    let mut labels = Vec::new();

    for line in reader.lines() {
        let line = line.unwrap();
        let mut vec = vec![0.0; vocab_size];
        let mut tokens = line.split_whitespace();

        // 1つ目のトークンがラベル
        let label: usize = tokens.next().unwrap().parse().unwrap();
        labels.push(label);

        for token in tokens {
            let mut split = token.split(':');
            let idx: usize = split.next().unwrap().parse().unwrap();
            let count: f32 = split.next().unwrap().parse().unwrap();
            vec[idx] = count;
        }

        data.push(vec);
    }

    (data, labels)
}
