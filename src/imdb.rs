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
