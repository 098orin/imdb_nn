use std::fs::File;
use std::io::{BufRead, BufReader, Seek, SeekFrom};

pub struct BoWDataset {
    reader: BufReader<File>,
    offsets: Vec<u64>,
    vocab_size: usize,
}

impl BoWDataset {
    pub fn new(path: &str, vocab_size: usize) -> Self {
        let file = File::open(path).unwrap();
        let mut reader = BufReader::new(file);

        let mut offsets = Vec::new();
        let mut line = String::new();

        loop {
            let pos = reader.stream_position().unwrap();
            if reader.read_line(&mut line).unwrap() == 0 {
                break;
            }
            offsets.push(pos);
            line.clear();
        }

        reader.seek(SeekFrom::Start(0)).unwrap();

        Self {
            reader,
            offsets,
            vocab_size,
        }
    }

    pub fn len(&self) -> usize {
        self.offsets.len()
    }

    fn read_at(&mut self, idx: usize) -> (Vec<f32>, usize) {
        self.reader
            .seek(SeekFrom::Start(self.offsets[idx]))
            .unwrap();

        let mut line = String::new();
        self.reader.read_line(&mut line).unwrap();

        let mut tokens = line.split_whitespace();
        let label: usize = tokens.next().unwrap().parse().unwrap();

        let mut bow = vec![0.0; self.vocab_size];
        for tok in tokens {
            let mut sp = tok.split(':');
            let i: usize = sp.next().unwrap().parse().unwrap();
            let v: f32 = sp.next().unwrap().parse().unwrap();
            bow[i] = v;
        }

        (bow, label)
    }
}

fn shuffle_indices(n: usize, seed: u64) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n).collect();

    let mut s = seed;
    for i in (1..n).rev() {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        let j = (s % (i as u64 + 1)) as usize;
        idx.swap(i, j);
    }
    idx
}

pub struct DataLoader<'a> {
    dataset: &'a mut BoWDataset,
    indices: Vec<usize>,
    batch_size: usize,
    cursor: usize,
}

impl<'a> DataLoader<'a> {
    pub fn new(dataset: &'a mut BoWDataset, batch_size: usize, seed: u64) -> Self {
        let indices = shuffle_indices(dataset.len(), seed);
        Self {
            dataset,
            indices,
            batch_size,
            cursor: 0,
        }
    }
}

impl<'a> Iterator for DataLoader<'a> {
    type Item = (Vec<Vec<f32>>, Vec<usize>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor >= self.indices.len() {
            return None;
        }

        let end = (self.cursor + self.batch_size).min(self.indices.len());
        let mut batch_idx = self.indices[self.cursor..end].to_vec();
        self.cursor = end;

        // batch内をソート
        batch_idx.sort_unstable();

        let mut xs = Vec::with_capacity(batch_idx.len());
        let mut ys = Vec::with_capacity(batch_idx.len());

        for i in batch_idx {
            let (x, y) = self.dataset.read_at(i);
            xs.push(x);
            ys.push((y > 5) as usize);
        }

        Some((xs, ys))
    }
}
