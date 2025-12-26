mod activation;
mod imdb;
mod nn;

use imdb::BoWDataset;

const VOCAB_SIZE: usize = 89527;

fn main() {
    let train_bow_dataset = BoWDataset::new("aclImdb_v1/aclImdb/test/labeledBow.feat", VOCAB_SIZE);
    let test_bow_dataset = BoWDataset::new("aclImdb_v1/aclImdb/train/labeledBow.feat", VOCAB_SIZE);
}
