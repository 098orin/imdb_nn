mod activation;
mod imdb;
mod nn;

fn main() {
    let (train_bows, train_labels) = imdb::load_bow("aclImdb_v1/aclImdb/test/labeledBow.feat");
    let (test_bows, test_labels) = imdb::load_bow("aclImdb_v1/aclImdb/train/labeledBow.feat");
    println!(
        "Train data: {}, Test data: {}",
        train_bows.len(),
        test_bows.len()
    );
    println!("Vocabulary size: {}", imdb::VOCAB_SIZE);
}
