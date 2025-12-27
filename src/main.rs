mod activation;
mod dataset_util;
mod imdb;
mod nn;

use dataset_util::BatchIter;
use imdb::BoWDataset;

use indicatif::{ProgressBar, ProgressStyle};

const VOCAB_SIZE: usize = 89527;
const DATASET_SIZE: usize = 25000;

fn main() {
    let bar_maker = |len| {
        ProgressBar::new(len)
            .with_style(
                ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] {msg} {percent:>3}% [{bar:60.cyan/blue}] {human_pos}/{human_len} ({per_sec}, {eta})")
                    .unwrap()
                    .progress_chars("#>-")
            )
    };

    let train_bow_dataset_maker =
        || BoWDataset::new("aclImdb_v1/aclImdb/train/unsupBow.feat", VOCAB_SIZE);
    let test_bow_dataset = BoWDataset::new("aclImdb_v1/aclImdb/test/labeledBow.feat", VOCAB_SIZE);

    let layer1_linear = Box::new(nn::Linear::new(VOCAB_SIZE, 128));
    let layer1_relu = Box::new(activation::relu::ReLU::new());
    let layer2_linear = Box::new(nn::Linear::new(128, 2));
    let layer2_softmax = Box::new(activation::softmax::Softmax);
    let mut model = nn::Model::new(vec![
        layer1_linear,
        layer1_relu,
        layer2_linear,
        layer2_softmax,
    ]);

    println!("Model initialized.");

    let batch_size: usize = 250;
    let train_epoch: usize = 5;

    let train_datasize: usize = DATASET_SIZE;

    println!("batch_size: {}", batch_size);

    let bar = bar_maker(
        ((train_datasize / batch_size) * train_epoch)
            .try_into()
            .unwrap(),
    );

    for epoch in 0..train_epoch {
        let train_batch_iter = BatchIter::new(train_bow_dataset_maker(), batch_size);

        bar.set_message(format!("Processing Epoch{} Positive.", epoch + 1));

        for data in train_batch_iter {
            let (data_batch, label_batch): (Vec<Vec<f32>>, Vec<usize>) =
                data.iter().cloned().unzip();
            let label_batch: Vec<usize> = label_batch.iter().map(|&u| (u > 5) as usize).collect();
            model.train_batch(&data_batch, &label_batch, 0.01);
            bar.inc(1);
        }
        bar.println(format!("Epoch {} completed.", epoch + 1));
    }

    bar.finish_and_clear();

    println!("Evaluating on test set...");

    let total = DATASET_SIZE;

    let bar = bar_maker(total.try_into().unwrap());

    let mut correct = 0;
    for (x, y) in test_bow_dataset {
        let output = model.forward(&x);
        let predicted: bool = output[0] > output[1];
        if predicted == (y > 5) {
            correct += 1;
        }
        bar.inc(1);
        bar.set_message(format!("Correct: {}", correct));
    }
    bar.finish_and_clear();
    let accuracy = correct as f32 / total as f32;
    println!("Test Accuracy: {:.2}%", accuracy * 100.0);
}
