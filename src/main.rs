mod activation;
mod imdb;
mod loss;
mod nn;

use imdb::{BoWDataset, DataLoader};

use indicatif::{ProgressBar, ProgressStyle};

const VOCAB_SIZE: usize = 89527;
const DATASET_SIZE: usize = 25000;

fn main() {
    let bar_maker = |len| {
        ProgressBar::new(len)
            .with_style(
                ProgressStyle::with_template(
                    "{spinner:.green} [{elapsed_precise}] {msg} {percent:>3}% [{wide_bar:.cyan/blue}] {human_pos}/{human_len} ({per_sec}, {eta})",
                )
                .unwrap()
                .progress_chars("#>-"),
            )
    };

    let train_bow_dataset_maker =
        || BoWDataset::new("aclImdb_v1/aclImdb/train/labeledBow.feat", VOCAB_SIZE);
    let mut test_bow_dataset =
        BoWDataset::new("aclImdb_v1/aclImdb/test/labeledBow.feat", VOCAB_SIZE);

    let layer1_linear = Box::new(nn::Linear::new(VOCAB_SIZE, 128));
    let layer1_relu = Box::new(activation::relu::ReLU::new(128));
    let layer2_linear = Box::new(nn::Linear::new(128, 2));

    let mut model = nn::Model::new(vec![layer1_linear, layer1_relu, layer2_linear]);

    println!("Model initialized.");

    let batch_size: usize = 128;
    let train_epoch: usize = 20;

    println!("batch_size: {}", batch_size);

    let bar = bar_maker(
        ((DATASET_SIZE / batch_size) * train_epoch)
            .try_into()
            .unwrap(),
    );

    /* ================= 学習 ================= */

    for epoch in 0..train_epoch {
        let mut dataset = train_bow_dataset_maker();

        let loader = DataLoader::new(&mut dataset, batch_size, (epoch + DATASET_SIZE) as u64);

        bar.set_message(format!("Epoch {}", epoch + 1));

        for (x, y) in loader {
            model.train_batch(&x, &y, &mut loss::crossentropy::CrossEntropyLoss, 0.01);
            bar.inc(1);
        }
    }

    bar.finish_with_message(format!("All of epochs ({}) completed.", train_epoch));

    /* ================= 評価 ================= */

    println!("Evaluating on test set...");

    let bar = bar_maker(DATASET_SIZE.try_into().unwrap());
    let mut correct = 0;

    let loader = DataLoader::new(&mut test_bow_dataset, 1, VOCAB_SIZE as u64);

    for (x, y) in loader {
        let (x, y) = (x[0].clone(), y[0]);
        let mut output_buffer = model.create_buffers(VOCAB_SIZE);
        model.forward(&x, &mut output_buffer.activations);
        let output = output_buffer.activations.last().unwrap();
        let predicted = (output[0] < output[1]) as usize;

        if predicted == y {
            correct += 1;
        }

        bar.inc(1);
        bar.set_message(format!("Correct: {}", correct));
    }

    bar.finish();

    let accuracy = correct as f32 / DATASET_SIZE as f32;
    println!("Test Accuracy: {:.2}%", accuracy * 100.0);
}
