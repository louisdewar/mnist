extern crate mnist;
extern crate network;
extern crate rand;

use rand::seq::SliceRandom;

use mnist::{Mnist, MnistBuilder};
use network::Network;

static MNIST_ROWS: usize = 28;
static MNIST_COLS: usize = 28;
static MNIST_TRAINING_DATA_SIZE: usize = 50_000;

static BATCH_SIZE: usize = 10;

fn main() {
    println!("Hello, world!");

    let (normalized, trn_lbl, normalized_tst, tst_lbl) = setup_mnist();

    let mut images = normalized
        .chunks(MNIST_ROWS * MNIST_COLS)
        .enumerate()
        .map(|(i, image)| (image.to_vec(), result_to_output_layer(trn_lbl[i])))
        .collect::<Vec<_>>();

    let test_images = normalized_tst
        .chunks(MNIST_ROWS * MNIST_COLS)
        .enumerate()
        .map(|(i, image)| (image.to_vec(), result_to_output_layer(tst_lbl[i])))
        .collect::<Vec<_>>();

    let mut network = Network::generate_random(vec![MNIST_ROWS * MNIST_COLS, 30, 10]);

    let mut before_n_correct = 0;

    for (i, (input, desired_output)) in test_images.clone().into_iter().enumerate() {
        let output = network.feed_forward(input);

        let number = find_index_of_max(output);

        assert!(tst_lbl[i] == find_index_of_max(desired_output) as u8);

        println!(
            "{}     Network gave: {}, expected: {}",
            if number as u8 == tst_lbl[i] { "✅" } else { "❌" },
            number,
            tst_lbl[i]
        );

        before_n_correct += if number as u8 == tst_lbl[i] { 1 } else { 0 };
    }

    println!(
        "Correctly identified {}/{} ({}%)",
        before_n_correct,
        test_images.len(),
        f64::from(before_n_correct * 100) / test_images.len() as f64
    );

    let mut rng = rand::thread_rng();

    images.shuffle(&mut rng);

    use std::time::Instant;
    let start = Instant::now();

    network.batch_train(images.clone(), 50, BATCH_SIZE, 0.5, Some(&test_images));

    println!("Took {}s to batch train", start.elapsed().as_secs());

    println!("{}", network.examine_error(test_images.clone()));

    let mut n_correct = 0;

    for (i, (input, desired_output)) in test_images.clone().into_iter().enumerate() {
        let output = network.feed_forward(input);

        let number = find_index_of_max(output);

        assert!(tst_lbl[i] == find_index_of_max(desired_output) as u8);

        n_correct += if number as u8 == tst_lbl[i] { 1 } else { 0 };
    }

    println!(
        "Correctly identified {}/{} ({}% - +{}%)",
        n_correct,
        test_images.len(),
        f64::from(n_correct * 100) / test_images.len() as f64,
        f64::from(n_correct * 100) / test_images.len() as f64 - f64::from(before_n_correct * 100) / test_images.len() as f64
    );
}

fn find_index_of_max(a: Vec<f64>) -> usize {
    use std::f64;
    a.into_iter()
        .enumerate()
        .fold((std::usize::MAX, f64::NAN), |(i1, x), (i2, y)| {
            if x > y {
                (i1, x)
            } else {
                (i2, y)
            }
        })
        .0
}

fn result_to_output_layer(result: u8) -> Vec<f64> {
    let mut output_layer = vec![0.0; 10];
    output_layer[result as usize] = 1.0;
    output_layer
}

fn setup_mnist() -> (Vec<f64>, Vec<u8>, Vec<f64>, Vec<u8>) {
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(MNIST_TRAINING_DATA_SIZE as u32)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();

    println!("The first label is: {}", trn_lbl[0]);

    let normalized_train = trn_img
        .iter()
        .map(|x| f64::from(*x) / 255.0)
        .collect::<Vec<f64>>();

    let normalized_test = tst_img
        .iter()
        .map(|x| f64::from(*x) / 255.0)
        .collect::<Vec<f64>>();

    (normalized_train, trn_lbl, normalized_test, tst_lbl)
}
