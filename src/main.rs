extern crate mnist;
extern crate network;
extern crate rand;
#[macro_use]
extern crate clap;
extern crate serde_json;

use std::fs::File;

use serde_json::{from_reader, to_writer};

use rand::seq::SliceRandom;

use mnist::{Mnist, MnistBuilder};
use network::Network;

static MNIST_ROWS: usize = 28;
static MNIST_COLS: usize = 28;
static MNIST_TRAINING_DATA_SIZE: usize = 50_000;

fn main() {
    let yaml = load_yaml!("cli.yml");
    let matches = clap::App::from_yaml(yaml).get_matches();

    let verbose_mode = matches.is_present("verbose");
    let should_save = matches.is_present("save");
    let dynamic_learn_rate = matches.is_present("dynamic_learn_rate");
    let epochs = matches.value_of("epochs").map(|v| v.parse::<usize>().expect("Epoch was not in valid format")).unwrap_or(50);
    let batch_size = matches.value_of("batch_size").map(|v| v.parse::<usize>().expect("Batch size was not in valid format")).unwrap_or(10);
    let mut learn_rate = matches.value_of("learn_rate").map(|v| v.parse::<f64>().expect("Learn rate was not in valid format")).unwrap_or(0.5);

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



    let mut network =
        File::open("network.json").map(|f| from_reader(f).expect("network.json file invalid, try deleting it.")).unwrap_or_else(|_| {
            println!("No network.json file, creating a new network");
            Network::generate_random(vec![MNIST_ROWS * MNIST_COLS, 30, 10])
        });

    let mut before_n_correct = 0;

    for (i, (input, desired_output)) in test_images.clone().into_iter().enumerate() {
        let output = network.feed_forward(input);

        let number = find_index_of_max(output);

        assert!(tst_lbl[i] == find_index_of_max(desired_output) as u8);

        // if verbose_mode {
        //     println!(
        //         "{}     Network gave: {}, expected: {}",
        //         if number as u8 == tst_lbl[i] { "✅" } else { "❌" },
        //         number,
        //         tst_lbl[i]
        //     );
        // }

        before_n_correct += if number as u8 == tst_lbl[i] { 1 } else { 0 };
    }

    if verbose_mode {
        println!(
            "Before training: correctly identified {}/{} ({}%)",
            before_n_correct,
            test_images.len(),
            f64::from(before_n_correct * 100) / test_images.len() as f64
        );
    }


    let mut rng = rand::thread_rng();

    images.shuffle(&mut rng);

    use std::time::Instant;
    let start = Instant::now();

    let mut last_error = network.examine_error(test_images.clone());

    for i in 0..epochs {
        network.batch_train_iteration(&images, batch_size, learn_rate);

        if i % 5 == 0 {
            let error = network.examine_error(test_images.clone());
            let delta = error - last_error;
            println!("Epoch {}, error: {} ({:>+02.3}%) - lr {:.5}", i, error, delta * 100.0 / last_error, learn_rate);

            if dynamic_learn_rate {
                // If error has increased
                if delta > 0.0 {
                    learn_rate *= 0.99;
                } else {
                    learn_rate *= 1.01
                }
            }

            last_error = error;
        }
    }

    if verbose_mode {
        println!("Took {}s to batch train", start.elapsed().as_secs());
        println!("{}", network.examine_error(test_images.clone()));
    }

    let mut n_correct = 0;

    // The list of incorrect identifications indexed by the number
    let mut incorrect = [0; 10];
    let mut incorrect_guesses = [0; 10];

    for (i, (input, desired_output)) in test_images.clone().into_iter().enumerate() {
        let output = network.feed_forward(input);

        let number = find_index_of_max(output);

        assert!(tst_lbl[i] == find_index_of_max(desired_output) as u8);

        if number as u8 == tst_lbl[i] {
            n_correct += 1;
        } else {
            incorrect_guesses[number] += 1;
            incorrect[tst_lbl[i] as usize] += 1;
        };

    }

    if verbose_mode {
        println!(
            "Correctly identified {}/{} ({}% - +{}%)",
            n_correct,
            test_images.len(),
            f64::from(n_correct * 100) / test_images.len() as f64,
            f64::from(n_correct * 100) / test_images.len() as f64 - f64::from(before_n_correct * 100) / test_images.len() as f64
        );

        println!("Wrong statistics: {}", incorrect.iter().enumerate().map(|(i, incorrect)| format!("{} => {} times\n", i, incorrect)).collect::<String>())
    }

    if should_save {
        let f = File::create("network.json").expect("Couldn't save file");
        to_writer(f, &network).expect("Couldn't write network file");
    }
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
