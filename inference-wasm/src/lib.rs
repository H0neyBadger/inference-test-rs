#![feature(generic_const_exprs)]
#![feature(float_algebraic)]

use wasm_bindgen::prelude::*;
use js_sys;

use inference_test::weights::{
    CONV_LAYER1_0, CONV_LAYER1_1, CONV_LAYER2_0, CONV_LAYER2_1, HIDDEN_LAYER1_0, HIDDEN_LAYER1_1,
    OUTPUT_0, OUTPUT_1,
};
use inference_test::{relu, sigmoid, Conv2D, Data2D, Dense, Flatten, MaxPooling2D};


#[wasm_bindgen]
pub fn predict(stft: js_sys::Array<js_sys::Float32Array>) -> f32 {
    // let input: Vec<Vec<f32>> = input.to_vec().into_iter().map(|x| x.to_vec()).collect();
    // Data2D<1, 99, 43, 1>
    // [
    //     Conv2D(
    //         4,
    //         3,
    //         padding="same",
    //         activation="relu",
    //         kernel_regularizer=regularizers.l2(0.001),
    //         name="conv_layer1",
    //         input_shape=(IMG_WIDTH, IMG_HEIGHT, 1),
    //     ),
    //     MaxPooling2D(name="max_pooling1", pool_size=(2, 2)),
    //     Conv2D(
    //         4,
    //         3,
    //         padding="same",
    //         activation="relu",
    //         kernel_regularizer=regularizers.l2(0.001),
    //         name="conv_layer2",
    //     ),
    //     MaxPooling2D(name="max_pooling2", pool_size=(2, 2)),
    //     Flatten(),
    //     Dropout(0.2),
    //     Dense(
    //         40,
    //         activation="relu",
    //         kernel_regularizer=regularizers.l2(0.001),
    //         name="hidden_layer1",
    //     ),
    //     Dense(
    //         1,
    //         activation="sigmoid",
    //         kernel_regularizer=regularizers.l2(0.001),
    //         name="output",
    //     ),
    // ]
    const BATCH: usize = 1;
    const WIDTH: usize = 99;
    const HEIGHT: usize = 43;
    const FILTER: usize = 1;

    let mut input: Data2D<BATCH, WIDTH, HEIGHT, FILTER> =
        [[[[0.; FILTER]; HEIGHT]; WIDTH]; BATCH];
    for batch in 0..BATCH {
        for width in 0..WIDTH {
            for height in 0..HEIGHT {
                for filter in 0..FILTER {
                    input[batch][width][height][filter] = stft.get(width.try_into().unwrap()).get_index(height.try_into().unwrap());
                }
            }
        }
    }

    // const FILTER: usize,
    // const KERNEL: usize,
    // const CHANNEL: usize,
    // const PADDING: usize = 1,
    // const POOL_SIZE_WIDTH: usize,
    // const POOL_SIZE_HEIGHT: usize
    let conv_layer1: Conv2D<4, 3, 1, 1> = Conv2D::new(&CONV_LAYER1_0, &CONV_LAYER1_1);
    let out: Data2D<1, 99, 43, 4> = conv_layer1.compute::<1, 99, 43, 1>(&input);
    // assert_eq!([[[[0.; 4]; 43]; 99]; 1], out);
    let out: Data2D<1, 49, 21, 4> = MaxPooling2D::<2, 2>::compute(&out);
    // assert_eq!([[[[0.; 4]; 21]; 49]; 1], out);
    let conv_layer2: Conv2D<4, 3, 4, 1> = Conv2D::new(&CONV_LAYER2_0, &CONV_LAYER2_1);
    let out: Data2D<1, 49, 21, 4> = conv_layer2.compute::<1, 49, 21, 4>(&out);
    // assert_eq!([[[[0.; 4]; 21]; 49]; 1], out);
    let out: Data2D<1, 24, 10, 4> = MaxPooling2D::<2, 2>::compute(&out);
    // assert_eq!([[[[0.; 4]; 10]; 24]; 1], out);
    let out: [[f32; 960]; 1] = Flatten::compute::<1, 24, 10, 4>(&out);
    // assert_eq!([[0.; 960]; 1], out);
    let hidden_layer1 = Dense::new(&HIDDEN_LAYER1_0, &HIDDEN_LAYER1_1);
    let out: [[f32; 40]; 1] = hidden_layer1.compute(&out, relu);
    // assert_eq!([[0.; 40]; 1], out);
    let output = Dense::new(&OUTPUT_0, &OUTPUT_1);
    let out: [[f32; 1]; 1] = output.compute(&out, sigmoid);
    // 0.99995637
    out[0][0]
}
