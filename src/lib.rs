#![feature(generic_const_exprs)]
#![feature(float_algebraic)]

pub mod weights;
use weights::{
    CONV_LAYER1_0, CONV_LAYER1_1, CONV_LAYER2_0, CONV_LAYER2_1, HIDDEN_LAYER1_0, HIDDEN_LAYER1_1,
    OUTPUT_0, OUTPUT_1,
};

#[inline(always)]
fn kahan_sum(input: &[f32]) -> f32 {
    // https://en.wikipedia.org/wiki/Kahan_summation_algorithm
    // Prepare the accumulator.
    let mut sum = 0.;
    // A running compensation for lost low-order bits.
    let mut c = 0.;
    for value in input {
        // c is zero the first time around.
        let y = value - c;
        // Alas, sum is big, y small, so low-order digits of y are lost.
        let t = sum + y;
        // (t - sum) cancels the high-order part of y;
        // subtracting y recovers negative (low part of y)
        c = (t - sum) - y;
        // Algebraically, c should always be zero. Beware
        // overly-aggressive optimizing compilers!
        sum = t;
        // Next time around, the lost low part will be added to y in a fresh attempt.
    }
    return sum;
}

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    // https://keras.io/api/layers/activations/#sigmoid-function
    1.0 / kahan_sum(&[1.0 + f32::exp(-x)])
}

#[inline]
pub fn relu(x: f32) -> f32 {
    if x < 0. {
        0.
    } else {
        x
    }
}

fn dot_product<const X: usize>(a: [f32; X], b: [f32; X]) -> f32 {
    let mut ret: f32 = 0.;
    for x in 0..X {
        ret = kahan_sum(&[ret, a[x] * b[x]]);
    }
    // return a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    return ret;
}

fn dot_product_2d<const X: usize, const Y: usize, const Z: usize>(
    a: [[f32; X]; Y],
    b: [[f32; Z]; X],
) -> [[f32; Z]; Y] {
    let mut ret = [[0.; Z]; Y];
    let mut new_b = [[0.; X]; Z];
    for z in 0..Z {
        for x in 0..X {
            // pivot
            new_b[z][x] = b[x][z];
        }
    }
    for y in 0..Y {
        for z in 0..Z {
            ret[y][z] = dot_product(a[y], new_b[z]);
        }
    }
    return ret;
}

pub type Data2D<const BATCH: usize, const WIDTH: usize, const HEIGHT: usize, const CHANNEL: usize> =
    [[[[f32; CHANNEL]; HEIGHT]; WIDTH]; BATCH];

pub struct Conv2D<
    const FILTER: usize,
    const KERNEL: usize,
    const CH: usize,
    const PADDING: usize = 1,
> {
    // assume square kernel
    // width, height, channels, filters
    kernels: &'static [[[[f32; FILTER]; CH]; KERNEL]; KERNEL],
    biases: &'static [f32; FILTER],
}

impl<const FILTER: usize, const KERNEL: usize, const CH: usize, const PADDING: usize>
    Conv2D<FILTER, KERNEL, CH, PADDING>
{
    pub fn new(
        kernels: &'static [[[[f32; FILTER]; CH]; KERNEL]; KERNEL],
        biases: &'static [f32; FILTER],
    ) -> Self {
        Self {
            kernels: kernels,
            biases: biases,
        }
    }

    fn apply_padding<const WIDTH: usize, const HEIGHT: usize, const CHANNEL: usize>(
        input: [[[f32; CHANNEL]; HEIGHT]; WIDTH],
        channel: usize,
    ) -> [[f32; HEIGHT + PADDING * 2]; WIDTH + PADDING * 2]
    where
        [(); WIDTH + PADDING * 2]:,
        [(); HEIGHT + PADDING * 2]:,
    {
        let mut ret = [[0.; HEIGHT + PADDING * 2]; WIDTH + PADDING * 2];
        // copy input
        for w in 0..WIDTH {
            for h in 0..HEIGHT {
                ret[w + PADDING][h + PADDING] = input[w][h][channel]
            }
        }
        return ret;
    }

    fn apply_filter(input: &[[f32; KERNEL]; KERNEL], kernel: &[[f32; KERNEL]; KERNEL]) -> f32
    where
        [(); KERNEL * KERNEL]:,
    {
        // naive
        let mut ret = [0.; KERNEL * KERNEL];
        for x in 0..KERNEL {
            for y in 0..KERNEL {
                ret[x + (y * KERNEL)] = input[x][y] * kernel[x][y];
            }
        }
        kahan_sum(&ret)
    }

    pub fn compute<
        const BATCH: usize,
        const WIDTH: usize,
        const HEIGHT: usize,
        const CHANNEL: usize,
    >(
        self,
        input: &Data2D<BATCH, WIDTH, HEIGHT, CHANNEL>,
    ) -> Data2D<BATCH, WIDTH, HEIGHT, FILTER>
    where
        [(); WIDTH + PADDING * 2]:,
        [(); HEIGHT + PADDING * 2]:,
        [(); KERNEL * KERNEL]:,
    {
        // input:            [[[[f32; CHANNEL]; HEIGHT]; WIDTH]; BATCH]
        // kernels: &'static [[[[f32; FILTER]; CHANNEL]; KERNEL]; KERNEL],

        let mut ret: Data2D<BATCH, WIDTH, HEIGHT, FILTER> =
            [[[[0.; FILTER]; HEIGHT]; WIDTH]; BATCH];
        for batch in 0..BATCH {
            for channel in 0..CHANNEL {
                let img = input[batch];
                let pad: [[f32; HEIGHT + PADDING * 2]; WIDTH + PADDING * 2] =
                    Conv2D::<FILTER, KERNEL, CHANNEL, PADDING>::apply_padding(img, channel);
                for width in 0..WIDTH {
                    for height in 0..HEIGHT {
                        // copy
                        let mut window = [[0.; KERNEL]; KERNEL];
                        for w in 0..KERNEL {
                            for h in 0..KERNEL {
                                window[w][h] = pad[w + width][h + height];
                            }
                        }
                        // copy
                        for filter in 0..FILTER {
                            let mut kernel = [[0.; KERNEL]; KERNEL];
                            for w in 0..KERNEL {
                                for h in 0..KERNEL {
                                    // println!("{:?} {:?} {:?} {:?}", w, h, filter, channel);
                                    // println!("{:?} {:?} {:?} {:?}", WIDTH, HEIGHT, FILTER, CHANNEL);
                                    // println!("{:?}", self.kernels[w][h][channel][filter]);
                                    kernel[w][h] = self.kernels[w][h][channel][filter];
                                }
                            }
                            let value: f32 = kahan_sum(&[
                                ret[batch][width][height][filter],
                                Conv2D::<FILTER, KERNEL, CHANNEL, PADDING>::apply_filter(
                                    &window, &kernel,
                                ),
                            ]);
                            ret[batch][width][height][filter] = value;
                        }
                    }
                }
            }
        }
        for batch in 0..BATCH {
            for width in 0..WIDTH {
                for height in 0..HEIGHT {
                    for filter in 0..FILTER {
                        // apply biase
                        let value =
                            kahan_sum(&[self.biases[filter], ret[batch][width][height][filter]]);
                        // relu
                        ret[batch][width][height][filter] = relu(value);
                    }
                }
            }
        }

        // kernels: &'static [[[[f32; FILTER]; CHANNEL]; KERNEL]; KERNEL],
        // self.kernels
        return ret;
    }
}

pub struct MaxPooling2D<const POOL_SIZE_WIDTH: usize, const POOL_SIZE_HEIGHT: usize> {}

impl<const POOL_SIZE_WIDTH: usize, const POOL_SIZE_HEIGHT: usize>
    MaxPooling2D<POOL_SIZE_WIDTH, POOL_SIZE_HEIGHT>
{
    pub fn new() -> Self {
        Self {}
    }

    pub fn compute<
        const BATCH: usize,
        const WIDTH: usize,
        const HEIGHT: usize,
        const CHANNEL: usize,
    >(
        input: &Data2D<BATCH, WIDTH, HEIGHT, CHANNEL>,
    ) -> Data2D<BATCH, { WIDTH / POOL_SIZE_WIDTH }, { HEIGHT / POOL_SIZE_HEIGHT }, CHANNEL>
    where
        [(); WIDTH / POOL_SIZE_WIDTH]:,
        [(); HEIGHT / POOL_SIZE_HEIGHT]:,
    {
        let mut ret: Data2D<
            BATCH,
            { WIDTH / POOL_SIZE_WIDTH },
            { HEIGHT / POOL_SIZE_HEIGHT },
            CHANNEL,
        > = [[[[0.; CHANNEL]; HEIGHT / POOL_SIZE_HEIGHT]; WIDTH / POOL_SIZE_WIDTH]; BATCH];
        for batch in 0..BATCH {
            for channel in 0..CHANNEL {
                for width in (0..WIDTH).step_by(POOL_SIZE_WIDTH) {
                    for height in (0..HEIGHT).step_by(POOL_SIZE_HEIGHT) {
                        // copy
                        for w in 0..POOL_SIZE_WIDTH {
                            for h in 0..POOL_SIZE_HEIGHT {
                                let pool_w = width / POOL_SIZE_WIDTH;
                                let pool_h = height / POOL_SIZE_HEIGHT;
                                if pool_w >= WIDTH / POOL_SIZE_WIDTH
                                    || pool_h >= HEIGHT / POOL_SIZE_HEIGHT
                                {
                                    continue;
                                }
                                let new_v = input[batch][w + width][h + height][channel];
                                // println!(
                                //     "{:?} {:?} {:?} {:?}",
                                //     height,
                                //     width,
                                //     width / POOL_SIZE_WIDTH,
                                //     height / POOL_SIZE_HEIGHT
                                // );
                                let old_v = ret[batch][pool_w][pool_h][channel];
                                // println!(
                                //     "w*h: {:?} {:?} = {:?}; widht*height: {:?} {:?} = {:?}",
                                //     w,
                                //     h,
                                //     old_v,
                                //     w + width,
                                //     h + height,
                                //     new_v
                                // );
                                if new_v > old_v {
                                    ret[batch][pool_w][pool_h][channel] = new_v;
                                }
                            }
                        }
                    }
                }
            }
        }
        return ret;
    }
}

pub struct Dense<const IN: usize, const UNITS: usize> {
    weights: &'static [[f32; UNITS]; IN],
    biases: &'static [f32; UNITS],
}

impl<const IN: usize, const UNITS: usize> Dense<IN, UNITS> {
    pub fn new(weights: &'static [[f32; UNITS]; IN], biases: &'static [f32; UNITS]) -> Self {
        Self {
            weights: weights,
            biases: biases,
        }
    }

    pub fn compute<const BATCH: usize>(
        self,
        input: &[[f32; IN]; BATCH],
        activation: fn(f32) -> f32,
    ) -> [[f32; UNITS]; BATCH] {
        let mut ret = [[0.; UNITS]; BATCH];
        let mut weight = [[0.; IN]; UNITS];
        for batch in 0..BATCH {
            for y in 0..IN {
                for x in 0..UNITS {
                    // pivot
                    weight[x][y] = self.weights[y][x];
                }
            }
            for u in 0..UNITS {
                // let inpt: [f32; IN] = input[0][0..IN].try_into().unwrap();
                let value = dot_product::<IN>(weight[u], input[batch][..].try_into().unwrap());
                let value = kahan_sum(&[self.biases[u], value]);
                ret[batch][u] = activation(value);
            }
        }
        return ret;
    }
}

pub struct Flatten {}

impl Flatten {
    pub fn new() -> Self {
        Self {}
    }

    pub fn compute<
        const BATCH: usize,
        const WIDTH: usize,
        const HEIGHT: usize,
        const CHANNEL: usize,
    >(
        input: &Data2D<BATCH, WIDTH, HEIGHT, CHANNEL>,
    ) -> [[f32; CHANNEL * HEIGHT * WIDTH]; BATCH]
    where
        [(); CHANNEL * HEIGHT * WIDTH]:,
    {
        let mut ret = [[0.; CHANNEL * HEIGHT * WIDTH]; BATCH];
        for batch in 0..BATCH {
            let mut idx = 0;
            for width in 0..WIDTH {
                for height in 0..HEIGHT {
                    for channel in 0..CHANNEL {
                        ret[batch][idx] = input[batch][width][height][channel];
                        idx = idx + 1;
                    }
                }
            }
        }
        return ret;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product() {
        // width, height, channel, filter
        const a: [f32; 5] = [1.1, 2.2, 3.3, 4.4, 5.5];
        const b: [f32; 5] = [6.6, 7.7, 8.8, 9.9, 10.10];
        const EXPECT: f32 = 152.35;
        let result = dot_product(a, b);
        assert_eq!(EXPECT, result);
    }

    #[test]
    fn test_dot_product_2d() {
        const A: [[f32; 2]; 3] = [
            [0.03141212835907936, 0.7923660278320312],
            [0.4824039936065674, 0.7104254961013794],
            [0.1428866684436798, 0.9398795962333679],
        ];
        const B: [[f32; 4]; 2] = [
            [
                0.33833572268486023,
                0.2711726725101471,
                0.08322228491306305,
                0.0010734768584370613,
            ],
            [
                0.7616878151893616,
                0.8389890193939209,
                0.19267819821834564,
                0.8613147735595703,
            ],
        ];
        const EXPECT: [[f32; 4]; 3] = [
            [0.6141634, 0.6733045, 0.15528585, 0.6825103],
            [0.70433694, 0.72685397, 0.17703027, 0.6124178],
            [0.7642385, 0.8272956, 0.19298565, 0.8096855],
        ];
        assert_eq!(EXPECT, dot_product_2d(A, B));
    }

    #[test]
    fn test_dense() {
        // <Dense name=hidden_layer1, built=True>
        const hidden_layer1_0: [[f32; 4]; 16] = [
            [0.3553579, 0.427508, 0.12232667, -0.17092472],
            [0.24856967, 0.5401479, -0.48204347, -0.25762317],
            [-0.26750925, 0.11283821, -0.10641712, 0.45842934],
            [0.11984646, 0.45651853, 0.13271505, 0.09734392],
            [-0.3387057, 0.00986415, -0.3714354, -0.24842408],
            [0.33188248, 0.47429693, -0.28113526, 0.04869717],
            [-0.07395872, -0.4884242, 0.41034818, -0.16948578],
            [-0.51135755, -0.25507802, -0.131192, -0.19819435],
            [0.2636695, 0.03073478, 0.07086343, 0.51168156],
            [-0.35927737, 0.45419002, 0.30080616, 0.21440703],
            [0.35459048, 0.25112015, -0.03825921, 0.03344887],
            [0.3461553, 0.1434198, -0.18238807, 0.08759952],
            [-0.36334592, -0.14893332, -0.37618083, -0.2457836],
            [-0.08559889, -0.36976874, 0.5415273, 0.04384309],
            [0.38561654, 0.407138, 0.2559715, 0.5248525],
            [-0.06982213, 0.49745238, -0.1945281, -0.4486451],
        ];
        // <Dense name=hidden_layer1, built=True>
        const hidden_layer1_1: [f32; 4] = [0., 0., 0., 0.];
        const INPUT: [[f32; 16]; 1] = [[
            0., 0., 0.08472509, 0.04946682, 0., 0., 0.24397539, 0., 0., 0., 0.4876667, 0.22362623,
            0., 0., 0.3061186, 0.15125376,
        ]];
        const EXPECT: [[f32; 4]; 1] = [[0.32303447, 0.26738867, 0.08715367, 0.1310147]];

        const DENSE: Dense<16, 4> = Dense {
            weights: &hidden_layer1_0,
            biases: &hidden_layer1_1,
        };
        let result = DENSE.compute(&INPUT, relu);
        assert_eq!(EXPECT, result);
    }

    #[test]
    fn test_sigmoid() {
        assert_eq!(0.99995637, sigmoid(10.039195));
    }

    #[test]
    fn test_dense_sigmoid() {
        // <Dense name=output, built=True>
        const output_0: [[f32; 1]; 40] = [
            [0.5638675],
            [-0.6254646],
            [0.],
            [0.75793827],
            [0.7556081],
            [-0.48717764],
            [-0.],
            [0.37855738],
            [-0.],
            [0.48090476],
            [-0.46746048],
            [-0.6898703],
            [-0.],
            [0.4250545],
            [-0.],
            [0.7001315],
            [0.60754246],
            [-0.5572943],
            [0.],
            [-0.70520324],
            [-0.68959004],
            [0.],
            [0.],
            [-0.],
            [-0.73336697],
            [0.],
            [-0.6262728],
            [0.],
            [0.],
            [0.5036727],
            [-0.],
            [-0.],
            [0.5800524],
            [0.],
            [-0.],
            [0.],
            [-0.66981155],
            [-0.],
            [0.],
            [-0.4503886],
        ];
        // <Dense name=output, built=True>
        const output_1: [f32; 1] = [-0.6205936];

        const INPUT: [[f32; 40]; 1] = [[
            9.054273, 0., 0., 0., 1.7320275, 0.38694936, 0., 7.11669, 0., 2.858392, 0., 0., 0.,
            0.85982084, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0.,
        ]];
        const EXPECT: [[f32; 1]; 1] = [[0.99995637]];

        const DENSE: Dense<40, 1> = Dense {
            weights: &output_0,
            biases: &output_1,
        };
        let result = DENSE.compute(&INPUT, sigmoid);
        assert_eq!(EXPECT, result);
    }

    #[test]
    fn test_flatten() {
        const INPUT: Data2D<1, 2, 2, 4> = [[
            [
                [0., 0., 0.7381524, 0.4555484],
                [0., 0.3886321, 0.5940174, 0.28782377],
            ],
            [
                [0., 0., 0.6561312, 0.57637906],
                [0., 0.08597483, 0.7157949, 0.4111214],
            ],
        ]];

        const EXPECT: [[f32; 16]; 1] = [[
            0., 0., 0.7381524, 0.4555484, 0., 0.3886321, 0.5940174, 0.28782377, 0., 0., 0.6561312,
            0.57637906, 0., 0.08597483, 0.7157949, 0.4111214,
        ]];

        let result = Flatten::compute(&INPUT);
        assert_eq!(EXPECT, result);
    }

    #[test]
    fn test_max_pooling() {
        const INPUT: [[[[f32; 2]; 4]; 5]; 1] = [[
            [
                [0.8618821, 0.7179448],
                [0.9245458, 0.5684368],
                [0.2874808, 0.7954998],
                [0.18317042, 0.7043545],
            ],
            [
                [0.42794177, 0.8792396],
                [0.8744181, 0.6444644],
                [0.3827954, 0.291953],
                [0.27141148, 0.94023806],
            ],
            [
                [0.13538386, 0.61883837],
                [0.43485984, 0.87023354],
                [0.81098336, 0.3656716],
                [0.24290077, 0.5401536],
            ],
            [
                [0.34529236, 0.68077046],
                [0.83808637, 0.57860255],
                [0.72407436, 0.34613603],
                [0.30683133, 0.8156411],
            ],
            [
                [0.3644888, 0.53631675],
                [0.7026905, 0.27365837],
                [0.6111978, 0.9249894],
                [0.00985912, 0.17269658],
            ],
        ]];
        const EXPECT: [[[[f32; 2]; 1]; 2]; 1] =
            [[[[0.9245458, 0.8792396]], [[0.83808637, 0.87023354]]]];

        let result = MaxPooling2D::<2, 3>::compute(&INPUT);
        assert_eq!(EXPECT, result);
    }

    #[test]
    fn test_apply_padding() {
        const PADDING: usize = 2;

        // [[f32; HEIGHT]; WIDTH]
        const INPUT: [[[f32; 1]; 3]; 2] = [[[1.], [2.], [3.]], [[4.], [5.], [6.]]];

        const EXPECT: [[f32; 3 + PADDING * 2]; 2 + PADDING * 2] = [
            [0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 1., 2., 3., 0., 0.],
            [0., 0., 4., 5., 6., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0.],
        ];
        // filter, kernel, channel, padding
        // width, height, channel
        let result = Conv2D::<0, 0, 1, PADDING>::apply_padding::<2, 3, 1>(INPUT, 0);
        assert_eq!(EXPECT, result);
    }

    #[test]
    fn test_apply_filter() {
        let mut input: [[f32; 2]; 2] = [[0., 1.], [2., 3.]];

        const KERNEL: [[f32; 2]; 2] = [[0., 1.], [-1., 0.]];

        // filter, kernel, channel, padding
        let result = Conv2D::<0, 2, 0>::apply_filter(&mut input, &KERNEL);
        assert_eq!(-1., result);
    }
    #[test]
    fn test_conv_2d_1() {
        // width, height, channel, filter
        // <Conv2D name=conv_layer2, built=True>
        const KERNEL: [[[[f32; 4]; 2]; 3]; 3] = [
            [
                [
                    [-0.10294533, -0.31922388, 0.28605798, 0.14521518],
                    [0.31365332, 0.29088387, -0.22836797, 0.01891699],
                ],
                [
                    [-0.04012999, -0.25735307, 0.03558517, 0.14372197],
                    [-0.01347336, -0.06273785, 0.15714788, 0.16883817],
                ],
                [
                    [-0.10451333, 0.2038621, 0.13296556, 0.15900835],
                    [0.02867231, 0.10157475, -0.14303868, 0.11090508],
                ],
            ],
            [
                [
                    [-0.24855892, 0.08615875, -0.12646827, 0.25789002],
                    [-0.21271817, -0.17818849, -0.12393165, -0.03717676],
                ],
                [
                    [-0.15483594, -0.16285253, 0.188939, -0.16984694],
                    [-0.00897932, 0.04393253, 0.24785164, 0.01429653],
                ],
                [
                    [0.33236822, 0.00291491, 0.19783428, -0.23403566],
                    [-0.02961078, -0.26299843, 0.14215884, -0.3164309],
                ],
            ],
            [
                [
                    [0.21933976, -0.04190993, 0.19987527, 0.31511292],
                    [0.2004784, 0.32533893, -0.17781147, -0.15614438],
                ],
                [
                    [0.12380132, -0.31084967, 0.08239189, -0.25888652],
                    [-0.17307696, 0.09909907, 0.19090703, -0.23401594],
                ],
                [
                    [-0.07136664, -0.291547, 0.22403315, 0.32967845],
                    [0.12239122, -0.19718298, -0.12862341, 0.14891806],
                ],
            ],
        ];
        // <Conv2D name=conv_layer2, built=True>
        const BIASE: [f32; 4] = [0., 0., 0., 0.];

        const INPUT: [[[[f32; 2]; 4]; 5]; 1] = [[
            [
                [0.13884754, 0.42307392],
                [0.38799563, 0.81333894],
                [0.73195446, 0.42373198],
                [0.29899904, 0.34157065],
            ],
            [
                [0.98544455, 0.5179999],
                [0.03709385, 0.50215906],
                [0.06041202, 0.48145622],
                [0.38670895, 0.41213065],
            ],
            [
                [0.8452877, 0.5825221],
                [0.97884613, 0.37441128],
                [0.2154824, 0.2524737],
                [0.91812366, 0.33173007],
            ],
            [
                [0.00938968, 0.9074183],
                [0.40721375, 0.95042],
                [0.8092663, 0.14492233],
                [0.8470683, 0.05134909],
            ],
            [
                [0.03217491, 0.30155632],
                [0.80078423, 0.04646876],
                [0.3897075, 0.12623146],
                [0.7701093, 0.03028799],
            ],
        ]];
        const EXPECT: [[[[f32; 4]; 4]; 5]; 1] = [[
            [
                [0.1707344, 0.0, 0.44727874, 0.0],
                [0.33113483, 0.0, 0.56533515, 0.0],
                [0.0, 0.0, 0.2498007, 0.0],
                [0.0, 0.028611444, 0.033075362, 0.0],
            ],
            [
                [0.0, 0.0, 0.75196636, 0.0],
                [0.042058818, 0.0, 0.37777904, 0.39691755],
                [0.39305502, 0.0, 0.5657418, 0.5877021],
                [0.014074668, 0.0, 0.42198786, 0.0],
            ],
            [
                [0.0735062, 0.0, 0.74352515, 0.0],
                [0.0, 0.0, 0.57834804, 0.032944143],
                [0.35372505, 0.0, 0.31800798, 0.12113665],
                [0.17385563, 0.0, 0.3985847, 0.039099522],
            ],
            [
                [0.0, 0.0, 0.8742154, 0.22436325],
                [0.18006934, 0.0, 0.6732707, 0.0],
                [0.0, 0.0, 0.8875246, 0.49846366],
                [0.0, 0.0, 0.26593047, 0.18083946],
            ],
            [
                [0.22917835, 0.118329555, 0.30698043, 0.121443346],
                [0.10331474, 0.06483744, 0.26260072, 0.11286594],
                [0.119357586, 0.104085416, 0.210844, 0.30885994],
                [0.0, 0.0, 0.32469484, 0.21611236],
            ],
        ]];
        // filter, kernel, channel, padding
        const CONV2D: Conv2D<4, 3, 2, 1> = Conv2D {
            kernels: &KERNEL,
            biases: &BIASE,
        };
        // batch, width, height
        let result = CONV2D.compute(&INPUT);
        assert_eq!(EXPECT, result);
    }
    #[test]
    fn test_conv_2d() {
        // width, height, channel, filter
        const KERNELS: [[[[f32; 4]; 1]; 3]; 3] = [
            [
                [[-0.0198544, -0.05535957, -0.1830187, -0.2850191]],
                [[-0.27483788, -0.26789814, 0.36401772, 0.36395854]],
                [[-0.21503669, 0.35114855, -0.08438653, 0.16821688]],
            ],
            [
                [[0.35011005, 0.2592522, 0.2360487, -0.30998302]],
                [[0.33286005, -0.19829306, 0.27226704, 0.24552095]],
                [[-0.33642894, 0.2218067, 0.26484728, 0.0039176]],
            ],
            [
                [[0.23130387, -0.07086238, -0.25532138, -0.07740435]],
                [[-0.32429624, -0.23164338, 0.30995, -0.12517083]],
                [[0.36194956, -0.00861764, 0.19947255, 0.17417186]],
            ],
        ];

        const BIASES: [f32; 4] = [0., 0., 0., 0.];
        // filter, kernel, channel, padding
        const CONV2D: Conv2D<4, 3, 1, 1> = Conv2D {
            kernels: &KERNELS,
            biases: &BIASES,
        };
        const INPUT: [[[[f32; 1]; 5]; 5]; 1] = [[
            [
                [0.0300258],
                [0.72792135],
                [0.07263154],
                [0.67781391],
                [0.17034528],
            ],
            [
                [0.48499305],
                [0.78204029],
                [0.56765779],
                [0.72132871],
                [0.18736969],
            ],
            [
                [0.07018744],
                [0.80709369],
                [0.60542652],
                [0.73412491],
                [0.08871114],
            ],
            [
                [0.00837785],
                [0.48322992],
                [0.68990537],
                [0.17449937],
                [0.95262542],
            ],
            [
                [0.82936199],
                [0.19334572],
                [0.22409978],
                [0.62298886],
                [0.32074535],
            ],
        ]];
        const EXPECT: [[[[f32; 4]; 5]; 5]; 1] = [[
            [
                [0.0, 0.03641916, 0.50728214, 0.0857261],
                [0.29240447, 0.0, 0.45630926, 0.13313779],
                [0.3088768, 0.13152899, 0.49127644, 0.0],
                [0.15893273, 0.0, 0.3628222, 0.04297634],
                [0.40009344, 0.04742842, 0.08028108, 0.0],
            ],
            [
                [0.0029174685, 0.301642, 0.47141966, 0.38730407],
                [0.0, 0.0, 1.0841045, 0.31147552],
                [0.29596823, 0.22467466, 0.49423504, 0.0],
                [0.08551864, 0.0, 0.6897409, 0.13321812],
                [0.39567465, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.3036797, 0.44240648, 0.41158047],
                [0.0, 0.0, 0.82992446, 0.47968823],
                [0.0, 0.114389844, 0.6776862, 0.0],
                [0.62446207, 0.0, 0.57711715, 0.2158545],
                [0.0, 0.0, 0.38434443, 0.0],
            ],
            [
                [0.0, 0.17634664, 0.38333267, 0.0951256],
                [0.0, 0.0, 0.4390011, 0.4449701],
                [0.19737409, 0.0071131336, 0.5032061, 0.1995726],
                [0.0, 0.0, 0.81146383, 0.0],
                [0.37931135, 0.0, 0.13884534, 0.0],
            ],
            [
                [0.10479966, 0.045869723, 0.23928662, 0.2887198],
                [0.0, 0.3387207, 0.42391592, 0.080801755],
                [0.0, 0.0, 0.41962317, 0.14024885],
                [0.0, 0.25527957, 0.16433287, 0.11186825],
                [0.05959586, 0.0, 0.54921997, 0.18261422],
            ],
        ]];
        // batch, width, height
        let result = CONV2D.compute(&INPUT);
        assert_eq!(EXPECT, result);
    }

    #[test]
    fn test_conv_2d_2() {
        // filter, kernel, channel, padding
        let CONV2D: Conv2D<4, 3, 4, 1> = Conv2D::new(&CONV_LAYER2_0, &CONV_LAYER2_1);

        const INPUT: [[[[f32; 4]; 9]; 3]; 1] = [[
            [
                [0.5455442, 0.8091933, 0.18543914, 0.26258802],
                [0.93572986, 0.44474605, 0.51912385, 0.28394154],
                [0.05650072, 0.16800629, 0.50034165, 0.2590678],
                [0.29519555, 0.5181893, 0.70944434, 0.8875523],
                [0.8623705, 0.72811335, 0.99623, 0.68407166],
                [0.01303922, 0.96377355, 0.64131063, 0.18172368],
                [0.09350159, 0.05790828, 0.07820505, 0.19998062],
                [0.4042971, 0.88240975, 0.35889035, 0.7085421],
                [0.62356406, 0.04042891, 0.67373264, 0.83260334],
            ],
            [
                [0.96270496, 0.70626956, 0.76873714, 0.59147704],
                [0.02834131, 0.3631355, 0.64297026, 0.5312712],
                [0.5502297, 0.1582327, 0.7816544, 0.9793155],
                [0.84208006, 0.78974706, 0.23638077, 0.50061125],
                [0.20438215, 0.99471986, 0.4764716, 0.785392],
                [0.39988613, 0.4509993, 0.12808377, 0.6716046],
                [0.9381935, 0.26094148, 0.81270576, 0.60286516],
                [0.24181552, 0.3686931, 0.5812463, 0.584009],
                [0.5954171, 0.7063083, 0.55417055, 0.5681642],
            ],
            [
                [0.06773934, 0.1645458, 0.7366788, 0.02417892],
                [0.03379698, 0.3187281, 0.7259706, 0.97601813],
                [0.7372511, 0.5295905, 0.5379987, 0.46282125],
                [0.10154913, 0.68767875, 0.34433648, 0.17845553],
                [0.07099258, 0.43652534, 0.43785042, 0.8069608],
                [0.5229556, 0.56896156, 0.5047504, 0.3884375],
                [0.12586349, 0.56530666, 0.8498019, 0.51525015],
                [0.8125338, 0.17129755, 0.97851723, 0.07544879],
                [0.32183638, 0.6163863, 0.04249009, 0.0496322],
            ],
        ]];

        const EXPECT: [[[[f32; 4]; 9]; 3]; 1] = [[
            [
                [0.0, 0.21746731, 1.2101898, 0.0],
                [0.0, 1.3387439, 0.9630165, 0.0],
                [0.03579089, 1.0124657, 0.08651537, 0.0],
                [0.0, 0.42422158, 0.9680302, 0.0],
                [0.0, 0.78293294, 0.81337583, 0.0],
                [0.21830809, 1.0684851, 0.48163652, 0.0],
                [0.07888746, 0.7431706, 0.46671158, 0.0],
                [0.0, 1.4801388, 0.646973, 0.0],
                [0.0, 0.848565, 1.2670228, 0.0],
            ],
            [
                [0.0, 0.0, 0.8029535, 0.0],
                [0.0, 0.64602184, 0.4402789, 0.0],
                [0.0, 0.0, 0.22565031, 0.0],
                [0.0, 0.0, 0.6284471, 0.0],
                [0.0, 0.43156433, 0.3925596, 0.0],
                [0.0, 0.0, 0.47808468, 0.0],
                [0.10104442, 1.3614111, 0.7110237, 0.0],
                [0.0, 0.11731458, 0.10571259, 0.0],
                [0.0, 0.7824625, 2.1540656, 0.0],
            ],
            [
                [0.0, 0.10595065, 0.59955573, 0.08652222],
                [0.0, 0.0, 0.1357885, 0.35522163],
                [0.089240685, 0.0, 1.1959964, 0.019840002],
                [0.0, 0.34807718, 0.19859618, 0.0],
                [0.0, 0.57321966, 0.9959043, 0.0],
                [0.0, 0.1016621, 0.68176806, 0.0],
                [0.0, 0.32265162, 1.1119791, 0.0],
                [0.1714893, 0.0, 0.57611334, 0.0],
                [0.0, 0.24417311, 1.1131107, 0.0],
            ],
        ]];

        // batch, width, height
        let result = CONV2D.compute(&INPUT);
        assert_eq!(EXPECT, result);
    }
}
