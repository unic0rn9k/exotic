use exotic::prelude::*;
use exotic_macro::*;
use mnist::*;
use slas_backend::*;

const TRN_IMAGES: usize = 60_000;

model! {(
    derive: [],
    name: "Net",
    layers: [
        ("DenseLayer::<f32, Blas, {28*28}, 15>", "DenseLayer::random(0.01)"),
        ("Tanh::<f32, 15>", "default()"),
        ("DenseLayer::<f32, Blas, 15, 10>", "DenseLayer::random(0.01)"),
        ("Softmax::<f32, 10>", "default()")
    ],
    float_type: "f32",
    input_len: 784,
    output_len: 10
)}

pub fn main() -> Result<()> {
    let Mnist {
        trn_img, trn_lbl, ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(TRN_IMAGES as u32)
        .base_path("./mnist/")
        .download_and_extract()
        .finalize();

    let trn_img = trn_img
        .iter()
        .map(|n| (*n as f32 / 25.).floor())
        .collect::<Vec<_>>();

    let mut net = Net::new();
    let mut buffer = unsafe { Net::uninit_cache() };

    for _ in 0..200000 {
        let idx = exotic::rand::random::<usize>() % TRN_IMAGES;

        let i = unsafe {
            std::mem::transmute::<_, StaticVecRef<f32, { 28 * 28 }>>(
                &trn_img[idx * 28 * 28] as *const f32,
            )
        };

        //println!("{:2.2?}", i.matrix::<Blas, 28, 28>());

        let o = net.predict_buffered(i, &mut buffer)?;

        *unsafe {
            std::mem::transmute::<_, MutStaticVecRef<f32, { 28 * 28 }>>(&mut buffer[0] as *mut f32)
        } = *i;

        let y = onehot::<f32, 10>(trn_lbl[idx] as usize);
        let dy = moo![|n| o[n] - y[n]; 10];

        let cost = o
            .moo_ref()
            .iter()
            .zip(y.iter())
            .map(|(o, y)| (o - y).powi_(2))
            .sum::<f32>()
            .abs();

        if cost.is_nan() {
            panic!("cost is nan");
        }

        //if epoch % 100 == 0 {
        println!(
            "cost: {cost:.3}, lbl: {}, out: {o:.3?}",
            trn_lbl[idx] as usize
        );
        //}
        //println!("\n{y:?}");

        //println!(
        //    "{:.3?}",
        //    net.l0.weights.matrix::<Blas, 10, 784>().as_transposed()
        //);

        net.backpropagate(&buffer, dy)?;
    }

    Ok(())
}
