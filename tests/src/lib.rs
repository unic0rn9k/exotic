use exotic_macro::*;

use exotic::activation::Softmax;
use exotic::anyhow::*;
use exotic::dense::*;
use exotic::slas::prelude::*;
use exotic::Layer;

use slas_backend::*;
use std::marker::PhantomData;

model! {(
    derive: [Copy, Clone],
    name: "Net",
    layers: [
        ("DenseLayer::<f32, Blas, 2, 3>", "DenseLayer::random(0.01)"),
        ("Softmax::<f32, 3>", "Softmax(PhantomData)")
    ],
    float_type: "f32",
    input_len: 2,
    output_len: 3
)}

//        struct Net {
//            l0: dense::DenseLayer<f32, slas_backend::Blas, 4, 2>,
//            l1: activation::Softmax<f32, 2>,
//        }
//
//        impl Net {
//            fn predict(&mut self, i: &[f32; 4]) -> [f32; 2] {
//                let hidden = self.l0.predict(i).unwrap();
//                self.l1.predict(hidden).unwrap()
//            }
//
//            fn backpropagate(&mut self, i: &[f32; 4], delta: &[f32; 2]) {
//                let hidden = self.l0.predict(i).unwrap();
//                let hidden_delta = self.l1.backpropagate(hidden, delta).unwrap();
//                self.l0.backpropagate(i, hidden_delta).unwrap();
//            }
//        }

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let net = crate::Net::new();
    }
}
