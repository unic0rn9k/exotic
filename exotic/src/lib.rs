#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use std::fmt::Display;

pub use anyhow;
use anyhow::*;
use rand;

fn random<T: Float>() -> T {
    T::from_f64(rand::random())
}

use slas::{backends::operations::DotProduct, prelude::*};

/// This trait needs to be implemented to expose needed types to the proc macro, without knowing
/// the input and output size of the layer.
pub trait LayerTy {
    type Gradient;
    type Output;
}

/// Trait for Layer in deep learning model.
pub trait Layer<T: Float, const I_LEN: usize, const O_LEN: usize>: LayerTy
where
    <Self as LayerTy>::Gradient: StaticVec<T, I_LEN>,
    <Self as LayerTy>::Output: StaticVec<T, O_LEN>,
{
    const O_LEN: usize = O_LEN;
    const I_LEN: usize = I_LEN;

    fn predict(&mut self, i: impl StaticVec<T, I_LEN>) -> Result<Self::Output>;
    fn backpropagate(
        &mut self,
        i: impl StaticVec<T, I_LEN>,
        gradient: impl StaticVec<T, O_LEN>,
    ) -> Result<Self::Gradient>;
}

#[macro_use]
pub mod activation;
pub mod dense;
pub use slas;

#[cfg(test)]
mod test {
    use crate::*;

    #[test]
    fn no_macro() {
        use std::marker::PhantomData;

        struct Net {
            l0: dense::DenseLayer<f32, slas_backend::Blas, 4, 2>,
            l1: activation::Softmax<f32, 2>,
        }

        impl Net {
            fn predict(&mut self, i: &[f32; 4]) -> [f32; 2] {
                let hidden = self.l0.predict(i).unwrap();
                self.l1.predict(hidden).unwrap()
            }

            fn backpropagate(&mut self, i: &[f32; 4], delta: &[f32; 2]) {
                let hidden = self.l0.predict(i).unwrap();
                let hidden_delta = self.l1.backpropagate(hidden, delta).unwrap();
                self.l0.backpropagate(i, hidden_delta).unwrap();
            }
        }

        let mut net = Net {
            l0: dense::DenseLayer::random(0.1),
            l1: activation::Softmax(PhantomData),
        };

        let y = moo![f32: 0., 1.];
        let i = moo![f32: 0..4];

        for _ in 0..5000 {
            let o = net.predict(&i);
            let dy = moo![|n|->f32 { o[n] - y[n] }; 2];
            net.backpropagate(&i, dy.slice());
        }

        let o = net.predict(&i);
        let cost = o
            .moo_ref()
            .iter()
            .zip(y.iter())
            .map(|(o, y)| (o - y).powi_(2))
            .sum::<f32>()
            .abs();

        assert!(cost < 0.0001, "Found {o:?}, expecteed {y:?} (cost: {cost})");
    }
}
