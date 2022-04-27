#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use std::{fmt::Display, mem::MaybeUninit};

pub use anyhow;
use anyhow::*;

pub fn random<T: Float>() -> T {
    T::from_f64(rand::random())
}

use slas::prelude::*;

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

pub fn onehot<T: Float, const LEN: usize>(i: usize) -> [T; LEN] {
    let mut tmp: [T; LEN] = unsafe { MaybeUninit::zeroed().assume_init() };
    tmp[i] = num!(1);
    tmp
}

#[macro_use]
pub mod activation;
pub mod dense;
pub use slas;
pub mod prelude;
