#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use std::{fmt::Display, mem::MaybeUninit};

pub use anyhow;
use anyhow::*;
pub use rand;

pub fn random<T: Float>() -> T {
    T::from_f64(rand::random())
}

use slas::prelude::*;

/// Trait for Layer in deep learning model.
pub trait Layer<T: Float, const I_LEN: usize, const O_LEN: usize, const BUFFER_LEN: usize> {
    const O_LEN: usize = O_LEN;
    const I_LEN: usize = I_LEN;
    type Gradient: StaticVec<T, I_LEN>;

    fn predict(
        &mut self,
        i: impl StaticVec<T, I_LEN>,
        buffer: &mut impl StaticVec<T, BUFFER_LEN>,
    ) -> Result<()>;
    fn backpropagate(
        &mut self,
        i: impl StaticVec<T, I_LEN>,
        buffer: &impl StaticVec<T, BUFFER_LEN>,
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
