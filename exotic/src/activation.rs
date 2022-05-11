use crate::Layer;
use anyhow::*;
use slas::prelude::*;
use std::marker::PhantomData;

/// Simple macro for creating simple activation functions.
/// ### Example
/// Example implementation of an approximation of the sigmoid activation function.
/// ```
/// use exotic::*;
/// use slas::prelude::*;
/// use std::marker::PhantomData;
/// use anyhow::*;
///
/// activation!{[f32,f64]:
///    Sigmoid =
///        |x| x/(x + 1.),
///        |x| 1. / (x.powi(2) + 2. * x + 1.)
/// }
/// ```
#[macro_export]
macro_rules! activation {
    ([$($T: ty),*]: $name: ident = $f: expr, $d: expr $(,)?) => {
        #[derive(Clone, Copy, Default)]
        pub struct $name<T: Float, const LEN: usize>(pub PhantomData<T>);

        $(
        impl<const LEN: usize> Layer<$T, LEN, LEN, LEN> for $name<$T, LEN> {
            type Gradient = [$T; LEN];

            fn predict(&mut self, i: impl StaticVec<$T, LEN>, buffer: &mut impl StaticVec<$T, LEN>) -> Result<()> {
                let buffer = buffer.mut_moo_ref();
                let fun: fn($T)->$T = $f;
                for n in 0..LEN {
                    buffer[n] = fun(i.moo_ref()[n])
                }
                Ok(())
            }

            /// Here buffer is shadowed, so a NullVec can safely be passed.
            fn backpropagate(
                &mut self,
                i: impl StaticVec<$T, LEN>,
                _buffer: &impl StaticVec<$T, LEN>,
                gradient: impl StaticVec<$T, LEN>,
            ) -> Result<[$T; LEN]> {
                let fun: fn($T)->$T = $d;
                let mut buffer = [num!(0); LEN];
                for n in 0..LEN {
                    buffer[n] = fun(i.moo_ref()[n]) * gradient.moo_ref()[n]
                }
                Ok(buffer)
            }
        }

        //impl<const LEN: usize> Serialization<0> for $name<$T, LEN> {
        //    fn serialize_into(&self, _: &mut [[u8; FLOAT_BYTES]]) {}
        //    fn deserialize(&mut self, _: &mut dyn std::iter::Iterator<Item = u8>) {}
        //}
    )*};
    (T: $name: ident = $f: expr, $d: expr $(,)?) => {
        #[derive(Clone, Copy, Default)]
        pub struct $name<T: Float, const LEN: usize>(pub PhantomData<T>);

        impl<T: Float, const LEN: usize> Layer<T, LEN, LEN, LEN> for $name<T, LEN> {
            type Gradient = [T; LEN];

            fn predict(&mut self, i: impl StaticVec<T, LEN>, buffer: &mut impl StaticVec<T, LEN>) -> Result<()> {
                let buffer = buffer.mut_moo_ref();
                let fun: fn(T)->T = $f;
                for n in 0..LEN {
                    buffer[n] = fun(i.moo_ref()[n])
                }
                Ok(())
            }

            /// Here buffer is shadowed, so a NullVec can safely be passed.
            fn backpropagate(
                &mut self,
                i: impl StaticVec<T, LEN>,
                _buffer: &impl StaticVec<T, LEN>,
                gradient: impl StaticVec<T, LEN>,
            ) -> Result<[T; LEN]> {
                let fun: fn(T)->T = $d;
                let mut buffer = [num!(0); LEN];
                for n in 0..LEN {
                    buffer[n] = fun(i.moo_ref()[n]) * gradient.moo_ref()[n]
                }
                Ok(buffer)
            }
        }
    };
}

pub fn sigmoid<T: Float>(x: T) -> T {
    T::_1 / (T::_1 + (-x).exp_())
}

activation! {
T: Sigmoid =
        |x| sigmoid(x),
        |x| sigmoid(T::_1 - x),
}

activation! {
[f32, f64]: Tanh =
        |x| x.tanh(),
        |x| (-x.tanh().powi(2)) + 1.,
}

// x \cdot \sigma(x)
// \sigma(x) * (1 + x * (1 - \sigma(x)))
activation! {
T: Swish =
        |x| x / ((-x).exp_() + T::_1),
        |x| {
            let sig = sigmoid(x);
            sig * (T::_1 + x * (T::_1 - sig))
        }
}

activation! {
[f32, f64]: Relu =
        |x| x.max(0.),
        |x| ((x>0.) as u8).into(),
}

activation! {
T: None =
        |x| x,
        |_| T::_1,
}

activation! {
T: Square =
        |x| x.powi_(2),
        |x| x * T::_2,
}

#[derive(Clone, Copy, Default)]
pub struct Softmax<T: Float, const LEN: usize>(pub PhantomData<T>);

impl<T: Float + std::iter::Sum, const LEN: usize> Layer<T, LEN, LEN, LEN> for Softmax<T, LEN> {
    type Gradient = [T; LEN];
    fn predict(
        &mut self,
        i: impl StaticVec<T, LEN>,
        buffer: &mut impl StaticVec<T, LEN>,
    ) -> Result<()> {
        let buffer = buffer.mut_moo_ref();
        let sum: T = (0..LEN).map(|n| i.moo_ref()[n].exp_()).sum();
        for n in 0..LEN {
            buffer[n] = i.moo_ref()[n].exp_() / sum
        }
        Ok(())
    }
    fn backpropagate(
        &mut self,
        i: impl StaticVec<T, LEN>,
        _buffer: &impl StaticVec<T, LEN>,
        gradient: impl StaticVec<T, LEN>,
    ) -> Result<[T; LEN]> {
        let mut buffer = [num!(0); LEN];
        let sum: T = (0..LEN).map(|n| i.moo_ref()[n].exp_()).sum();
        for n in 0..LEN {
            buffer[n] = i.moo_ref()[n].exp_() / sum;
            buffer[n] = buffer[n] * (T::_1 - buffer[n]) * gradient.moo_ref()[n]
        }
        Ok(buffer)
    }
}
