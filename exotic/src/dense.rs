use slas::backends::operations::MatrixMul;

use crate::*;

#[derive(Clone, Copy)]
pub struct DenseLayer<
    T: Float,
    B: Backend<T> + DotProduct<T, DotOutput = T>,
    const I_LEN: usize,
    const O_LEN: usize,
> where
    [(); O_LEN * I_LEN]:,
{
    pub weights: [T; O_LEN * I_LEN],
    pub biasies: [T; O_LEN],
    pub lr: T,
    backend: B,
}

impl<
        T: Float + Display,
        B: Backend<T> + DotProduct<T, DotOutput = T>,
        const I_LEN: usize,
        const O_LEN: usize,
    > DenseLayer<T, B, I_LEN, O_LEN>
where
    [(); O_LEN * I_LEN]:,
{
    pub fn random(lr: T) -> Self {
        let xavier = || -> T {
            (random::<T>() - num!(0.5)) * (T::_2 / (T::from_f64((O_LEN + I_LEN) as f64)))
        };

        let mut buffer: Self = unsafe { std::mem::MaybeUninit::uninit().assume_init() };
        buffer.backend = B::default();
        buffer.weights.iter_mut().for_each(|n| *n = xavier());
        buffer.biasies.iter_mut().for_each(|n| *n = xavier());
        buffer.lr = lr;
        buffer
    }
}

impl<
        T: Float,
        B: Backend<T> + DotProduct<T, DotOutput = T>,
        const I_LEN: usize,
        const O_LEN: usize,
    > LayerTy for DenseLayer<T, B, I_LEN, O_LEN>
where
    [(); O_LEN * I_LEN]:,
{
    type Gradient = [T; I_LEN];
    type Output = [T; O_LEN];
}

macro_rules! impl_dense {
    ($T: ty) => {
        impl<
                B: Backend<$T> + DotProduct<$T, DotOutput = $T> + MatrixMul<$T>,
                const I_LEN: usize,
                const O_LEN: usize,
            > Layer<$T, I_LEN, O_LEN> for DenseLayer<$T, B, I_LEN, O_LEN>
        where
            [(); O_LEN * I_LEN]:,
        {
            fn predict(&mut self, i: impl StaticVec<$T, I_LEN>) -> Result<[$T; O_LEN]> {
                Ok(*self
                    .weights
                    .moo_ref()
                    .matrix::<B, O_LEN, I_LEN>()
                    .vector_mul(i.moo_ref())
                    .moo_owned()
                    .add(self.biasies.moo_ref()))
            }

            fn backpropagate(
                &mut self,
                i: impl StaticVec<$T, I_LEN>,
                gradient: impl StaticVec<$T, O_LEN>,
            ) -> Result<[$T; I_LEN]> {
                let mut buffer = [num!(0); I_LEN];
                let gradient = gradient.moo_ref();
                let input = i.moo_ref();
                let mut weights = self.weights.mut_moo_ref().matrix::<B, O_LEN, I_LEN>();

                for j in 0..O_LEN {
                    self.biasies[j] -= gradient[j] * self.lr;

                    for i in 0..I_LEN {
                        weights[(j, i)] -= gradient[j] * input[i] * self.lr;
                        buffer[i] += weights[(j, i)] * gradient[j];
                    }
                }

                Ok(buffer)
            }
        }
    };
}

impl_dense!(f32);
impl_dense!(f64);
