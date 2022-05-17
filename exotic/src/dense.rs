use crate::*;
use slas::backends::operations::MatrixMul;

#[derive(Clone, Copy)]
pub struct DenseLayer<T: Float, B: Backend<T>, const I_LEN: usize, const O_LEN: usize>
where
    [(); O_LEN * I_LEN]:,
{
    pub weights: [T; O_LEN * I_LEN],
    pub biasies: [T; O_LEN],
    pub lr: T,
    backend: B,
}

impl<T: Float + Display, B: Backend<T>, const I_LEN: usize, const O_LEN: usize>
    DenseLayer<T, B, I_LEN, O_LEN>
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

#[derive(Clone)]
pub struct DenseHeapLayer<T: Float, B: Backend<T>, const I_LEN: usize, const O_LEN: usize> {
    pub weights: Vec<T>,
    pub biasies: Vec<T>,
    pub lr: T,
    backend: B,
}

impl<T: Float + Display, B: Backend<T>, const I_LEN: usize, const O_LEN: usize>
    DenseHeapLayer<T, B, I_LEN, O_LEN>
where
    [(); O_LEN * I_LEN]:,
{
    pub fn random(lr: T) -> Self {
        let xavier = || -> T {
            (random::<T>() - num!(0.5)) * (T::_2 / (T::from_f64((O_LEN + I_LEN) as f64)))
        };

        let backend = B::default();
        let lr = lr;
        let mut weights = vec![T::_0; O_LEN * I_LEN];
        let mut biasies = vec![T::_0; O_LEN];
        weights.iter_mut().for_each(|n| *n = xavier());
        biasies.iter_mut().for_each(|n| *n = xavier());
        Self {
            weights,
            biasies,
            lr,
            backend,
        }
    }
}

macro_rules! impl_dense {
    ($T:ty: $layer_ty: ident $($w_len: expr)?) => {
        impl<B: Backend<$T> + MatrixMul<$T>, const I_LEN: usize, const O_LEN: usize>
            Layer<$T, I_LEN, O_LEN, O_LEN> for $layer_ty<$T, B, I_LEN, O_LEN>
        where
            [(); O_LEN * I_LEN]:,
        {
            type Gradient = [$T; I_LEN];

            fn predict(
                &mut self,
                i: impl StaticVec<$T, I_LEN>,
                buffer: &mut impl StaticVec<$T, O_LEN>,
            ) -> Result<()> {
                self.weights
                    .moo_ref::<$($w_len)?>()
                    .matrix_ref::<B, O_LEN, I_LEN>()
                    .vector_mul(i.moo_ref())
                    .moo_ref()
                    .add_into(self.biasies.moo_ref(), buffer.mut_moo_ref());
                Ok(())
            }

            /// Here buffer is shadowed, so a NullVec can safely be passed.
            fn backpropagate(
                &mut self,
                i: impl StaticVec<$T, I_LEN>,
                _buffer: &impl StaticVec<$T, O_LEN>,
                gradient: impl StaticVec<$T, O_LEN>,
            ) -> Result<[$T; I_LEN]> {
                let mut buffer = [num!(0); I_LEN];
                let gradient = gradient.moo_ref();
                let input = i.moo_ref();
                let mut weights = self
                    .weights
                    .mut_moo_ref::<$($w_len)?>()
                    .matrix_mut_ref::<B, O_LEN, I_LEN>();

                //let weights = weights.as_transposed_mut();

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

impl_dense!(f32: DenseLayer);
impl_dense!(f32: DenseHeapLayer {O_LEN*I_LEN});
impl_dense!(f64: DenseLayer);
impl_dense!(f64: DenseHeapLayer {O_LEN*I_LEN});
