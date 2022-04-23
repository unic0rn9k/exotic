#[cfg(test)]
mod test {
    use exotic::prelude::*;
    use exotic_macro::*;
    use slas_backend::*;

    #[test]
    fn basic_with_no_macro() {
        use std::marker::PhantomData;

        struct Net {
            l0: DenseLayer<f32, slas_backend::Blas, 4, 2>,
            l1: Softmax<f32, 2>,
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
            l0: DenseLayer::random(0.1),
            l1: Softmax(PhantomData),
        };

        let y = moo![f32: 0, 1];
        let i = moo![f32: 0..4];

        for _ in 0..5000 {
            let o = net.predict(&i);
            let dy = moo![|n| -> f32 { o[n] - y[n] }; 2];
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

    #[test]
    fn basic_with_macro() -> Result<()> {
        model! {(
            derive: [Copy, Clone],
            name: "MacroNet",
            layers: [
                ("DenseLayer::<f32, Blas, 4, 2>", "DenseLayer::random(0.1)"),
                ("Softmax::<f32, 2>", "default()")
            ],
            float_type: "f32",
            input_len: 4,
            output_len: 2
        )}

        let mut net = MacroNet::new();

        let y = moo![f32: 0, 1];
        let i = moo![f32: 0..4];
        let mut buffer = unsafe { MacroNet::uninit_cache() };

        for _ in 0..2000 {
            let o = net.predict_buffered(i, &mut buffer)?;

            let dy = moo![|n| -> f32 { o[n] - y[n] }; 2];

            net.backpropagate(&mut buffer, dy.slice())?;
        }

        let o = net.predict(&i)?;
        let cost = o
            .moo_ref()
            .iter()
            .zip(y.iter())
            .map(|(o, y)| (o - y).powi_(2))
            .sum::<f32>()
            .abs();

        assert!(cost < 0.0001, "Found {o:?}, expecteed {y:?} (cost: {cost})");

        Ok(())
    }
}
