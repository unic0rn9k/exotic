<div align="center">

# Exotic

*Exotical Deep Learning*

[![Crates.io](https://img.shields.io/crates/v/exotic?logo=rust)](https://crates.io/crates/exotic)
[![GitHub Workflow Status](https://img.shields.io/github/workflow/status/unic0rn9k/exotic/Tests?label=tests&logo=github)](https://github.com/unic0rn9k/exotic/actions/workflows/rust.yml)
[![Coverage Status](https://coveralls.io/repos/github/unic0rn9k/exotic/badge.svg?branch=master)](https://coveralls.io/github/unic0rn9k/exotic?branch=master)
[![Docs](https://img.shields.io/docsrs/exotic/latest?logo=rust)](https://docs.rs/exotic/latest/exotic/)
[![Donate on paypal](https://img.shields.io/badge/paypal-donate-1?logo=paypal&color=blue)](https://www.paypal.com/paypalme/unic0rn9k/5usd)

<img align="right" src="logo.png" height="150px"/>
  
</div>

Are you tired of modern deep learning frameworks making you choose between performance and being eassy to work with?
Exotic might just be what you're looking for!


Built with [unic0rn9k/slas](https://github.com/unic0rn9k/slas) (It might be a good idea to take a look at the installation section in the slas readme)

## Basic example

``` rust
use exotic::prelude::*;
use exotic_macro::*;
use slas_backend::*;

model! {(
    derive: [Copy, Clone],
    name: "ExampleNet",
    layers: [
        ("DenseLayer::<f32, Blas, 4, 2>", "DenseLayer::random(0.1)"),
        ("Softmax::<f32, 2>", "default()")
    ],
    float_type: "f32",
    input_len: 4,
    output_len: 2
)}

let mut net = ExampleNet::new();

let y = moo![f32: 0, 1];
let i = moo![f32: 0..4];
let mut buffer = unsafe { ExampleNet::uninit_cache() };

for _ in 0..2000 {
    let o = net.predict_buffered(&i, &mut buffer)?;

    let dy = moo![|n| o[n] - y[n]; 2];

    net.backpropagate(&mut buffer, dy)?;
}

let o = net.predict(i)?;

let cost = o
    .moo_ref()
    .iter()
    .zip(y.iter())
    .map(|(o, y)| (o - y).powi_(2))
    .sum::<f32>()
    .abs();

assert!(cost < 0.0001, "Found {o:?}, expecteed {y:?} (cost: {cost})");
```
