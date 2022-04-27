#![feature(generic_arg_infer)]

#[cfg(feature = "mnist_example")]
mod mnist_example;

#[cfg(feature = "mnist_example")]
fn main() {
    mnist_example::main().unwrap();
}
