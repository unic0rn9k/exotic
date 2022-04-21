extern crate proc_macro;
use std::fmt;

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::*;
use ron::from_str;
use serde::de::{self, Deserialize, Visitor};

macro_rules! bail{
    ($($t: tt)*) => {{
        let err = format!($($t)*);
        return quote!{compile_error!(#err);}.into()
    }}
}

struct Ident(String);
struct IdentVisitor;

impl<'de> Visitor<'de> for IdentVisitor {
    type Value = Ident;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("Any identifier")
    }

    fn visit_str<E>(self, value: &str) -> Result<Ident, E>
    where
        E: de::Error,
    {
        Ok(Ident(value.to_string()))
    }
}

impl<'de> Deserialize<'de> for Ident {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let ret = deserializer.deserialize_identifier(IdentVisitor)?;
        Ok(ret)
    }
}

struct Model {
    derive: Vec<proc_macro2::Ident>,
    name: proc_macro2::Ident,
    float_type: proc_macro2::TokenStream,
    input_len: usize,
    output_len: usize,
    cache_len: proc_macro2::TokenStream,
    layers: (Vec<proc_macro2::TokenStream>, Vec<proc_macro2::TokenStream>),
}

#[derive(serde::Deserialize)]
struct ModelRon {
    derive: Vec<Ident>,
    name: String,
    float_type: String,
    input_len: usize,
    output_len: usize,
    layers: Vec<(String, String)>,
}

impl ModelRon {
    fn to_model(self) -> Model {
        Model {
            derive: self
                .derive
                .iter()
                .map(|t| format_ident!("{}", t.0))
                .collect(),
            name: format_ident!("{}", self.name),
            float_type: self.float_type.parse().unwrap(),
            input_len: self.input_len,
            output_len: self.output_len,
            layers: (
                self.layers
                    .iter()
                    .map(|(t, _)| t.parse().unwrap())
                    .collect(),
                self.layers
                    .iter()
                    .map(|(_, f)| f.parse().unwrap())
                    .collect(),
            ),
            cache_len: {
                let lt = self
                    .layers
                    .iter()
                    .map(|(t, _)| t.parse::<TokenStream2>().unwrap());
                let input_len = self.input_len;
                quote! {{#(#lt::O_LEN +)* #input_len}}
            },
        }
    }
}

fn layer_names(len: usize) -> Vec<proc_macro2::Ident> {
    (0..len).map(|n| format_ident!("l{n}")).collect()
}

fn predict(model: &Model) -> TokenStream2 {
    let Model {
        float_type,
        input_len,
        layers,
        ..
    } = model;

    let output_type = layers.0.last().unwrap();
    let output_type = quote! { <#output_type as exotic::LayerTy>::Gradient };

    let layer_names = layer_names(layers.0.len());

    let layer_inputs: Vec<_> = (0..layers.0.len())
        .map(|n| {
            if n == 0 {
                format_ident!("i")
            } else {
                let n = n - 1;
                format_ident!("l{n}")
            }
        })
        .collect();

    let ret = format_ident!("l{}", layers.0.len() - 1);

    quote! {
        fn predict(&mut self, i: impl exotic::slas::prelude::StaticVec<#float_type, #input_len>) -> Result<#output_type>{
            #(
                let #layer_names = self.#layer_names.predict(#layer_inputs)?;
            )*

            Ok(#ret)
        }
    }
}

fn backprop(model: &Model) -> TokenStream2 {
    let Model {
        float_type,
        input_len,
        output_len,
        cache_len,
        layers,
        ..
    } = model;

    let output_type = layers.0.last().unwrap();
    let output_type = quote! { <#output_type as exotic::LayerTy>::Gradient };

    let layer_names = layer_names(layers.0.len());

    let layer_inputs: Vec<_> = (0..layers.0.len())
        .map(|n| {
            if n == 0 {
                format_ident!("i")
            } else {
                let n = n - 1;
                format_ident!("l{n}")
            }
        })
        .collect();

    let layer_deltas: Vec<_> = (0..layers.0.len())
        .rev()
        .map(|n| {
            if n == layers.0.len() - 1 {
                format_ident!("gradient")
            } else {
                format_ident!("l{n}_delta")
            }
        })
        .collect();

    let ret = format_ident!("l{}_delta", layers.0.len() - 1);

    quote! {
        fn backpropagate(&mut self, i: impl exotic::slas::prelude::StaticVec<#float_type, #cache_len>, gradient: impl exotic::slas::prelude::StaticVec<#float_type, #output_len>) -> Result<#output_type>{
            todo!()
        }
    }
}

#[proc_macro]
pub fn model(input: TokenStream) -> TokenStream {
    let model = match from_str::<ModelRon>(&input.to_string()) {
        Err(e) => bail!("Error parsing macro input: {}", e),
        Ok(ok) => ok,
    }
    .to_model();

    let layer_names = layer_names(model.layers.0.len());
    let model_name = model.name.clone();
    let derive = model.derive.clone();
    let layer_types = model.layers.0.clone();
    let layer_init = model.layers.1.clone();

    let def = quote! {
        #[derive(#(#derive),*)]
        struct #model_name{
            #(
                #layer_names: #layer_types,
            )*
        }
    };

    let predict = predict(&model);
    let backprop = backprop(&model);

    let impl_model = quote! {
        impl #model_name{

            #predict
            #backprop

            fn new() -> Self{
                Self{#(
                    #layer_names : #layer_init,
                )*}
            }
        }
    };

    quote! {
        #def
        #impl_model
    }
    .into()
}
