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
        output_len,
        layers,
        cache_len,
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

    let buffer_output: Vec<_> = (0..layers.0.len())
        .map(|n| {
            let ofset = {
                let lt: Vec<_> = (0..n)
                    .map(|n| {
                        let t = &layers.0[n];
                        quote! {#t::O_LEN}
                    })
                    .collect();

                quote! {{#(#lt +)* #input_len}}
            };

            let layer = &layers.0[n];
            //quote! { unsafe{ std::mem::transmute::<_, MutStaticVecRef::<#float_type, {#layer::O_LEN}>>(o.as_mut_ptr().add(#ofset)) } }
            quote! { unsafe{ std::mem::transmute::<_, MutStaticVecRef::<#float_type, {#layer::O_LEN}>>(&mut o.mut_moo_ref()[#ofset] as *mut #float_type) } }
            //quote! { unsafe{ o.mut_static_slice_unchecked::<{#layer::O_LEN}>(#ofset) } }
        })
        .collect();

    let buffered_ret = format_ident!("l{}", layers.0.len() - 1);

    quote! {
        fn predict(&mut self, i: impl exotic::slas::prelude::StaticVec<#float_type, #input_len>) -> Result<#output_type>{
            #(
                let #layer_names = self.#layer_names.predict(#layer_inputs)?;
            )*

            Ok(#ret)
        }

        fn predict_buffered(&mut self, i: impl exotic::slas::prelude::StaticVec<#float_type, #input_len>, o: &mut impl StaticVec<#float_type, #cache_len>)
        -> Result<StaticVecRef::<#float_type, #output_len>>{
            #(
                **(#buffer_output) = self.#layer_names.predict(#layer_inputs)?;
                let #layer_names = #buffer_output;
            )*
            Ok(#buffered_ret)
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

    let layer_names: Vec<_> = layer_names(layers.0.len()).iter().cloned().rev().collect();

    let layer_inputs: Vec<_> = (0..layers.0.len())
        .rev()
        .map(|n| {
            let ofset = if n == 0{quote!{0}}else{
                let lt: Vec<_> = (0..n-1)
                    .map(|n| {
                        let t = &layers.0[n];
                        quote! {#t::O_LEN}
                    })
                    .collect();

                quote! {{#(#lt +)* #input_len }}
            };

            let layer = &layers.0[n];
            //quote! { unsafe{ std::mem::transmute::<_, StaticVecRef::<#float_type, {#layer::I_LEN}>>(i.as_mut_ptr().add(#ofset)) } }
            quote! { unsafe{ std::mem::transmute::<_, StaticVecRef::<#float_type, {#layer::I_LEN}>>(&i.moo_ref()[#ofset] as *const #float_type) } }
        })
        .collect();

    let layer_deltas: Vec<_> = (0..layers.0.len())
        .rev()
        .map(|n| {
            if n == layers.0.len() - 1 {
                format_ident!("gradient")
            } else {
                let n = n + 1;
                format_ident!("l{n}")
            }
        })
        .collect();

    let ret = format_ident!("l0");

    quote! {
        fn backpropagate(&mut self, mut i: impl exotic::slas::prelude::StaticVec<#float_type, #cache_len>, gradient: impl exotic::slas::prelude::StaticVec<#float_type, #output_len>) -> Result<[#float_type; #input_len]>{
            #(
                let #layer_names = self.#layer_names.backpropagate(#layer_inputs, #layer_deltas)?;
            )*

            Ok(#ret)
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
    let float_type = model.float_type.clone();
    let cache_len = model.cache_len.clone();

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
                fn default<T: Default>()->T{T::default()}
                Self{#(
                    #layer_names : #layer_init,
                )*}
            }

            unsafe fn uninit_cache() -> [#float_type; #cache_len]{
                unsafe{ core::mem::MaybeUninit::uninit().assume_init() }
            }
        }
    };

    quote! {
        #def
        #impl_model
    }
    .into()
}
