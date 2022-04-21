extern crate proc_macro;
use proc_macro::TokenStream;
use quote::*;
use ron::de::from_str;
use serde::Deserialize;
use std::fs;
use syn::Index;

macro_rules! bail{
    ($($t: tt)*) => {{
        let err = format!($($t)*);
        return quote!{compile_error!(#err);}.into()
    }}
}

#[derive(Deserialize)]
struct Layout {
    name: String,
    float_type: String,
    derive: Option<Vec<String>>,
    embedable: Option<String>,
    layers: Vec<(String, Vec<String>)>,
    cached: bool,
    serialization: Option<bool>,
}

#[proc_macro]
pub fn model(input: TokenStream) -> TokenStream {
    let layout: Layout = match from_str(&input.to_string()) {
        Err(e) => bail!("Error parsing macro: {}", e),
        Ok(ok) => ok,
    };
    if !layout.cached {
        bail!("Non-Cached networks not implemented yet")
    }

    let model_name = format_ident!("{}", layout.name);
    let float_ty = format_ident!("{}", layout.float_type);

    let derive = layout
        .derive
        .unwrap_or(vec![])
        .iter()
        .map(|d| format_ident!("{}", d))
        .collect::<Vec<_>>();

    let layers = layout
        .layers
        .iter()
        .map(|l| {
            let ty = l.0.split("::").map(|p| format_ident!("{}", p));
            let generics = l.1.iter();
            quote! {
                #(#ty::)*<#(#generics),*>
            }
        })
        .collect::<Vec<_>>();

    let fields = layers.iter().enumerate().map(|(n, l)| {
        let field = format_ident!("l{}", n);
        quote!(#field : #l)
    });

    let last = layers.last().unwrap();
    let first = &layers[0];
    let o_size = quote! {{ #last::O_LEN  }};
    let i_size = quote! {{ #first::I_LEN }};

    let cache_ty = quote!((#(<#layers as exotic::LayerTy>::Output),*));

    let mut predict_body = quote!(i);
    for l in 0..layout.layers.len() {
        let l = format_ident!("l{}", l);
        predict_body = quote!(self.#l.predict(#predict_body).context(format!("Error in layer {} while predicting", stringify!(#l)))?);
    }

    let write_cache = (0..layout.layers.len())
        .map(|l| {
            if l == 0 {
                quote!(cache.0 = self.l0.predict(i)?;)
            } else {
                let prev_l = Index::from(l - 1);
                let ll = format_ident!("l{}", l);
                let l = Index::from(l);
                quote! {
                    cache.#l = self.#ll.predict(cache.#prev_l)?;
                }
            }
        })
        .collect::<Vec<_>>();

    let mut backprop_body = quote!(delta);
    for l in 0..layout.layers.len() - 1 {
        let l = layout.layers.len() - l - 1;
        let ll = format_ident!("l{}", l);
        let l = Index::from(l - 1);
        backprop_body = quote!(self.#ll.backpropagate(cache.#l, #backprop_body).context(format!("Error in layer {} while backpropagating", stringify!(#ll)))?);
    }
    backprop_body = quote!(self.l0.backpropagate(i, #backprop_body).context(format!("Error in input layer while backpropagating"))?);

    let o_layer = Index::from(layout.layers.len() - 1);
    let cache = (0..layout.layers.len()).map(|n| Index::from(n));

    let serialization = if layout.serialization.unwrap_or(false) {
        let l: Vec<_> = (0..layout.layers.len())
            .map(|l| format_ident!("l{}", l))
            .collect();
        quote! {
            pub fn serialize_to(&self, writer: &mut dyn std::io::Write) -> Result<()>{
                use exotic::serialization::Serialization;
                #(
                    for n in &self.#l.serialize(){
                        writer.write(n)?;
                    }
                );*
                writer.flush()?;
                Ok(())
            }
            pub fn deserialize(&mut self, bytes: &mut dyn std::iter::Iterator<Item=u8>){
                use exotic::serialization::Serialization;
                #(self.#l.deserialize(bytes));*
            }
        }
    } else {
        quote!()
    };

    let impl_default = if let Some(embed) = layout.embedable {
        let embed = fs::read(embed).expect("Unable to read serialized model for embedding");
        quote! {
            impl #model_name{
                fn embed(&mut self){
                    const SOURCE: &[u8] = &[#(#embed),*];
                    self.deserialize(&mut SOURCE.iter().map(|u|*u))
                }
            }
        }
    } else {
        quote!()
    };

    TokenStream::from(quote!(//println!("{}", stringify!(
        #[derive(#(#derive),*)]
        struct #model_name{
            #(#fields,)*
            cache: #cache_ty,
        }

        impl #model_name{
            pub fn new_cache() -> #cache_ty{
                (#(<#layers as exotic::LayerTy>::Output::zero()),*)
            }
            pub fn cache(&mut self, i: impl exotic::slas::StaticVec<#float_ty, #i_size>) -> Result<exotic::slas::StaticVecRef<'a, <Self as exotic::LayerTy>::Output, #o_size>>{
                let cache = &mut self.cache;
                #(#write_cache)*
                Ok(self.cache.#o_layer.moo_ref())
            }
            pub fn backpropagate_from_cache(&mut self, i: impl exotic::slas::StaticVec<#float_ty, #i_size>, delta: impl exotic::slas::StaticVec<#float_ty, #o_size>) -> Result<<Self as exotic::LayerTy>::Gradient>{
                let cache = &self.cache;
                Ok(#backprop_body)
            }
            pub fn printcache(&self){
                #(
                    println!("cache {} : {}", #cache, self.cache.#cache.to_ref());
                )*
            }
            #serialization
        }

        impl exotic::LayerTy for #model_name{
            type Output = [#float_ty; #o_size];
            type Gradient = [#float_ty; #i_size];
        }

        impl exotic::Layer<#i_size, #o_size> for #model_name{
            fn backpropagate(&mut self, i: impl exotic::slas::StaticVec<#float_ty, #i_size>, delta: impl exotic::slas::StaticVec<#o_size>) -> Result<Self::Gradient>{
                let cache = &mut self.cache;
                #(#write_cache)*
                Ok(#backprop_body)
            }

            fn predict(&mut self, i: impl exotic::slas::StaticVec<#float_ty, #i_size>) -> Result<Self::Output>{
                Ok(#predict_body)
            }
        }
        #impl_default
    ))
}
