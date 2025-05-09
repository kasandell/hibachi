mod model;
mod token_output_stream;

use std::io;
use std::io::Write;
use std::sync::Arc;
use candle_core::{Tensor, Device};
use hibachi::autoregressive::{
    AutoregressiveBatchInference,
    AutoregressiveBatcher
};
use futures::stream::StreamExt;
use crate::model::Model;
use crate::token_output_stream::TokenOutputStream;


#[tokio::main]
async fn main() {
    let model = Model::new(Some(0.2), Some(25), Some(0.7));
    let stop_token = model.stop_token();
    let padding_token = model.padding_token();
    let tokenizer = model.tokenizer();

    let bi = Arc::new(AutoregressiveBatchInference::<Tensor, 2>::new(
        model,
        &stop_token,
        &padding_token,
    ));

    let handles = (0..2).map(|_e| {
        let bic = bi.clone();
        let tc = tokenizer.clone();
        let mut token_stream = TokenOutputStream::new(tc.clone());

        tokio::spawn(async move {
            async {
                let device = Device::Cpu;
                let sentence = tc.encode("Echo: 'hi'. Output: ", true).expect("ok")
                    .get_ids()
                    .to_vec();
                let toks = Tensor::new(sentence, &device).unwrap();
                let mut it = bic.clone().run(toks).await;
                let mut count = 0;
                let mut output = vec![];
                while let Some(tok) = it.next().await {
                    let value = tok.to_scalar::<u32>().expect("ok");
                    if let Some(out) = token_stream.next_token(value).unwrap() {
                        print!("{}", out);
                        let _ = io::stdout().flush();
                    };
                    output.push(value);
                    count+=1;
                }
                println!("Generated {}", count);
            }.await
        })
    }).collect::<Vec<_>>();  // Collect into Vec to avoid lazy evaluation

    // Wait for all tasks to complete
    for handle in futures::future::join_all(handles).await {
        match handle {
            Ok(_) => {},
            Err(e) => println!("Err joining handle: {:?}", e),
        }
    }
}
