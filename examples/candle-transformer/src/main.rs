mod model;

use std::sync::Arc;
use candle_core::{Tensor, Device, DType};
use hibachi::BatchedRegressiveInference;
use hibachi::CandleForward;
use futures::stream::StreamExt;
use hibachi::Batcher;
use crate::model::Model;


#[tokio::main]
async fn main() {
    let model_pre = Model::new(Some(5), None);
    let stop_token = model_pre.stop_token();
    let tokenizer = model_pre.tokenizer();
    let model: Box<dyn CandleForward + Send + Sync> = Box::new(model_pre);

    let bi = Arc::new(BatchedRegressiveInference::<2>::new(
        model,
        stop_token,
    ));

    let handles = (0..5).map(|e| {
        let bic = bi.clone();
        let tc = tokenizer.clone();

        tokio::spawn(async move {
            async {
                let device = Device::Cpu;
                let sentence = tc.encode("HI!", true).expect("ok")
                    .get_ids()
                    .to_vec();
                let toks = Tensor::new(sentence, &device).unwrap();//?.unsqueeze(0)?;
                let mut it = bic.clone().run(toks).await;
                let mut count = 0;
                let mut output = vec![];
                while let Some(tok) = it.next().await {
                    let value = tok.to_vec1::<u32>().expect("ok")[0];
                    //println!("Index {} tok {}", e, value);
                    output.push(value);
                    count+=1;
                }
                println!();
                println!("{}",tc.decode(&*output, true).unwrap());
                println!("Index {} count {}", e, count);
                println!();
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
