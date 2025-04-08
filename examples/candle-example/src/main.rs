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
    let model: Box<dyn CandleForward + Send + Sync> = Box::new(Model::new());


    let device = Device::Cpu;
    let stop_token = Tensor::ones(
        &[1, ],
        DType::U8,
        &device
    ).unwrap();

    let bi = Arc::new(BatchedRegressiveInference::<10>::new(
        model,
        stop_token,
    ));

    let handles = (0..100).map(|e| {
        let bic = bi.clone();

        tokio::spawn(async move {
            async {
                let device = Device::Cpu;
                let toks = Tensor::zeros(&[3],
                    DType::U8,
                    &device,
                ).expect("creates start token");
                let mut it = bic.clone().run(toks).await;
                let mut count = 0;
                while let Some(_tok) = it.next().await {
                    count+=1;
                }
                println!("Index {} count {}", e, count);
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
