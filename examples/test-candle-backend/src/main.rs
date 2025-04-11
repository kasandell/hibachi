mod model;

use std::sync::Arc;
use candle_core::{Tensor, Device, DType};
use hibachi_core::{Autoregressive, BatchedRegressiveInference};
use futures::stream::StreamExt;
use hibachi_core::AutoregressiveBatcher;
use crate::model::Model;

type Tensor1D = Tensor;
type Tensor2D = Tensor;

#[tokio::main]
async fn main() {
    let model = Model::new();


    let device = Device::Cpu;
    // will be of rank + 1
    let stop_token = Tensor::ones(
        &[1, 2, 4, 6],
        DType::U8,
        &device
    ).unwrap();

    let padding_token = Tensor::zeros(
        &[1, 2, 4, 6],
        DType::U8,
        &device
    ).unwrap();

    let bi = Arc::new(BatchedRegressiveInference::<Tensor, Model, 100>::new(
        model,
        &stop_token,
        &padding_token
    ));

    let handles = (0..1000).map(|e| {
        let bic = bi.clone();

        tokio::spawn(async move {
            async {
                let device = Device::Cpu;
                let toks = Tensor::zeros(&[3, 2, 4, 6],
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
