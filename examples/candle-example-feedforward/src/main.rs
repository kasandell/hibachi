mod model;

use std::sync::Arc;
use candle_core::{Tensor, Device, DType};
use hibachi::feedforward::{FeedforwardBatchInference, FeedforwardBatcher};
use futures::stream::StreamExt;
use crate::model::Model;

#[tokio::main]
async fn main() {
    let model = Model::new();
    let device = Device::Cpu;

    let bi = Arc::new(FeedforwardBatchInference::<Tensor, Tensor, 10>::new(
        model,
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
                let tensor = it.await.unwrap();
                println!("Index {} shape {:?}", e, tensor.shape());
            }.await
        })
    }).collect::<Vec<_>>();

    // Wait for all tasks to complete
    for handle in futures::future::join_all(handles).await {
        match handle {
            Ok(_) => {},
            Err(e) => println!("Err joining handle: {:?}", e),
        }
    }
}

