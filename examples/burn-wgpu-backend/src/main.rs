mod model;

use std::sync::Arc;
use tokio;
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::tensor::Tensor;
use hibachi::BatchedRegressiveInference;
use hibachi::Forward;
use futures::stream::{Stream, StreamExt};
use hibachi::Batcher;
use crate::model::Model;

type Backend = Wgpu;

#[tokio::main]
async fn main() {
    let device = WgpuDevice::DefaultDevice;
    let model: Box<dyn Forward<Backend> + Send + Sync> = Box::new(Model::<Backend>::new());


    let stop_token = Tensor::<Backend, 1>::from_data(
        [1, 1, 1],
        &device,
    );

    let mut bi = Arc::new(BatchedRegressiveInference::<Backend, 10>::new(
        model,
        stop_token,
    ));

    let handles = (0..100).into_iter().map(|e| {
        let device = device.clone();
        let bic = bi.clone();
        let h = tokio::spawn(async move {
            async {
                let toks = Tensor::<Backend, 2>::from_data(
                    [
                        [1, 1, 0],
                    ],
                    &device,
                );
                let mut it = bic.clone().run(toks).await;
                let mut count = 0;
                while let Some(tok) = it.next().await {
                    count+=1;
                }
                println!("Index {} count {}", e, count);
            }.await
        });
        return h;
    }).collect::<Vec<_>>();  // Collect into Vec to avoid lazy evaluation

    // Wait for all tasks to complete
    for handle in futures::future::join_all(handles).await {
        match handle {
            Ok(_) => {},
            Err(e) => println!("Err joining handle: {:?}", e),
        }
    }
}
