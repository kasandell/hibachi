mod model;

use std::sync::Arc;
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::tensor::Tensor;
use hibachi::autoregressive::{
AutoregressiveBatcher,
AutoregressiveBatchInference
};
use futures::stream::StreamExt;
use crate::model::Model;

type Backend = Wgpu;

#[tokio::main]
async fn main() {
    let device = WgpuDevice::DefaultDevice;
    let model = Model::new();


    let stop_token = Tensor::<Backend, 1>::from_data(
        [1,],
        &device,
    );
    let padding_token = Tensor::<Backend, 1>::from_data(
        [0,],
        &device,
    );

    let bi = Arc::new(AutoregressiveBatchInference::<Tensor<Backend, 1>, 10>::new(
        model,
        &stop_token,
        &padding_token
    ));

    let handles = (0..100).map(|e| {
        let device = device.clone();
        let bic = bi.clone();

        tokio::spawn(async move {
            async {
                let toks = Tensor::<Backend, 1>::from_data(
                        [2, 2, 0],
                    &device,
                );
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
