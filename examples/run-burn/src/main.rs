mod model;

use std::sync::Arc;
use tokio;
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::tensor::{Shape, Tensor};
use burn_batcher::BatchedRegressiveInference;
use burn_batcher::Forward;
use futures::stream::{Stream, StreamExt};
use batcher::r#async::batcher::Batcher;
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
    println!("Init bi!");
    let handles = (0..20).into_iter().map(|e| {
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
                println!("{} count {}", e, count);
            }.await
        });
        return h;
    }).collect::<Vec<_>>();  // Collect into Vec to avoid lazy evaluation

    // Wait for all tasks to complete
    for handle in futures::future::join_all(handles).await {
        match handle {
            Ok(_) => println!("Fin!"),
            Err(_) => println!("Err!"),
        }
    }
}


/*
let model = Box::new(Model::<Backend>::new());
let stop_token = Tensor::<Backend, 1>::from_data(
    [1.0, 1.0, 1.0],
    &device,
);
let batched_inference = BatchedRegressiveInference::<Backend, 2, 1>::new(model, stop_token, 64);

let run_tok = Tensor::<Backend, 2>::from_data(
    [[0.0, 0.0, 0.0]],
    &device,
);
let mut toks = batched_inference.run(run_tok).await;
while let Some(token)  = toks.next().await {
    println!("{}", token);
}
 */

//let tensor = Arc::new(Mutex::new(Tensor::<Backend, 3>::zeros(Shape::new([3, 1, 3]), &device)));
/*
let tensor = Tensor::<Backend, 3>::zeros(Shape::new([3, 3, 3]), &device);
//BatchedRegressiveInference::<Backend>::run_inference_loop(model, tensor).await;
let mut toks = Tensor::<Backend, 2>::from_data(
    [
            [1, 1, 1],
            [0, 0, 2],
            [1, 1, 1],
        ],
    &device
);
let stop_token = Tensor::<Backend, 1>::from_data(
    [1, 1, 1],
    &device
);
//BatchedRegressiveInference::<Backend, 1>::where_equals_stop_token_vec(&toks, &stop_token).await;
let mut sequence_lengths = [3; 3];
let mut in_use = [true; 3];
let mut updated = BatchedRegressiveInference::<Backend, 1>::concat_output(tensor, toks.clone()).await;
let mut senders: [Option<mpsc::UnboundedSender<Tensor<Backend, 1>>>; 3] = [const {None}; 3];
let active_count = Arc::new(Mutex::new(3));
println!("{:?}", updated.shape());
BatchedRegressiveInference::<Backend, 3>::update_state_for_output(
    &mut updated,
    &mut toks,
    &stop_token,
    &mut sequence_lengths,
    &mut in_use,
    &mut senders,
    active_count.clone()
).await;
println!("{}", updated);
println!("{:?}", sequence_lengths);
println!("{:?}", in_use);
println!("{}", *active_count.lock().await);
*/
