//! # Autoregressive Batching
//!
//! A module for efficient batched processing of autoregressive model inference.
//!
//! ## Overview
//!
//! This module provides a framework for optimizing autoregressive model inference
//! by dynamically batching multiple generation requests. Autoregressive models
//! generate output tokens sequentially, with each token depending on previously
//! generated ones. This architecture enables efficient resource utilization
//! when handling multiple concurrent generation requests.
//!
//! ## Key Components
//!
//! * [`Autoregressive`] - A trait for models that support autoregressive generation
//! * [`AutoregressiveBatcher`] - A trait for components that manage batched inference
//! * [`AutoregressiveBatchInference`] - An implementation that provides dynamic batching
//!
//! ## Features
//!
//! - **Dynamic Batching**: Efficiently groups multiple requests into optimal batches
//! - **Resource Management**: Ensures bounded memory usage during inference
//! - **Streaming Results**: Returns generated tokens as they become available
//! - **Efficient Tensor Handling**: Manages padding and sequence management automatically
//! - **Asynchronous Processing**: Non-blocking architecture for concurrent operation
//!
//!
//! # Example
//!
//! ```rust
//! # use std::io;
//! use hibachi::{
//!     autoregressive::AutoregressiveBatchInference,
//!     autoregressive::Autoregressive,
//!     backend::Backend,
//!     backend::Unsqueezable
//! };
//! use hibachi::autoregressive::AutoregressiveBatcher;
//! use candle_core::{DType, Device, Tensor};
//! use futures::StreamExt;
//! use async_trait::async_trait;
//!
//!
//!
//! pub struct Model {}
//!
//! impl Model {
//!     pub fn new() -> Self  {
//!         Self {
//!         }
//!     }
//! }
//!
//! #[async_trait]
//! impl Autoregressive<Tensor> for Model {
//!
//!     async fn forward(&self, tensor: Tensor) -> Tensor {
//!         // Extract the dimensions we need
//!         let batch_size = tensor.dims()[0];
//!         Tensor::ones(&[batch_size], tensor.dtype(), tensor.device()).unwrap()
//!     }
//! }
//!
//! # #[tokio::main]
//! # async fn main() -> io::Result<()> {
//!
//! let device = Device::Cpu;
//! // will be of rank + 1
//! let stop_token = Tensor::ones(
//!     &[1],
//!     DType::U8,
//!     &device
//! ).unwrap();
//!
//! let padding_token = Tensor::zeros(
//!     &[1],
//!     DType::U8,
//!     &device
//! ).unwrap();
//!
//! // Create model and tokens
//! let model = Model::new();
//!
//! // Create the inference engine with batch size 4
//! let inference = AutoregressiveBatchInference::<Tensor, 4>::new(
//!     model,
//!     &stop_token,
//!     &padding_token
//! );
//!
//! let input = Tensor::zeros(
//!     &[3],
//!     DType::U8,
//!     &device,
//!  ).expect("creates start token");
//!
//! // Use the engine to process generation requests
//! let mut stream = inference.run(input).await;
//!
//! // Process generated tokens as they become available
//! while let Some(token) = stream.next().await {
//!     println!("Generated token: {}", token);
//! }
//!
//! # Ok(())
//! # }
//! ```
//!
//! ## Implementation Details
//!
//! The batching system maintains an active tensor containing all sequences currently
//! being processed. As new tokens are generated, they are appended to this tensor and
//! streamed back to their respective requesters. When a sequence completes (by generating
//! a stop token), it is removed from the batch to make room for new requests.
//!
//! The implementation handles sequences of varying lengths through dynamic padding and
//! automatically manages resource allocation to ensure efficient processing.

mod batcher;
mod core_trait;
mod item_stream;
mod handler;
mod queue_item;

pub use core_trait::*;
pub use batcher::AutoregressiveBatchInference;
