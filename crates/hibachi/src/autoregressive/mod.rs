//! # Autoregressive Batch
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
//! ## Example Usage
//!
//! ```no_run
//! use hibachi::autoregressive::{Autoregressive, AutoregressiveBatcher, AutoregressiveBatchInference};
//! use candle_core::{DType, Device, Tensor};
//! use futures::StreamExt;
//!
//! // Create your autoregressive model
//! let model = MyModel::new();
//!
//! let device = Device::Cpu;
//! // Define stop and padding tokens of rank 1 higher than needed (rank 1 for scalars)
//! let stop_token = Tensor::ones(
//!         &[1],
//!         DType::U8,
//!         &device
//!     ).unwrap();
//!
//!     let padding_token = Tensor::zeros(
//!         &[1],
//!         DType::U8,
//!         &device
//!     ).unwrap();
//!
//! // Create the batched inference engine with a max batch size of 16
//! let inference_engine = AutoregressiveBatchInference::<Tensor, 16>::new(
//!     model,
//!     &stop_token,
//!     &padding_token
//! );
//!
//! // Process a generation request
//! async fn generate(engine: &impl AutoregressiveBatcher<Tensor, Tensor>, input: Tensor) {
//!     let mut stream = engine.run(input).await;
//!
//!     // Consume generated tokens as they become available
//!     while let Some(token) = stream.next().await {
//!         // Process the token
//!         println!("Generated token: {:?}", token);
//!     }
//! }
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
