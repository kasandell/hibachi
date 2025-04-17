//! # Feedforward Module
//!
//! This module provides infrastructure for efficient batch processing of feed-forward models.
//!
//! ## Overview
//!
//! Feed-forward models process inputs in a single pass without autoregressive behavior.
//! This module enables dynamic batching for such models, allowing multiple inference
//! requests to be processed together for improved throughput and resource utilization.
//!
//! ## Components
//!
//! The module consists of the following key components:
//!
//! - **Traits**:
//!   - [`Feedforward`]: Defines the interface for feed-forward model implementations
//!   - [`FeedforwardBatcher`]: Defines the interface for submitting inference requests
//!
//! ## Usage Example
//!
//! ```ignore
//! use hibachi::feedforward::{Feedforward, FeedforwardBatcher, FeedforwardBatchInference};
//! use async_trait::async_trait;
//!
//! // 1. Define a model that implements the Feedforward trait
//! struct MyModel {
//!     weights: Tensor,
//! }
//!
//! #[async_trait]
//! impl Feedforward<Tensor, Tensor> for MyModel {
//!     async fn forward(&self, input: Tensor) -> Tensor {
//!         // Model implementation that processes batched inputs
//!         input.matmul(&self.weights)
//!     }
//! }
//!
//! // 2. Create a batcher with your model
//! let model = MyModel { weights: Tensor::ones(vec![64, 10]) };
//! let batcher = FeedforwardBatchInference::<Tensor, Tensor, 16>::new(model);
//!
//! // 3. Submit inference requests and receive results
//! async fn run_inference(batcher: &impl FeedforwardBatcher<Tensor, Tensor>) {
//!     let input = Tensor::ones(vec![64]);
//!     let result_item = batcher.run(input).await;
//!
//!     // Await the result
//!     let output = result_item.await.unwrap();
//!     println!("Result shape: {:?}", output.shape());
//! }
//! ```


mod batcher;
mod core_trait;
mod item;
mod queue_item;
mod handler;

pub use core_trait::{
    Feedforward,
    FeedforwardBatcher
};

pub use batcher::FeedforwardBatchInference;
