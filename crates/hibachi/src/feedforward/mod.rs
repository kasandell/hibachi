//! # Feedforward Module
//!
//! This module provides infrastructure for efficient batch processing of feed-forward models.
//!
//! ```rust
//! # use std::io;
//! use hibachi::feedforward::*;
//! use candle_core::{Tensor, DType, Device};
//! use async_trait::async_trait;
//!
//! struct MyModel {
//!     weights: Tensor,
//! }
//!
//! #[async_trait]
//! impl Feedforward<Tensor, Tensor> for MyModel {
//!     async fn forward(&self, input: Tensor) -> Tensor {
//!         input.matmul(&self.weights).unwrap()
//!     }
//! }
//!
//! # #[tokio::main]
//! # async fn main() -> io::Result<()> {
//! let device = Device::Cpu;
//! let model = MyModel { weights: Tensor::ones(&[64, 10], DType::F16, &device).unwrap() };
//! // Batcher with max batch size of 16
//! let batcher = FeedforwardBatchInference::<Tensor, Tensor, 16>::new(model);
//!
//! // Notice the singular dimension
//! let input = Tensor::ones(&[64], DType::F16, &device).expect("creates start token");
//! let result_item = batcher.run(input).await;
//!
//! let output = result_item.await.unwrap();
//! # Ok(())
//! # }
//! ```
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
