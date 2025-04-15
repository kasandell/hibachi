//! # Hibachi
//!
//! A **hi**gh-performance **batch** tensor processing library for efficiently batching
//! autoregressive model inference across multiple concurrent requests.
//!
//! ## Overview
//!
//! This library provides a flexible and extensible framework for processing
//! autoregressive model inference requests in batches. It's designed to maximize
//! throughput when handling multiple concurrent generation requests by dynamically
//! batching them and efficiently managing computational resources.
//!
//! Key components include:
//!
//! - A tensor abstraction layer supporting various backends
//! - A batched inference engine for autoregressive models
//! - Utilities for tensor manipulation and sequence management
//! - Asynchronous streaming of generated outputs
//!
//! ## Architecture
//!
//! The library is built around several key abstractions:
//!
//! ### Assumptions
//! Regardless of backend used, hibachi reserves two dimensions with special meanings:
//!  - The `0th` dimension is reserved as the batch dimension
//!  - The `1st` dimension is reserved as the sequence dimension
//!  - Tensors may fill in other dimensions
//!
//!
//! ### Backend Traits
//!
//! The `Backend` and `Unsqueezable` traits define the interface that any tensor
//! implementation must satisfy to work with the library. This allows the core
//! batching logic to remain independent of the specific tensor implementation.
//!
//! ### Autoregressive Processing
//!
//! The `Autoregressive` trait defines the interface for models that generate
//! outputs sequentially, while the `AutoregressiveBatcher` trait encapsulates
//! the logic for efficiently batching multiple generation requests.
//!
//! ## Features
//!
//! - **autoregressive** - Enables autoregressive model batching functionality
//! - **candle** - Enables candle backend
//! - **burn** - Enables burn backend
//!
//! ## Implementation Details
//!
//! The library maintains internal state to track active sequences and efficiently
//! allocate batch slots. When sequences complete (by generating a stop token),
//! they are automatically removed from the batch to make room for waiting requests.
//!
//! Tensor operations are abstracted through the `Backend` trait, allowing for
//! different tensor implementations to be used without changing the core batching logic.
//!


mod communication;
mod tensor;

pub mod backend;

/// Constants for client reference
pub use tensor::constant;

#[cfg(feature = "autoregressive")]
pub mod autoregressive;

