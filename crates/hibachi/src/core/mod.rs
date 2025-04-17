//! # Core Hibachi Framework
//!
//! The core module provides fundamental components for efficient batch processing
//! of inference operations. It implements a robust architecture for handling batched
//! requests with support for both synchronous and asynchronous execution patterns.
//!
//! ## Module Structure
//!
//! * [`batch`] - Implements batch processing loops and algorithms for efficient inference execution.
//!   Contains the core logic for dynamic batching, request queueing, and execution scheduling.
//!
//! * [`handler`] - Defines traits and interfaces for processing batches of inference requests.
//!   The `BatchHandler` trait provides the abstraction layer for implementing model-specific
//!   inference logic.
//!
//! * [`worker`] - Provides background worker management for long-running batch operations.
//!   The `BatchWorkerHandle` type offers a convenient interface for spawning, notifying,
//!   and gracefully shutting down background workers.
//!
//! ## Usage
//!
//! The framework is designed to support different inference patterns:
//!
//! - Feed-forward models: Complete processing of inputs in a single pass
//! - Autoregressive models: Process inputs incrementally over multiple iterations
//!
//! The modular design allows for customization while providing efficient batching
//! mechanisms to maximize throughput and resource utilization.
//!
pub mod worker;
pub mod batch;
pub mod handler;
