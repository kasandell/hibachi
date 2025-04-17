//! # Tensor Backend
//!
//! This module provides a unified interface for different tensor backends,
//! allowing for batching apis to work in a backend-agnostic manner

//! ## Feature Flags
//!
//! The module uses feature flags to conditionally compile support for different backends:
//!
//! - `candle`: Enables support for the Candle tensor library
//! - `burn`: Enables support for the Burn tensor library
//!
//! ## Usage
//!
//! Users of this crate can work with tensors in a backend-agnostic way by:
//!
//! 1. Importing the traits ([`Backend`], [`Unsqueezable`])
//! 2. Writing code against these trait interfaces
//! 3. Enabling the appropriate feature flag for their desired backend
//!
//! This allows for easy switching between tensor backends without changing application code.

mod core_trait;

#[cfg_attr(docsrs, doc(cfg(feature = "candle")))]
#[cfg(feature = "candle")]
/// Candle tensor backend implementation.
///
/// This module is only available when the `candle` feature flag is enabled.
/// It provides an implementation of the [`Backend`] and [`Unsqueezable`] traits
/// for Candle's `Tensor` type.
///
/// The implementation wraps candle-core's tensor operations to match the
/// expected behavior of our tensor backend abstraction.
pub mod candle;

#[cfg_attr(docsrs, doc(cfg(feature = "burn")))]
#[cfg(feature = "burn")]
/// Burn tensor backend implementation.
///
/// This module is only available when the `burn` feature flag is enabled.
/// It provides an implementation of the [`Backend`] and [`Unsqueezable`] traits
/// for Burn's tensor types.
///
/// Burn uses a different approach to tensors than some other libraries,
/// with stronger compile-time guarantees through its type system.
pub mod burn;


// Re-export the core traits for convenient imports
pub use core_trait::*;


#[cfg(test)]
/// Mock tensor implementation.
///
/// Operates on simple vector tensors
pub(crate) mod mock_tensor;
