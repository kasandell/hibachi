mod core_trait;
#[cfg(feature = "candle")]
mod candle;

#[cfg(feature = "burn")]
mod burn;

pub use core_trait::*;
