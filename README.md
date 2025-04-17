# Hibachi

> Efficient batched inference tensor models

[![Crates.io](https://img.shields.io/crates/v/hibachi.svg)](https://crates.io/crates/hibachi)
[![Documentation](https://docs.rs/hibachi/badge.svg)](https://docs.rs/hibachi)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

![Hibachi](hibachi.png)

**Hibachi** is a Rust library for efficient batched inference with autoregressive and feedforward models. It dynamically groups multiple generation requests into batches, manages tensor operations, and streams results back to clients as they become available.

## Key Features

- **Dynamic Batching** - Optimizes resource utilization by batching requests
- **Asynchronous Processing** - Non-blocking architecture built on Tokio
- **Stream-Based API** - Tokens are streamed back to clients as they're generated
- **Backend Agnostic** - Works with any tensor library that implements the `Backend` trait, includes implementations for `Candle` and `Burn` backends (max `Burn` tensor rank of `9`)
- **Memory Efficient** - Manages tensor padding, concatenation, and cleanup

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
hibachi = {version = "0.1.0", features = ["candle", "autoregressive"] }# burn, feedforward flags available as well
tokio = { version = "1", features = ["full"] }
```


## Early Stage Notice
This package is still in its early stages. Until `1.x` releases, hibachi reserves the right to break interfaces. Though we will try our best not to,
this packaage is in its infancy, and may need to be adjusted as it grows. 

## Quick Start

```rust
use hibachi::autoregressive::{Autoregressive, AutoregressiveBatcher, AutoregressiveBatchInference};
use hibachi::backend::{Backend, Unsqueezable};
use std::sync::Arc;
use candle_core::{Tensor, Device, DType};

// 1. Implement the Autoregressive trait for your model
struct MyModel { /* ... */ }

#[async_trait]
impl Autoregressive<Tensor> for MyModel {
    async fn forward(&self, tensor: Tensor) -> Tensor {
        // Implement your model's forward pass
    }
}

// 3. Create the batched inference engine
#[tokio::main]
async fn main() {
    // Initialize model
    let model = MyModel::new();

    let device = Device::Cpu;
    // will be of rank + 1
    let stop_token = Tensor::ones(&[1], DType::U8, &device).unwrap();

    let padding_token = Tensor::zeros(&[1], DType::U8, &device).unwrap();
    
    // Create inference engine with max batch size of 16
    let engine = AutoregressiveBatchInference::<Tensor, 16>::new(
        model,
        &stop_token,
        &padding_token
    );
    
    // Process requests
    let input = Tensor::arange(2., 5., &device);
    let mut stream = engine.run(input).await;
    
    // Stream results
    while let Some(token) = stream.next().await {
        println!("Generated token: {:?}", token);
    }
}
```

## Architecture

Tensor Batch consists of several core components:

1. **Backend Abstraction**
    - Traits that define required tensor operations
    - Enables support for different tensor libraries

2. **Autoregressive Models**
    - Interface for models that predict the next token based on previous tokens
    - Supports variable batch and sequence dimensions

3. **Feedforward Models**
   - Interface for models that predict outputs in one shot
   - Supports variable batch dimensions

4. **Batching Engine**
    - Dynamically manages multiple generation requests
    - Handles tensor padding, concatenation, and state management
    - Streams generated tokens back to clients

5. **Communication Layer**
    - Asynchronous channels for efficient token streaming
    - Proper error handling and resource cleanup

## Advanced Usage

### Custom Tensor Backends

To use with a custom tensor library, implement the `Backend` and `Unsqueezable` traits:

```rust
use hibachi::backend::{Backend, Unsqueezable};

impl Backend for MyCustomTensor {
    fn shape(&self) -> Vec<usize> { /* ... */ }
    fn clone(&self) -> Self { /* ... */ }
    // ... implement other required methods
}

impl Unsqueezable for MyCustomTensor {
    type Unsqueezed = MyCustomTensorHigherDim;
    fn unsqueeze(&self, dim: usize) -> Self::Unsqueezed { /* ... */ }
}
```

### Custom Autoregressive Models

Implement the `Autoregressive` trait for your model:

```rust
use hibachi::autoregressive::Autoregressive;
use async_trait::async_trait;

#[async_trait]
impl Autoregressive<Tensor> for MyTransformerModel {
    async fn forward(&self, tensor: <Tensor as Unsqueezable>::Unsqueezed) -> Tensor {
        // Your transformer forward logic here
        // Input shape: (batch, seq, ...)
        // Output shape: (batch, ...)
    }
}
```

### Custom Feedforward Models

Implement the `Autoregressive` trait for your model:

```rust
use hibachi::autoregressive::Autoregressive;
use async_trait::async_trait;

#[async_trait]
impl Feedforward<Tensor, Tensor> for MyTransformerModel {
    async fn forward(&self, tensor: Tensor) -> Tensor {
        // Your feedforward forward logic here
        // Input shape: (batch,  ...)
        // Output shape: (batch, ...)
    }
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## TODOs
- [ ] High throughput batching (provide some way to split model by layers / blocks)
- 
