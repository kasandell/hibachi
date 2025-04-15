use burn::prelude::{Backend, Tensor};
use hibachi::autoregressive::{Autoregressive};
use async_trait::async_trait;
use burn::tensor::Shape;
use rand::{thread_rng, Rng};

pub struct Model {}

impl Model {
    pub fn new() -> Self  {
        Self {
        }
    }
}

#[async_trait]
impl <B> Autoregressive<Tensor<B, 1>> for Model
where B: Backend {
    async fn forward(&self, tensor: Tensor<B, 2>) -> Tensor<B, 1> {
        // Extract the dimensions we need
        let batch_size = tensor.shape().dims[0];
        let mut rng = thread_rng();
        let val = rng.gen_range(0..2);
        let mut zeros = Tensor::<B, 1>::zeros(Shape::new([batch_size]), &tensor.device());
        let ones = Tensor::<B, 1>::ones(Shape::new([1]), &tensor.device());
        for _ in 0..val {
            let idx = rng.gen_range(0..batch_size);
            zeros = zeros.slice_assign([idx..idx+1], ones.clone());
        }
        zeros
    }
}
