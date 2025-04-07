use std::marker::PhantomData;
use burn::prelude::{Backend, Tensor};
use hibachi::Forward;
use async_trait::async_trait;
use burn::tensor::Shape;
use rand::{thread_rng, Rng};

pub struct Model<B>
    where B: Backend {
    _marker: PhantomData<B>
}

impl <B> Model<B>
    where B: Backend {
    pub fn new() -> Self  {
        Self {
            _marker: PhantomData,
        }

    }
}

#[async_trait]
impl <B> Forward<B> for Model<B>
where B: Backend {
    fn forward(&self, tensor: Tensor<B, 3>) -> Tensor<B, 2> {
        // Extract the dimensions we need
        let batch_size = tensor.shape().dims[0];
        let token_size = tensor.shape().dims[2];

        let mut rng = thread_rng();
        let val = rng.gen_range(0..(batch_size/2));
        let mut zeros = Tensor::<B, 2>::zeros(Shape::new([batch_size, token_size]), &tensor.device());
        let ones = Tensor::<B, 2>::ones(Shape::new([1, token_size]), &tensor.device());
        for _ in 0..val {
            let idx = rng.gen_range(0..batch_size);
            zeros = zeros.slice_assign([idx..idx+1, 0..token_size], ones.clone());
        }
        zeros
    }
}
