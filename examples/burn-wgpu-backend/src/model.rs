use std::marker::PhantomData;
use burn::prelude::{Backend, Tensor};
use hibachi::Autoregressive;
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
impl <B> Autoregressive for Model<B>
where B: Backend {
    type Sequence = Tensor<B, 2>;
    type Output = Tensor<B, 1>;

    fn forward(&self, tensor: Self::Sequence) -> Self::Output {
        // Extract the dimensions we need
        let batch_size = tensor.shape().dims[0];
        //let token_size = tensor.shape().dims[2];
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
