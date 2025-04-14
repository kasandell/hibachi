//! The burn implementation for backend provision.
//! Since burn Tensor are constrained with const generics, we must macro apply the given
//! lower rank and core operations
use std::fmt::{Debug, Display};
use super::{Backend, LowerRankedTensorOps};
use burn::prelude::{Tensor, Backend as BurnBackend};
use burn::tensor::{BasicOps};

macro_rules! impl_lower_ranked_tensor_ops {
    ($d:literal) => {
        impl <B, K> LowerRankedTensorOps for Tensor<B, $d, K>
        where B: BurnBackend,
        K: BasicOps<B> + 'static {
            type Unsqueezed = Tensor<B, {$d + 1}, K>;

            fn unsqueeze(&self, dim: usize) -> Self::Unsqueezed {
                self.clone().unsqueeze_dim::<{$d + 1}>(dim)
            }
        }
    }
}

macro_rules! impl_core_tensor_ops {
    ($d:literal) => {
        impl <B, K> Backend for Tensor<B, $d, K>
        where B: BurnBackend,
        K: BasicOps<B> + 'static {

            fn shape(&self) -> Vec<usize> {
                self.shape().dims.to_vec()
            }

            fn cat(tensors: &[Self], dim: usize) -> Self {
                let owned: Vec<_> = tensors.iter().map(|e| e.clone()).collect();
                Tensor::cat(
                    owned, dim
                )
            }

            fn eq(&self, other: &Self) -> impl Backend {
                let x = self.clone().equal(other.clone());
                x
            }

            fn vectorize_dim(&self, dim: usize) -> Vec<Self> {
                let sizes = self.dims();
                self.clone().chunk(sizes[dim], dim)
            }

            fn slice(&self, dimension: usize, seq_start_idx: usize, len: usize) -> Self {
                self.clone().narrow(dimension, seq_start_idx, len)
            }

            fn idx_where_all_true(&self, dim: usize) -> Vec<usize> {

                let dims = self.dims();
                if dim >= dims.len() {
                    return vec![]; // Invalid dimension
                }

                let dim_size = dims[dim];
                let mut result = Vec::new();

                // For each index along the specified dimension
                for i in 0..dim_size {
                    // Extract the slice at this index
                    let slice = self.clone().narrow(dim, 0, 1);//slice_at_dim(dim, i);
                    let all_tensor = slice.all();
                    let is_all_true = all_tensor.into_scalar(); // Or similar method to extract the scalar value

                    if is_all_true {
                        result.push(i);
                    }
                }
                result
            }

            fn pop(&self, dim: usize, index: usize) -> Self {
                let dim_size = self.dims()[dim];

                if index >= dim_size {
                    panic!("Index {} is out of bounds for dimension {} with size {}", index, dim, dim_size);
                }

                // If idx is 0, we just take everything after idx
                if index == 0 {
                    return self.clone().narrow(dim, 1, dim_size - 1);
                }

                // If idx is the last element, we take everything before idx
                if index == dim_size - 1 {
                    return self.clone().narrow(dim, 0, dim_size - 1);
                }

                // Otherwise, we need to concatenate the parts before and after idx
                let first_part = self.clone().narrow(dim, 0, index);
                let second_part = self.clone().narrow(dim, index + 1, dim_size - index - 1);
                Tensor::cat(vec![first_part, second_part], dim)
            }

            fn repeat(&self, dim: usize, times: usize) -> Self {
                self.clone().repeat_dim(dim, times)
            }
        }
    }
}

impl_lower_ranked_tensor_ops!(1);
impl_lower_ranked_tensor_ops!(2);
impl_lower_ranked_tensor_ops!(3);
impl_lower_ranked_tensor_ops!(4);
impl_lower_ranked_tensor_ops!(5);
impl_lower_ranked_tensor_ops!(6);
impl_lower_ranked_tensor_ops!(7);
impl_lower_ranked_tensor_ops!(8);

impl_core_tensor_ops!(1);
impl_core_tensor_ops!(2);
impl_core_tensor_ops!(3);
impl_core_tensor_ops!(4);
impl_core_tensor_ops!(5);
impl_core_tensor_ops!(6);
impl_core_tensor_ops!(7);
impl_core_tensor_ops!(8);
impl_core_tensor_ops!(9);
