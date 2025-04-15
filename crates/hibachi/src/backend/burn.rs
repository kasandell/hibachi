//! Implementation of Backend and Unsqueezable traits for burn's Tensor types.
//!
//! This module provides implementations of the Backend and Unsqueezable traits
//! for Burn's tensor types. Since Burn uses const generics to track tensor dimensions
//! at compile time, the implementation uses macros to generate the appropriate
//! trait implementations for tensors with different numbers of dimensions.
//!
//! ## Implementation Details
//!
//! - `impl_lower_ranked_tensor_ops!` - Implements the Unsqueezable trait for tensors
//!   with dimensions 1 through 8, allowing them to be unsqueezed to tensors with one
//!   more dimension.
//!
//! - `impl_core_tensor_ops!` - Implements the Backend trait for tensors with dimensions
//!   1 through 9, providing the core tensor operations.
//!
//! ## Supported Dimensions
//!
//! The implementation supports tensors with between 1 and 9 dimensions. Tensors with more
//! dimensions than this will not implement the required traits.
//!
//! ## Type Parameters
//!
//! - `B` - The Burn backend (CPU, CUDA, etc.)
//! - `K` - The kind of tensor operations supported
//! - The const generic parameter represents the number of dimensions

use super::{Backend, Unsqueezable};
use burn::prelude::{Tensor, Backend as BurnBackend};
use burn::tensor::{BasicOps};

/// Implements the Unsqueezable trait for tensors with the specified number of dimensions.
///
/// This allows a tensor with D dimensions to be unsqueezed to a tensor with D+1 dimensions.
/// The implementation uses Burn's unsqueeze_dim method and handles the type-level dimension tracking.
macro_rules! impl_lower_ranked_tensor_ops {
    ($d:literal) => {
        impl <B, K> Unsqueezable for Tensor<B, $d, K>
        where B: BurnBackend,
        K: BasicOps<B> + 'static {
            type Unsqueezed = Tensor<B, {$d + 1}, K>;

            /// Adds a dimension of size 1 at the specified position.
            ///
            /// # Parameters
            ///
            /// * `dim` - The position at which to insert the new dimension (0-indexed)
            ///
            /// # Returns
            ///
            /// A tensor with one more dimension than the input tensor
            ///
            /// # Panics
            ///
            /// May panic if `dim` is greater than the current number of dimensions.
            fn unsqueeze(&self, dim: usize) -> Self::Unsqueezed {
                self.clone().unsqueeze_dim::<{$d + 1}>(dim)
            }
        }
    }
}

/// Implements the Backend trait for tensors with the specified number of dimensions.
///
/// This provides the core tensor operations for Burn tensors, adapting Burn's native
/// operations to match the expected behavior of our Backend trait.
macro_rules! impl_core_tensor_ops {
    ($d:literal) => {
        impl <B, K> Backend for Tensor<B, $d, K>
        where B: BurnBackend,
        K: BasicOps<B> + 'static {

            /// Returns the shape of this tensor as a vector of dimension sizes.
            ///
            /// # Returns
            ///
            /// * `Vec<usize>` - A vector containing the size of each dimension
            fn shape(&self) -> Vec<usize> {
                self.shape().dims.to_vec()
            }

            /// Concatenates multiple tensors along the specified dimension.
            ///
            /// # Parameters
            ///
            /// * `tensors` - Slice of tensors to concatenate
            /// * `dim` - The dimension along which to concatenate (0-indexed)
            ///
            /// # Returns
            ///
            /// A new tensor containing the concatenated data
            ///
            /// # Panics
            ///
            /// May panic if:
            /// * `dim` is out of bounds for any of the tensors
            /// * The tensors have incompatible shapes outside of the concatenation dimension
            fn cat(tensors: &[Self], dim: usize) -> Self {
                let owned: Vec<_> = tensors.iter().map(|e| e.clone()).collect();
                Tensor::cat(owned, dim)
            }

            /// Performs element-wise equality comparison against another tensor.
            ///
            /// # Parameters
            ///
            /// * `other` - The tensor to compare against (must have the same shape)
            ///
            /// # Returns
            ///
            /// A new boolean tensor where each element is `true` if the corresponding
            /// elements in the two input tensors are equal, `false` otherwise.
            ///
            /// # Panics
            ///
            /// May panic if the shapes of `self` and `other` are not identical.
            fn eq(&self, other: &Self) -> impl Backend {
                let x = self.clone().equal(other.clone());
                x
            }

            /// Splits a tensor into a vector of slices along the specified dimension.
            ///
            /// # Parameters
            ///
            /// * `dim` - The dimension along which to split (0-indexed)
            ///
            /// # Returns
            ///
            /// A vector of tensors, each representing a slice along the specified dimension
            ///
            /// # Panics
            ///
            /// May panic if `dim` is out of bounds for this tensor.
            fn vectorize_dim(&self, dim: usize) -> Vec<Self> {
                let sizes = self.dims();
                self.clone().chunk(sizes[dim], dim)
            }

            /// Extracts a contiguous slice from the tensor along the specified dimension.
            ///
            /// # Parameters
            ///
            /// * `dimension` - The dimension to slice along (0-indexed)
            /// * `seq_start_idx` - Starting index for the slice
            /// * `len` - Length of the slice to extract
            ///
            /// # Returns
            ///
            /// A new tensor containing the extracted slice
            ///
            /// # Panics
            ///
            /// May panic if:
            /// * `dimension` is out of bounds for this tensor
            /// * `seq_start_idx + len` exceeds the size of the specified dimension
            fn slice(&self, dimension: usize, seq_start_idx: usize, len: usize) -> Self {
                self.clone().narrow(dimension, seq_start_idx, len)
            }

            /// Returns indices along a dimension where all values in the corresponding slice are `true`.
            ///
            /// # Parameters
            ///
            /// * `dim` - The dimension to check (0-indexed)
            ///
            /// # Returns
            ///
            /// A vector of indices where all elements in the slice at that index are `true`
            ///
            /// # Panics
            ///
            /// May panic if:
            /// * `dim` is out of bounds for this tensor
            /// * The tensor does not contain boolean data
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
                    let slice = self.clone().narrow(dim, i, 1);
                    let all_tensor = slice.all();
                    let is_all_true = all_tensor.into_scalar(); // Extract the scalar value

                    if is_all_true {
                        result.push(i);
                    }
                }
                result
            }

            /// Removes a single slice at the specified index from the tensor along the given dimension.
            ///
            /// # Parameters
            ///
            /// * `dim` - The dimension from which to remove the slice (0-indexed)
            /// * `index` - The index of the slice to remove
            ///
            /// # Returns
            ///
            /// A new tensor with the specified slice removed
            ///
            /// # Panics
            ///
            /// May panic if:
            /// * `dim` is out of bounds for this tensor
            /// * `index` is out of bounds for the specified dimension
            fn pop(&self, dim: usize, index: usize) -> Self {
                let dim_size = self.dims()[dim];

                if index >= dim_size {
                    panic!("Index {} is out of bounds for dimension {} with size {}",
                          index, dim, dim_size);
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

            /// Repeats the tensor a specified number of times along the given dimension.
            ///
            /// # Parameters
            ///
            /// * `dim` - The dimension along which to repeat (0-indexed)
            /// * `times` - How many times to repeat the tensor
            ///
            /// # Returns
            ///
            /// A new tensor with the repeated data
            ///
            /// # Panics
            ///
            /// May panic if `dim` is out of bounds for this tensor.
            fn repeat(&self, dim: usize, times: usize) -> Self {
                self.clone().repeat_dim(dim, times)
            }
        }
    }
}

// Implement Unsqueezable for tensors with dimensions 1-8
impl_lower_ranked_tensor_ops!(1);
impl_lower_ranked_tensor_ops!(2);
impl_lower_ranked_tensor_ops!(3);
impl_lower_ranked_tensor_ops!(4);
impl_lower_ranked_tensor_ops!(5);
impl_lower_ranked_tensor_ops!(6);
impl_lower_ranked_tensor_ops!(7);
impl_lower_ranked_tensor_ops!(8);

// Implement Backend for tensors with dimensions 1-9
impl_core_tensor_ops!(1);
impl_core_tensor_ops!(2);
impl_core_tensor_ops!(3);
impl_core_tensor_ops!(4);
impl_core_tensor_ops!(5);
impl_core_tensor_ops!(6);
impl_core_tensor_ops!(7);
impl_core_tensor_ops!(8);
impl_core_tensor_ops!(9);

#[cfg(test)]
mod test {
    use super::{Backend, Unsqueezable};
    use burn::backend::ndarray::{NdArray, NdArrayDevice};
    use burn::tensor::{Tensor, Float, TensorData};

    type BurnBackend = NdArray;
    type Device = NdArrayDevice;
    type TensorF<const D: usize> = Tensor<BurnBackend, D, Float>;

    fn create_sequential_tensor<const D: usize>(shape: &[usize]) -> TensorF<D> {
        let size: usize = shape.iter().product();
        let data: Vec<f32> = (1..=size).map(|x| x as f32).collect();
        let d = burn::tensor::TensorData::new(data, shape);
        Tensor::from_data(d, &Device::default())
    }

    #[test]
    fn test_shape() {
        let tensor: TensorF<2> = create_sequential_tensor(&[2, 3]);
        let shape = Backend::shape(&tensor);
        assert_eq!(shape, vec![2, 3]);
    }

    #[test]
    fn test_unsqueeze() {
        let tensor: TensorF<2> = create_sequential_tensor(&[2, 3]);
        let unsqueezed0 = Unsqueezable::unsqueeze(&tensor, 0);
        assert_eq!(Backend::shape(&unsqueezed0), vec![1, 2, 3]);

        let unsqueezed1 = Unsqueezable::unsqueeze(&tensor, 1);
        assert_eq!(Backend::shape(&unsqueezed1), vec![2, 1, 3]);

        let unsqueezed2 = Unsqueezable::unsqueeze(&tensor, 2);
        assert_eq!(Backend::shape(&unsqueezed2), vec![2, 3, 1]);
    }

    #[test]
    fn test_cat() {
        let tensor1: TensorF<2> = create_sequential_tensor(&[2, 3]);
        let tensor2: TensorF<2> = Tensor::from_data(TensorData::new((7..=12).map(|x| x as f32).collect(), &[2, 3]), &Device::default());

        let cat_dim0 = Backend::cat(&[tensor1.clone(), tensor2.clone()], 0);
        assert_eq!(Backend::shape(&cat_dim0), vec![4, 3]);

        let cat_dim1 = Backend::cat(&[tensor1, tensor2], 1);
        assert_eq!(Backend::shape(&cat_dim1), vec![2, 6]);
    }

    #[test]
    fn test_vectorize_dim() {
        let tensor: TensorF<2> = create_sequential_tensor(&[2, 3]);
        let vectorized0 = Backend::vectorize_dim(&tensor, 0);
        assert_eq!(vectorized0.len(), 2);
        assert_eq!(Backend::shape(&vectorized0[0]), vec![1, 3]);

        let tensor2: TensorF<2> = create_sequential_tensor(&[2, 3]);
        let vectorized1 = Backend::vectorize_dim(&tensor2, 1);
        assert_eq!(vectorized1.len(), 3);
        assert_eq!(Backend::shape(&vectorized1[0]), vec![2, 1]);
    }

    #[test]
    fn test_slice() {
        let tensor: TensorF<3> = create_sequential_tensor(&[2, 3, 2]);
        let slice0 = Backend::slice(&tensor, 0, 0, 1);
        assert_eq!(Backend::shape(&slice0), vec![1, 3, 2]);

        let slice1 = Backend::slice(&tensor, 1, 1, 2);
        assert_eq!(Backend::shape(&slice1), vec![2, 2, 2]);

        let slice2 = Backend::slice(&tensor, 2, 0, 1);
        assert_eq!(Backend::shape(&slice2), vec![2, 3, 1]);
    }

    #[test]
    fn test_idx_where_all_true() {
        let bool_tensor1: TensorF<2> = Tensor::from_data(
            [ [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]], &Device::default());

        eprintln!("{:?}", bool_tensor1.shape());
        let indices0 = Backend::idx_where_all_true(&bool_tensor1, 0);
        assert_eq!(indices0, vec![0]);

        let indices1 = Backend::idx_where_all_true(&bool_tensor1, 1);
        assert_eq!(indices1.len(), 0);

        let bool_tensor2: TensorF<2> = Tensor::from_data(
            [ [1.0, 0.0], [1.0, 0.0], [1.0, 1.0]], &Device::default());
        let indices2 = Backend::idx_where_all_true(&bool_tensor2, 0);
        assert_eq!(indices2, vec![2]);
        let indices3 = Backend::idx_where_all_true(&bool_tensor2, 1);
        assert_eq!(indices3, vec![0]);
    }

    #[test]
    fn test_pop() {
        let tensor: TensorF<2> = create_sequential_tensor(&[2, 3]);
        let popped0 = Backend::pop(&tensor, 0, 0);
        assert_eq!(Backend::shape(&popped0), vec![1, 3]);

        let tensor: TensorF<2> = create_sequential_tensor(&[2, 3]);
        let popped1 = Backend::pop(&tensor, 1, 1);
        assert_eq!(Backend::shape(&popped1), vec![2, 2]);

        let tensor: TensorF<2> = create_sequential_tensor(&[2, 3]);
        let popped2 = Backend::pop(&tensor, 1, 2);
        assert_eq!(Backend::shape(&popped2), vec![2, 2]);
    }

    #[test]
    fn test_repeat() {
        let tensor: TensorF<2> = Tensor::from_data([[1.0, 2.0, 3.0]], &Device::default());
        let repeated0 = Backend::repeat(&tensor, 0, 2);
        assert_eq!(Backend::shape(&repeated0), vec![2, 3]);

        let tensor2: TensorF<2> = Tensor::from_data([[1.0, 2.0], [3.0, 4.0]], &Device::default());
        let repeated1 = Backend::repeat(&tensor2, 1, 3);
        assert_eq!(Backend::shape(&repeated1), vec![2, 6]);
    }

    #[test]
    #[should_panic(expected = "Index 3 is out of bounds for dimension 0 with size 2")]
    fn test_pop_out_of_bounds() {
        let tensor: TensorF<2> = create_sequential_tensor(&[2, 3]);
        let _ = Backend::pop(&tensor, 0, 3);
    }

    #[test]
    fn test_tensor_3d() {
        let tensor: TensorF<3> = create_sequential_tensor(&[2, 3, 2]);
        assert_eq!(Backend::shape(&tensor), vec![2, 3, 2]);

        let unsqueezed = Unsqueezable::unsqueeze(&tensor, 1);
        assert_eq!(Backend::shape(&unsqueezed), vec![2, 1, 3, 2]);

        let sliced = Backend::slice(&tensor, 0, 1, 1);
        assert_eq!(Backend::shape(&sliced), vec![1, 3, 2]);

        let vectorized = Backend::vectorize_dim(&tensor, 0);
        assert_eq!(vectorized.len(), 2);
        assert_eq!(Backend::shape(&vectorized[0]), vec![1, 3, 2]);
    }
}
