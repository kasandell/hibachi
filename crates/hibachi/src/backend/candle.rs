//! Implementation of Backend and Unsqueezable traits for candle-core's Tensor type.
//!
//! This module provides the necessary implementations to adapt candle-core's Tensor
//! to work with the generic tensor backend system. Since candle uses a single Tensor type
//! for all tensors (regardless of dimensionality), the implementation is relatively straightforward.

use super::{Backend, Unsqueezable};
use candle_core::{Tensor, Device};

impl Unsqueezable for Tensor {
    type Unsqueezed = Tensor;

    /// Adds a dimension of size 1 at the specified position.
    ///
    /// This implementation directly uses candle's `unsqueeze` method
    /// and unwraps the result, which will panic on failure.
    ///
    /// # Parameters
    ///
    /// * `dim` - The position at which to insert the new dimension (0-indexed)
    ///
    /// # Returns
    ///
    /// A tensor with the new dimension added
    ///
    /// # Panics
    ///
    /// Will panic if `dim` is greater than the current number of dimensions.
    fn unsqueeze(&self, dim: usize) -> Self::Unsqueezed {
        self.unsqueeze(dim).unwrap()
    }
}

impl Backend for Tensor {
    /// Returns the shape of this tensor as a vector of dimension sizes.
    ///
    /// # Returns
    ///
    /// * `Vec<usize>` - A vector containing the size of each dimension
    fn shape(&self) -> Vec<usize> {
        self.dims().to_vec()
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
    /// Will panic if:
    /// * `dim` is out of bounds for any of the tensors
    /// * The tensors have incompatible shapes outside of the concatenation dimension
    fn cat(tensors: &[Self], dim: usize) -> Self {
        Tensor::cat(tensors, dim).unwrap()
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
    /// Will panic if the shapes of `self` and `other` are not broadcastable.
    fn eq(&self, other: &Self) -> impl Backend {
        self.eq(other).unwrap()
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
    /// Will panic if:
    /// * `dim` is out of bounds for this tensor
    /// * The tensor has empty dimensions
    fn vectorize_dim(&self, dim: usize) -> Vec<Self> {
        let dims = self.dims();

        // Ensure tensor has at least one dimension
        if dims.is_empty() {
            panic!("Cannot vectorize tensor with empty dimensions");
        }

        let dim_size = dims[dim];
        let mut result = Vec::with_capacity(dim_size);

        // Extract each slice along the specified dimension
        for i in 0..dim_size {
            // Get a narrow slice containing just the i-th element along dimension dim
            let slice = self.narrow(dim, i, 1).unwrap();
            // Remove the dimension which now has size 1
            let slice = slice.squeeze(dim).unwrap();
            result.push(slice);
        }

        result
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
    /// Will panic if:
    /// * `dimension` is out of bounds for this tensor
    /// * `seq_start_idx + len` exceeds the size of the specified dimension
    fn slice(&self, dimension: usize, seq_start_idx: usize, len: usize) -> Self {
        self.narrow(dimension, seq_start_idx, len)
            .unwrap_or_else(|_| panic!(
                "Failed to slice tensor: dimension={}, start_idx={}, len={}, tensor_dims={:?}",
                dimension, seq_start_idx, len, self.dims()
            ))
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
    /// Will panic if:
    /// * `dim` is out of bounds for this tensor
    /// * Operations on the tensor fail during processing
    fn idx_where_all_true(&self, dim: usize) -> Vec<usize> {
        let dims = self.dims();
        if dim >= dims.len() {
            return vec![]; // Invalid dimension
        }

        let dim_size = dims[dim];
        let mut result = Vec::new();

        for i in 0..dim_size {
            // Extract the slice at index i along dimension dim
            let slice = self.narrow(dim, i, 1).unwrap();

            // Calculate total number of elements in this slice
            let total_elements: usize = slice.dims().iter().product();

            // Flatten the slice to make it easier to check all elements
            let flattened = slice.reshape(&[total_elements]).unwrap();

            // Move to CPU for element-wise checking
            let cpu_tensor = flattened.to_device(&Device::Cpu).unwrap();
            let bool_data = cpu_tensor.to_vec1::<u8>().unwrap();

            // Check if all elements are true (1)
            if bool_data.iter().all(|&v| v == 1) {
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
    /// Will panic if:
    /// * `dim` is out of bounds for this tensor
    /// * `index` is out of bounds for the specified dimension
    fn pop(&self, dim: usize, index: usize) -> Self {
        let dim_size = self.dim(dim).unwrap();

        if index >= dim_size {
            panic!("Index {} is out of bounds for dimension {} with size {}",
                   index, dim, dim_size);
        }

        // Case 1: Remove the first slice
        if index == 0 {
            return self.narrow(dim, 1, dim_size - 1).unwrap();
        }

        // Case 2: Remove the last slice
        if index == dim_size - 1 {
            return self.narrow(dim, 0, dim_size - 1).unwrap();
        }

        // Case 3: Remove a slice from the middle
        // - Get slices before and after the index
        // - Concatenate them
        let first_part = self.narrow(dim, 0, index).unwrap();
        let second_part = self.narrow(dim, index + 1, dim_size - index - 1).unwrap();

        Tensor::cat(&[&first_part, &second_part], dim).unwrap()
    }

    /// Repeats the tensor a specified number of times along the given dimension.
    ///
    /// This implementation uses candle's broadcast_as functionality to efficiently
    /// repeat the tensor.
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
    /// Will panic if:
    /// * `dim` is out of bounds for this tensor
    /// * Broadcasting fails due to incompatible dimensions
    fn repeat(&self, dim: usize, times: usize) -> Self {
        let dims = self.dims();
        if dim >= dims.len() {
            panic!("Dimension {} is out of bounds for tensor with {} dimensions", dim, dims.len());
        }

        // Create a repeat array for the expand operation
        let mut repeats = vec![1; dims.len()];
        repeats[dim] = times;

        // Use repeat instead of broadcast, as broadcast requires dimension size 1
        self.repeat(repeats).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Tensor, Device, DType};

    // Helper function to create a test tensor
    fn create_test_tensor(shape: &[usize], device: &Device) -> Tensor {
        // Create a tensor with sequential values starting from 1
        let size: usize = shape.iter().product();
        let data: Vec<f32> = (1..=size).map(|x| x as f32).collect();
        Tensor::from_vec(data, shape, device).unwrap()
    }

    #[test]
    fn test_shape() {
        let device = Device::Cpu;
        let tensor = create_test_tensor(&[2, 3, 4], &device);

        // Use the trait method shape() instead of dims()
        let shape = Backend::shape(&tensor);
        assert_eq!(shape, vec![2, 3, 4]);
    }

    #[test]
    fn test_unsqueeze() {
        let device = Device::Cpu;
        let tensor = create_test_tensor(&[2, 3], &device);

        // Test unsqueeze at beginning using the trait method
        let unsqueezed0 = Unsqueezable::unsqueeze(&tensor, 0);
        assert_eq!(Backend::shape(&unsqueezed0), vec![1, 2, 3]);

        // Test unsqueeze in middle
        let unsqueezed1 = Unsqueezable::unsqueeze(&tensor, 1);
        assert_eq!(Backend::shape(&unsqueezed1), vec![2, 1, 3]);

        // Test unsqueeze at end
        let unsqueezed2 = Unsqueezable::unsqueeze(&tensor, 2);
        assert_eq!(Backend::shape(&unsqueezed2), vec![2, 3, 1]);
    }

    #[test]
    fn test_cat() {
        let device = Device::Cpu;
        let tensor1 = create_test_tensor(&[2, 3], &device);
        let tensor2 = (tensor1.clone() + 6.0).unwrap();

        // Concatenate along first dimension (dim=0) using the trait method
        let cat_dim0 = Backend::cat(&[tensor1.clone(), tensor2.clone()], 0);
        assert_eq!(Backend::shape(&cat_dim0), vec![4, 3]);

        // Check some values to ensure correct concatenation
        let cat_dim0_data = cat_dim0.to_vec2::<f32>().unwrap();
        // First two rows should be from tensor1
        assert_eq!(cat_dim0_data[0][0], 1.0);
        assert_eq!(cat_dim0_data[1][0], 4.0);
        // Last two rows should be from tensor2
        assert_eq!(cat_dim0_data[2][0], 7.0);
        assert_eq!(cat_dim0_data[3][0], 10.0);

        // Concatenate along second dimension (dim=1)
        let cat_dim1 = Backend::cat(&[tensor1, tensor2], 1);
        assert_eq!(Backend::shape(&cat_dim1), vec![2, 6]);

        // Check some values to ensure correct concatenation
        let cat_dim1_data = cat_dim1.to_vec2::<f32>().unwrap();
        // First three columns of each row should be from tensor1
        assert_eq!(cat_dim1_data[0][0], 1.0);
        assert_eq!(cat_dim1_data[0][2], 3.0);
        // Last three columns of each row should be from tensor2
        assert_eq!(cat_dim1_data[0][3], 7.0);
        assert_eq!(cat_dim1_data[0][5], 9.0);
    }

    #[test]
    fn test_eq() {
        let device = Device::Cpu;
        let tensor1 = create_test_tensor(&[2, 3], &device);
        let tensor2 = create_test_tensor(&[2, 3], &device);
        let tensor3 = (create_test_tensor(&[2, 3], &device) + 1.0).unwrap();

        // Test equality with identical tensor using the trait method
        let eq_result = Backend::eq(&tensor1, &tensor2);
        let idx_where = eq_result.idx_where_all_true(0);
        assert_eq!(idx_where, vec![0, 1]);

        // Test equality with different tensor
        let neq_result = Backend::eq(&tensor1, &tensor3);
        let neq_idx_where = neq_result.idx_where_all_true(0);
        assert_eq!(neq_idx_where, vec![] as Vec<usize>);

    }

    #[test]
    fn test_vectorize_dim() {
        let device = Device::Cpu;
        let tensor = create_test_tensor(&[2, 3], &device);

        // Vectorize along first dimension (dim=0) using the trait method
        let vectorized0 = Backend::vectorize_dim(&tensor, 0);
        assert_eq!(vectorized0.len(), 2);
        assert_eq!(Backend::shape(&vectorized0[0]), vec![3]);
        assert_eq!(Backend::shape(&vectorized0[1]), vec![3]);

        // Check values
        let vec0_data = vectorized0[0].to_vec1::<f32>().unwrap();
        let vec1_data = vectorized0[1].to_vec1::<f32>().unwrap();
        assert_eq!(vec0_data, vec![1.0, 2.0, 3.0]);
        assert_eq!(vec1_data, vec![4.0, 5.0, 6.0]);

        // Vectorize along second dimension (dim=1)
        let vectorized1 = Backend::vectorize_dim(&tensor, 1);
        assert_eq!(vectorized1.len(), 3);
        assert_eq!(Backend::shape(&vectorized1[0]), vec![2]);
        assert_eq!(Backend::shape(&vectorized1[1]), vec![2]);
        assert_eq!(Backend::shape(&vectorized1[2]), vec![2]);

        // Check values
        let vec0_data = vectorized1[0].to_vec1::<f32>().unwrap();
        let vec1_data = vectorized1[1].to_vec1::<f32>().unwrap();
        let vec2_data = vectorized1[2].to_vec1::<f32>().unwrap();
        assert_eq!(vec0_data, vec![1.0, 4.0]);
        assert_eq!(vec1_data, vec![2.0, 5.0]);
        assert_eq!(vec2_data, vec![3.0, 6.0]);
    }

    #[test]
    fn test_slice() {
        let device = Device::Cpu;
        let tensor = create_test_tensor(&[2, 3, 4], &device);

        // Slice along first dimension using the trait method
        let slice0 = Backend::slice(&tensor, 0, 0, 1);
        assert_eq!(Backend::shape(&slice0), vec![1, 3, 4]);

        // Slice along second dimension
        let slice1 = Backend::slice(&tensor, 1, 1, 2);
        assert_eq!(Backend::shape(&slice1), vec![2, 2, 4]);

        // Slice along third dimension
        let slice2 = Backend::slice(&tensor, 2, 0, 2);
        assert_eq!(Backend::shape(&slice2), vec![2, 3, 2]);

        // Check some values
        let slice1_data = slice1.to_vec3::<f32>().unwrap();
        assert_eq!(slice1_data[0][0][0], 5.0);  // Value at [0,1,0] in original tensor
        assert_eq!(slice1_data[0][1][0], 9.0);  // Value at [0,2,0] in original tensor
    }

    #[test]
    fn test_idx_where_all_true() {
        let device = Device::Cpu;

        // Create a boolean tensor with specific pattern
        // [[1, 1], [0, 1], [1, 0]]
        let data = vec![1u8, 1, 0, 1, 1, 0];
        let bool_tensor = Tensor::from_vec(data, &[3, 2], &device).unwrap();

        // Test along dim=0 (rows) using the trait method
        let true_rows = Backend::idx_where_all_true(&bool_tensor, 0);
        assert_eq!(true_rows, vec![0]);  // one row is all true

        // Test along dim=1 (columns)
        let true_cols = Backend::idx_where_all_true(&bool_tensor, 1);
        assert_eq!(true_cols, vec![] as Vec<usize>);  // No column is all true

        // Create another boolean tensor
        // [[1, 0], [1, 0], [1, 1]]
        let data2 = vec![1u8, 0, 1, 0, 1, 1];
        let bool_tensor2 = Tensor::from_vec(data2, &[3, 2], &device).unwrap();

        // Test along dim=0 (rows)
        let true_rows2 = Backend::idx_where_all_true(&bool_tensor2, 0);
        assert_eq!(true_rows2, vec![2] );  // last row is all true

        // Test along dim=1 (columns)
        let true_cols2 = Backend::idx_where_all_true(&bool_tensor2, 1);
        assert_eq!(true_cols2, vec![0]);  // Only first column is all true
    }

    #[test]
    fn test_pop() {
        let device = Device::Cpu;
        let tensor = create_test_tensor(&[2, 3], &device);

        // Pop the first row (index 0, dim 0) using the trait method
        let popped0 = Backend::pop(&tensor, 0, 0);
        assert_eq!(Backend::shape(&popped0), vec![1, 3]);
        let popped0_data = popped0.to_vec2::<f32>().unwrap();
        assert_eq!(popped0_data[0], vec![4.0, 5.0, 6.0]);

        // Pop the middle column (index 1, dim 1)
        let popped1 = Backend::pop(&tensor, 1, 1);
        assert_eq!(Backend::shape(&popped1), vec![2, 2]);
        let popped1_data = popped1.to_vec2::<f32>().unwrap();
        assert_eq!(popped1_data[0], vec![1.0, 3.0]);
        assert_eq!(popped1_data[1], vec![4.0, 6.0]);

        // Pop the last item (index 2, dim 1)
        let popped2 = Backend::pop(&tensor, 1, 2);
        assert_eq!(Backend::shape(&popped2), vec![2, 2]);
        let popped2_data = popped2.to_vec2::<f32>().unwrap();
        assert_eq!(popped2_data[0], vec![1.0, 2.0]);
        assert_eq!(popped2_data[1], vec![4.0, 5.0]);
    }

    #[test]
    fn test_repeat() {
        let device = Device::Cpu;

        // Create a simple 1x3 tensor
        let data = vec![1.0f32, 2.0, 3.0];
        let tensor = Tensor::from_vec(data, &[1, 3], &device).unwrap();

        // Test repeat along dim 0 using the trait method
        let repeated0 = Backend::repeat(&tensor, 0, 4);
        assert_eq!(Backend::shape(&repeated0), vec![4, 3]);
        let repeated0_data = repeated0.to_vec2::<f32>().unwrap();
        // All rows should have the same values
        for i in 0..4 {
            assert_eq!(repeated0_data[i], vec![1.0, 2.0, 3.0]);
        }

        // Create a 2x2 tensor
        let data2 = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor2 = Tensor::from_vec(data2, &[2, 2], &device).unwrap();

        // Test repeat along dim 1
        let repeated1 = Backend::repeat(&tensor2, 1, 3);
        assert_eq!(Backend::shape(&repeated1), vec![2, 6]);
        let repeated1_data = repeated1.to_vec2::<f32>().unwrap();
        assert_eq!(repeated1_data[0], vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
        assert_eq!(repeated1_data[1], vec![3.0, 4.0, 3.0, 4.0, 3.0, 4.0]);
    }

    #[test]
    #[should_panic(expected = "Index 3 is out of bounds for dimension 0 with size 2")]
    fn test_pop_out_of_bounds() {
        let device = Device::Cpu;
        let tensor = create_test_tensor(&[2, 3], &device);

        // This should panic because index 3 is out of bounds for dimension 0
        let _ = Backend::pop(&tensor, 0, 3);
    }

    #[test]
    fn test_vectorize_empty_tensor() {
        let device = Device::Cpu;

        // Create an empty tensor (shape [0])
        let empty_tensor = Tensor::zeros(&[0], DType::F32, &device).unwrap();

        // This should panic
        let output = Backend::vectorize_dim(&empty_tensor, 0);
        assert_eq!(output.len() , 0);
    }

    #[test]
    #[should_panic]
    fn test_slice_out_of_bounds() {
        let device = Device::Cpu;
        let tensor = create_test_tensor(&[2, 3], &device);

        // This should panic - trying to slice 4 elements from dim 1 of size 3
        let _ = Backend::slice(&tensor, 1, 0, 4);
    }
}
