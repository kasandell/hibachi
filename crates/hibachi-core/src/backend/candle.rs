use super::Backend;
use candle_core::{Tensor, WithDType};

impl Backend for Tensor {
    type DType = candle_core::DType;
    type Device = candle_core::Device;

    fn zeros(shape: &[usize], dtype: Self::DType, device: &Self::Device) -> Self {
        Tensor::zeros(
            shape,
            dtype,
            device
        ).expect("Creates active tensor")
    }

    fn shape(&self) -> &[usize] {
        self.shape().dims()
    }

    fn device(&self) -> &Self::Device {
        self.device()
    }

    fn dtype(&self) -> Self::DType {
        self.dtype()
    }

    fn cat(tensors: &[Self], dim: usize) -> Self {
        Tensor::cat(
            tensors, 1
        ).unwrap()
    }

    fn squeeze(&self, dim: usize) -> Self {
        self.squeeze(dim).unwrap()
    }

    fn unsqueeze(&self, dim: usize) -> Self {
        self.unsqueeze(dim).unwrap()
    }

    fn eq(&self, other: &Self) -> Self {
        self.eq(other).unwrap()
    }

    fn to_vec_u8(&self) -> Vec<u8> {
        self.to_vec1::<u8>().unwrap()
    }

    fn vectorize_dim(&self, dim: usize) -> Vec<Self> {
        let dims = self.dims();

        // Ensure tensor has at least one dimension
        if dims.is_empty() {
            panic!("Empty dimension tensor")
        }

        let dim0_size = dims[0];
        let mut result = Vec::with_capacity(dim0_size);

        // Extract each slice along the first dimension
        for i in 0..dim0_size {
            let slice = self.narrow(i, 0, 1).unwrap();
            // Remove the first dimension which now has size 1
            let slice = slice.squeeze(0).unwrap();
            result.push(slice);
        }

        result
    }

    fn slice_assign(&self, batch_index: usize, seq_start_idx: usize, seq_end_idx: usize, other: &Self) -> Self {
        self.slice_assign(&[batch_index..batch_index+1, seq_start_idx..seq_end_idx], &other.unsqueeze(0).unwrap()).unwrap()
    }

    fn slice(&self, dimension: usize, seq_start_idx: usize, seq_end_idx: usize) -> Self {
        self.narrow(dimension, seq_start_idx, seq_end_idx - seq_start_idx).expect(&format!("Unwraps: {}, {}, {:?}", seq_start_idx, seq_end_idx, self.dims()))
    }

    fn broadcast_as(&self, dims: &[usize]) -> Self {
        self.broadcast_as(dims).unwrap()
    }

    fn transpose_dims(&self, first: usize, second: usize) -> Self {
        self.transpose(first, second).unwrap()
    }

    fn all_dim(&self, dim: usize) -> Self {
        self.all_dim(dim)
    }
}
