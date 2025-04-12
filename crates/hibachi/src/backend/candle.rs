use super::Backend;
use candle_core::Tensor;

impl Backend for Tensor {
    type DType = candle_core::DType;
    type Device = candle_core::Device;

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
            tensors, dim
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

        let dim0_size = dims[dim];
        let mut result = Vec::with_capacity(dim0_size);

        // Extract each slice along the first dimension
        for i in 0..dim0_size {
            let slice = self.narrow(dim, i, 1).unwrap();
            // Remove the first dimension which now has size 1
            let slice = slice.squeeze(dim).unwrap();
            result.push(slice);
        }

        result
    }

    fn slice(&self, dimension: usize, seq_start_idx: usize, len: usize) -> Self {
        self.narrow(dimension, seq_start_idx, len).expect(&format!("Unwraps: {}, {}, {:?}", seq_start_idx, len, self.dims()))
    }

    fn broadcast_as(&self, dims: &[usize]) -> Self {
        self.broadcast_as(dims).unwrap()
    }

    fn all_dim(&self, dim: usize) -> Self {
        self.min(dim).unwrap()
    }

    fn pop(&self, dim: usize, index: usize) -> Self {
        let dim_size = self.dim(dim).unwrap();

        if index >= dim_size {
            panic!("Index {} is out of bounds for dimension {} with size {}", index, dim, dim_size);
        }

        // If idx is 0, we just take everything after idx
        if index == 0 {
            return self.narrow(dim, 1, dim_size - 1).unwrap();
        }

        // If idx is the last element, we take everything before idx
        if index == dim_size - 1 {
            return self.narrow(dim, 0, dim_size - 1).unwrap();
        }

        // Otherwise, we need to concatenate the parts before and after idx
        let first_part = self.narrow(dim, 0, index).unwrap();
        let second_part = self.narrow(dim, index + 1, dim_size - index - 1).unwrap();

        Tensor::cat(&[&first_part, &second_part], dim).unwrap()
    }
}
