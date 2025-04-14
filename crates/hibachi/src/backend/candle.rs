use super::{Backend, LowerRankedTensorOps};
use candle_core::Tensor;


impl LowerRankedTensorOps for Tensor {
    type Unsqueezed = Tensor;

    fn unsqueeze(&self, dim: usize) -> Self::Unsqueezed {
        self.unsqueeze(dim).unwrap()
    }
}

impl Backend for Tensor {
    fn shape(&self) -> Vec<usize> {
        self.dims().to_vec()
    }

    fn cat(tensors: &[Self], dim: usize) -> Self {
        Tensor::cat(
            tensors, dim
        ).unwrap()
    }

    fn eq(&self, other: &Self) -> impl Backend {
        self.eq(other).unwrap()
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

    fn idx_where_all_true(&self, dim: usize) -> Vec<usize> {
        let dims = self.dims();
        if dim >= dims.len() {
            return vec![]; // Invalid dimension
        }

        let dim_size = dims[dim];
        let mut result = Vec::new();
        for i in 0..dim_size {
            // Use narrow instead of index
            let slice = self.narrow(dim, i, 1).unwrap();

            // Reshape to flatten all other dimensions
            let mut new_shape: Vec<usize> = vec![1; dims.len()];
            new_shape[dim] = 1;
            let flattened_size: usize = dims.iter()
                .enumerate()
                .filter(|&(d, _)| d != dim)
                .map(|(_, &s)| s)
                .product();
            new_shape.push(flattened_size);

            let reshaped = slice.reshape(&*new_shape).unwrap();
            let squeezed = reshaped.squeeze(dim).unwrap();

            // Convert to CPU and check if all elements are true
            let cpu_tensor = squeezed.to_device(&candle_core::Device::Cpu).unwrap();
            let bool_data = cpu_tensor.to_vec1::<u8>().unwrap();

            if bool_data.iter().all(|&v| v == 1) {
                result.push(i);
            }
        }

        result
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

    fn repeat(&self, dim: usize, times: usize) -> Self {
        let mut dims = self.dims().to_vec();
        dims[dim] = times;
        self.broadcast_as(dims).unwrap()
    }
}
