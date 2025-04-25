use std::fmt;
use crate::backend::{Backend, Unsqueezable};

// A simple mock tensor implementation for testing
#[derive(Clone, Debug)]
pub struct MockTensor {
    pub(crate) shape: Vec<usize>,
    pub(crate) value: i32,
}

impl MockTensor {
    pub fn new(shape: Vec<usize>, value: i32) -> Self {
        Self { shape, value }
    }
}

impl fmt::Display for MockTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MockTensor({:?}, {})", self.shape, self.value)
    }
}

impl Backend for MockTensor {
    fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    fn cat(tensors: &[Self], dim: usize) -> Self {
        // Simple concatenation logic for testing
        let mut new_shape = tensors[0].shape.clone();
        new_shape[dim] = tensors.iter().map(|t| t.shape[dim]).sum();

        // Use the value from the first tensor
        MockTensor::new(new_shape, tensors[0].value)
    }

    fn eq(&self, other: &Self) -> impl Backend {
        // For simplicity, compare only values not shape
        MockTensor::new(self.shape.clone(), if self.value == other.value { 1 } else { 0 })
    }

    fn vectorize_dim(&self, dim: usize) -> Vec<Self> {
        let mut result = Vec::new();
        let mut new_shape = self.shape.clone();

        // Set the dimension to 1
        new_shape[dim] = 1;

        // Create a tensor for each index of the dimension
        for _ in 0..self.shape[dim] {
            result.push(MockTensor::new(new_shape.clone(), self.value));
        }

        result
    }

    fn slice(&self, dimension: usize, _seq_start_idx: usize, len: usize) -> Self {
        let mut new_shape = self.shape.clone();
        new_shape[dimension] = len;
        MockTensor::new(new_shape, self.value)
    }

    fn idx_where_all_true(&self, _dim: usize) -> Vec<usize> {
        // For our mock, consider indices true if value is 1
        if self.value == 1 {
            (0..self.shape[_dim]).collect()
        } else {
            vec![]
        }
    }

    fn pop(&self, dim: usize, _index: usize) -> Self {
        let mut new_shape = self.shape.clone();
        if new_shape[dim] > 0 {
            new_shape[dim] -= 1;
        }
        MockTensor::new(new_shape, self.value)
    }

    fn repeat(&self, dim: usize, times: usize) -> Self {
        let mut new_shape = self.shape.clone();
        new_shape[dim] *= times;
        MockTensor::new(new_shape, self.value)
    }
}

// Implement Unsqueezable for MockTensor
impl Unsqueezable for MockTensor {
    type Unsqueezed = MockTensor;

    fn unsqueeze(&self, _dim: usize) -> Self::Unsqueezed {
        let mut new_shape = self.shape.clone();
        new_shape.insert(_dim, 1);
        MockTensor::new(new_shape, self.value)
    }
}
