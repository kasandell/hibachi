pub trait Backend: Clone + Send + Sync + 'static {
    type DType;
    type Device;
    // Create a tensor of given shape, filled with zeros
    fn zeros(shape: &[usize], dtype: Self::DType, device: &Self::Device) -> Self;
    // Return the shape of the tensor
    fn shape(&self) -> &[usize];
    // Return the device (useful if we need to create new tensors on the same device)
    fn device(&self) -> &Self::Device;
    // Return the data type of the tensor
    fn dtype(&self) -> Self::DType;
    // Concatenate tensors along a dimension
    fn cat(tensors: &[Self], dim: usize) -> Self;
    // Unsqueeze (add a dimension of size 1)
    fn squeeze(&self, dim: usize) -> Self;
    // Unsqueeze (add a dimension of size 1)
    fn unsqueeze(&self, dim: usize) -> Self;
    // Narrow (slice) the tensor along a dimension (start index and length)
    //fn narrow(&self, dim: usize, start: usize, len: usize) -> Self;
    // Element-wise equality comparison (typically for stop-token checking)
    fn eq(&self, other: &Self) -> Self;
    // (Optional) convert to a Rust vector for small tensors (like boolean masks)
    fn to_vec_u8(&self) -> Vec<u8>;  // e.g., used if dtype is bool/u8 for masks
    // slice a dimension returning a vectorized view over the data along that dimension
    // TODO: combine with above?
    fn vectorize_dim(&self, dim: usize) -> Vec<Self>;
    // assign the dimension and range with a new tensor
    fn slice_assign(&self, batch_index: usize, seq_start_idx: usize, seq_end_idx: usize, other: &Self) -> Self;
    // provide the sliced tensor along dimension, from start to end
    fn slice(&self, dimension: usize, seq_start_idx: usize, seq_end_idx: usize) -> Self;
    fn broadcast_as(&self, dims: &[usize]) -> Self;
    fn transpose_dims(&self, first: usize, second: usize) -> Self;
    fn all_dim(&self, dim: usize) -> Self;
}
