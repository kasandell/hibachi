use candle_core::{Tensor};


pub(crate) fn where_equals_stop_token(
    outputs: &Tensor,
    stop_token: &Tensor
) -> Vec<usize> {
    let output_dims = outputs.dims();

    // Broadcast/repeat stop_token to match outputs dimensions
    // Assuming stop_token is a single token that needs to be compared across the entire output
    let broadcasted_stop_token = stop_token.broadcast_as(output_dims).unwrap();

    let mask = outputs.eq(&broadcasted_stop_token).unwrap();


    let mask_flat = mask.reshape(&[mask.dims()[0]]).unwrap();

    // Step 3: Convert to CPU and get the data as a Vec<bool> or Vec<u8>
    let mask_vec = mask_flat.to_vec1::<u8>().unwrap();

    // Step 4: Find the indices where mask is true (1)
    let indices: Vec<usize> = mask_vec.iter()
        .enumerate()
        .filter_map(|(idx, &val)| if val == 1 { Some(idx) } else { None })
        .collect();

    indices
}


pub(crate) fn concat_output(input: Tensor, output: Tensor) -> Tensor {
    Tensor::cat(
        &[ &input, &output.unsqueeze(1).unwrap()], 1
    ).unwrap()
}

pub fn trim_sequence(
    tensor: &Tensor,
    max_sequence_length: usize
) -> Tensor {
    if max_sequence_length == 0 {
        return tensor.clone();
    }
    let dims = tensor.dims();
    let seq = dims[1];
    let start_idx = seq-max_sequence_length;
    tensor.clone().narrow(1, start_idx, max_sequence_length).expect(&format!("Unwraps: {}, {}, {:?}", start_idx, seq, tensor.dims()))
}

pub(crate) fn zero_pad_sequence(
    mut active_tensor: &mut Tensor,
    amount: usize
) {
    let active_dims = active_tensor.dims();
    let device = active_tensor.device();
    let dtype = active_tensor.dtype();
    let batch_size = active_dims[0];
    let mut padding = Tensor::zeros((batch_size, amount), dtype, device).expect("creates tensor");
    let mut atc = active_tensor.clone();
    *active_tensor = Tensor::cat(&[&atc, &padding], 1).expect("unwraps");
}

pub(crate) fn slice_tensor_by_batch_dimension<const S: usize>(
    tensor: Tensor
) -> Vec<Tensor>
{
    tensor.chunk(S, 0).expect("should create chunks")
}
