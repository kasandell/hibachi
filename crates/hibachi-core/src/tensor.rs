use crate::backend::Backend;

/// Return a vector of indices of dimension `0` (the batch size dimension)
///  where the token (dimension `1`) equals our stop token
pub(crate) fn where_equals_stop_token<B>(
    outputs: &B,
    stop_token: &B
) -> Vec<usize>
    where B: Backend {
    // repeat stop token across batch
    let output_dims = outputs.shape();
    let element_wise_stop = stop_token.clone().unsqueeze(0)
        .broadcast_as(output_dims);
    // element wise equal
    let eq =  outputs.eq(&element_wise_stop);

    let mut dim_eq = eq;
    for &dim in &output_dims[2..] {
        dim_eq = dim_eq.all_dim(dim);
    }
    for &dim in &output_dims[2..] {
        dim_eq = dim_eq.squeeze(2);
    }
    // need to collapse eq down to be eq over (batch_dim) only.


    let t = dim_eq.transpose_dims(0, 1);
    if t.shape()[1] == 0 {
        return vec![]
    }
    let items = t.squeeze(0);
    let mask_vec = items.to_vec_u8();

    // Step 4: Find the indices where mask is true (1)
    let indices: Vec<usize> = mask_vec.iter()
        .enumerate()
        .filter_map(|(idx, &val)| if val == 1 { Some(idx) } else { None })
        .collect();

    indices
}


/// Concatenate a tensor of rank 2 to the end of a tensor of rank 3,
/// along dimension 1.
/// Assuming that input is of shape (batch, seq, ...), and output is of (batch, ...)
pub(crate) fn concat_output<B>(input: &B, output: &B) -> B
where B: Backend{
    B::cat(
        &[ input.clone(), output.clone().unsqueeze(1)], 1
    )
}


/// Given a max sequence length, trim the tensor to be
/// max_sequence_length long, dropping elements from the front of the tensor
/// along the seq dimension (`2`)
pub fn trim_sequence<B>(
    tensor: &B,
    max_sequence_length: usize
) -> B
where B: Backend {
    if max_sequence_length == 0 {
        return tensor.clone();
    }
    let dims = tensor.shape();
    let seq = dims[1];
    let msl = max_sequence_length;
    let start_idx = seq-msl;
    tensor.clone().slice(1, start_idx, msl)
}


pub(crate) fn zero_pad_sequence<B>(
    active_tensor: &B,
    amount: usize
) -> B
where B: Backend {
    let active_dims = active_tensor.shape();
    let device = active_tensor.device();
    let batch_size = active_dims[0];
    let padding = B::zeros(&[batch_size, amount], active_tensor.dtype(), &device);
    B::cat(&[active_tensor.clone(), padding], 1)
}

pub(crate) fn zero_sequence<B>(
    tensor: &B,
    batch_idx: usize
) -> B
where B: Backend {
    let current_dims = tensor.shape();
    let zeros = B::zeros(&[1, current_dims[1]], tensor.dtype(), &tensor.device());
    let batch_size = current_dims[0];
    assign_sequence_to_slot(tensor, &zeros, batch_idx, 0, batch_size)
}

pub(crate) fn assign_sequence_to_slot<B>(
    active_tensor: &B,
    sequence: &B,
    batch_index: usize,
    start_index: usize,
    end_index: usize
) -> B
where B: Backend {
    active_tensor
        .slice_assign(
            batch_index,
            start_index,
            end_index,
            &sequence.unsqueeze(0)
        )
}


pub(crate) fn slice_tensor_by_batch_dimension<B>(
    tensor: B
) -> Vec<B>
where B: Backend {
    tensor.vectorize_dim(0)
}
