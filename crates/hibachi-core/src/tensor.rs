use crate::backend::Backend;
use crate::constant::{BATCH_DIM, SEQ_DIM};

/// Return a vector of indices of dimension `0` (the batch size dimension)
///  where the token (dimension `1`) equals our stop token
pub(crate) fn where_equals_stop_token<B>(
    outputs: &B,
    stop_token: &B
) -> Vec<usize>
    where B: Backend {
    // repeat stop token across batch
    let output_dims = outputs.shape();
    let element_wise_stop = stop_token.clone()
        .broadcast_as(output_dims);
    // element wise equal
    let eq =  outputs.eq(&element_wise_stop);

    let mut dim_eq = eq;
    if output_dims.len() > SEQ_DIM {
        for (idx, &dim) in output_dims[SEQ_DIM..].iter().enumerate() {
            dim_eq = dim_eq.all_dim(SEQ_DIM);
        }
    }
    // need to collapse eq down to be eq over (batch_dim) only.
    let t = dim_eq;
    if t.shape()[BATCH_DIM] == 0 {
        return vec![]
    }
    let items = t.squeeze(BATCH_DIM);
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
        &[ input.clone(), output.clone().unsqueeze(SEQ_DIM)], SEQ_DIM
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
        let t = tensor.clone().slice(SEQ_DIM, 0, 0);
        return t
    }
    let dims = tensor.shape();
    let seq = dims[SEQ_DIM];
    let msl = max_sequence_length;
    let start_idx = seq-msl;
    let t = tensor.clone().slice(SEQ_DIM, start_idx, msl);
    t
}


pub(crate) fn pad_all_sequences<B>(
    active_tensor: &B,
    amount: usize,
    padding_token: &B
) -> B
where B: Backend {
    let active_dims = active_tensor.shape();
    let batch_size = active_dims[BATCH_DIM];
    let mut shape = vec![batch_size];
    for &dim in padding_token.shape() {
        shape.push(dim);
    }
    shape[1] = amount;
    let padding = padding_token.broadcast_as(&shape);
    B::cat(&[active_tensor.clone(), padding], 1)
}

pub(crate) fn zero_sequence<B>(
    tensor: &B,
    batch_idx: usize,
    padding_token: &B
) -> B
where B: Backend {
    let current_dims = tensor.shape();
    let seq_size = current_dims[SEQ_DIM];
    let mut padding_token_shape = padding_token.shape().to_vec();
    padding_token_shape[BATCH_DIM] = seq_size;
    let zeros = padding_token.broadcast_as(&padding_token_shape);
    assign_sequence_to_slot(tensor, &zeros, batch_idx, 0, seq_size)
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
            &sequence.unsqueeze(BATCH_DIM)
        )
}


pub(crate) fn slice_tensor_by_batch_dimension<B>(
    tensor: B
) -> Vec<B>
where B: Backend {
    tensor.vectorize_dim(BATCH_DIM)
}
