use crate::backend::Backend;
use crate::constant::{BATCH_DIM, SEQ_DIM};

/// Return a vector of indices of dimension `0` (the batch size dimension)
///  where the token (dimension `1`) equals our stop token
/// outputs of dimensions (batch, ...tok), stop_token of dimensions (1, ...tok)
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
    // need to collapse eq down to be eq over (batch_dim) only.
    let non_batch_dim = 1;
    if output_dims.len() > non_batch_dim {
        for _ in output_dims[non_batch_dim..].iter() {
            dim_eq = dim_eq.all_dim(non_batch_dim);
        }
    }
    let items = dim_eq;
    if items.shape().len() == 0 {
        return vec![]
    }
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
    tensor: &Option<B>,
    max_sequence_length: usize
) -> Option<B>
where B: Backend {
    match tensor {
        None => None,
        Some(tensor) => {
            if max_sequence_length == 0 {
                let t = tensor.clone().slice(SEQ_DIM, 0, 0);
                return Some(t)
            }
            let dims = tensor.shape();
            let seq = dims[SEQ_DIM];
            let msl = max_sequence_length;
            let start_idx = seq-msl;
            let t = tensor.clone().slice(SEQ_DIM, start_idx, msl);
            Some(t)
        }
    }

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

// active tensor of shape (batch, seq, ..), sequence in (seq, ...)
pub(crate) fn add_sequence_to_outside_of_slot<B>(
    active_tensor: &B,
    sequence: &B,
) -> B
where B: Backend {
    B::cat(&[active_tensor.clone(), sequence.unsqueeze(0)], 0)
}


// active tensor of shape (batch, seq, ..), sequence in (seq, ...)
pub(crate) fn pop_sequence_from_slot<B>(
    active_tensor: &mut Option<B>,
    index: usize,
) -> Option<B>
where B: Backend {
    match active_tensor {
        None => None,
        Some(active_tensor) => {
            let dims = active_tensor.shape();
            if dims[BATCH_DIM]  == 1 {
                None
            } else {
                Some(active_tensor.pop(BATCH_DIM, index))
            }
        }
    }
}

// sequence of shape (seq, ...), padding of shape (1, ...)
pub(crate) fn pad_single_sequence<B>(
    sequence: &B,
    amount: usize,
    padding_token: &B
) -> B
where B: Backend {
    let dim_to_use = 0;
    let mut shape = padding_token.shape().to_vec();//vec![batch_size];
    shape[dim_to_use] = amount;
    let padding = padding_token.broadcast_as(&shape);
    B::cat(&[padding, sequence.clone()], dim_to_use)
}


pub(crate) fn slice_tensor_by_batch_dimension<B>(
    tensor: B
) -> Vec<B>
where B: Backend {
    tensor.vectorize_dim(BATCH_DIM)
}
