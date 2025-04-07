use burn::prelude::{Backend, Shape, Tensor};

/// Return a vector of indices of dimension `0` (the batch size dimension)
///  where the token (dimension `1`) equals our stop token
pub(crate) fn where_equals_stop_token<B>(
    outputs: &Tensor<B, 2>,
    stop_token: &Tensor<B, 1>
) -> Vec<usize>
    where B: Backend {
    // repeat stop token across batch
    let element_wise_stop = stop_token.clone().unsqueeze_dim(0)
        .repeat_dim(0, outputs.shape().dims[0]);
    // element wise equal
    let eq =  outputs.clone().equal(element_wise_stop);
    // collapse equal by row
    let collapsed = eq.all_dim(1);
    // squeeze to rows
    let collapsed_squeezed = collapsed.squeeze::<1>(1);
    let indices = collapsed_squeezed.argwhere();
    let t = indices.transpose();
    if t.shape().dims[1] == 0 {
        return vec![]
    }
    let items = t.squeeze::<1>(0);
    let data = items.to_data();
    let mut final_indices = vec![];
    for row in data.iter::<u32>() {
        final_indices.push(row as usize);
    }
    final_indices
}


/// Concatenate a tensor of rank 2 to the end of a tensor of rank 3,
/// along dimension 1.
/// Assuming that input is of shape (batch, seq, tok), and output is of (batch, tok)
pub(crate) fn concat_output<B>(input: Tensor<B, 3>, output: Tensor<B, 2>) -> Tensor<B, 3>
where B: Backend{
    let concatenated = Tensor::cat(
        vec![ input, output.unsqueeze_dim(1)], 1
    );
    concatenated
}


/// Given a max sequence length, trim the tensor to be
/// max_sequence_length long, dropping elements from the front of the tensor
/// along the seq dimension (`2`)
pub fn trim_sequence<B>(
    tensor: &Tensor<B, 3>,
    max_sequence_length: usize
) -> Tensor<B, 3>
where B: Backend {
    if max_sequence_length == 0 {
        return tensor.clone();
    }
    let dims = tensor.shape().dims;
    let seq = dims[1] as i64;
    let msl = max_sequence_length as i64;
    let start_idx = seq-msl;
    tensor.clone().slice([None, Some((start_idx, seq)), None])
}


pub(crate) fn zero_pad_sequence<B>(
    active_tensor: &mut Tensor<B, 3>,
    amount: usize
) where B: Backend {
    let active_dims = active_tensor.dims();
    let device = active_tensor.device();
    let (batch_size, tok_width) = (active_dims[0], active_dims[2]);
    let padding = Tensor::zeros(Shape::from([batch_size, amount, tok_width]), &device);
    *active_tensor = Tensor::cat(vec![active_tensor.clone(), padding], 1);
}


pub(crate) fn slice_tensor_by_batch_dimension<B, const S: usize>(
    tensor: Tensor<B, 2>,
) -> Vec<Tensor<B, 1>>
where B: Backend {
    tensor.chunk(S, 0).into_iter().map(|tensor| tensor.squeeze(0)).collect()
}
