use crate::backend::{Backend, Unsqueezable};
use super::constant::{BATCH_DIM, SEQ_DIM};

/// Identifies batch indices where generated tokens match the stop token.
///
/// This function compares each output token in the batch with the stop token
/// and returns the indices of batch elements that have generated the stop token.
///
/// # Parameters
///
/// * `outputs` - Tensor of shape `(batch, ...tok)` containing generated tokens
/// * `stop_token` - Tensor of shape `(1, ...tok)` representing the stop token
///
/// # Returns
///
/// A vector of batch indices (positions in dimension 0) where the generated token
/// exactly matches the stop token.
///
/// # Implementation Notes
///
/// The function:
/// 1. Repeats the stop token across the batch dimension to match the output shape
/// 2. Performs element-wise equality comparison between outputs and the repeated stop token
/// 3. Identifies indices where all elements in the token representation are equal
#[allow(dead_code)]
pub(crate) fn where_equals_stop_token<B>(
    outputs: &B,
    stop_token: &B,
) -> Vec<usize>
    where B: Backend
{
    // repeat stop token across batch
    let output_dims = outputs.shape();
    let element_wise_stop = stop_token.clone()
        .repeat(0, output_dims[0]);
    // element wise equal
    let eq =  outputs.eq(&element_wise_stop);



    eq.idx_where_all_true(0)
}


/// Appends a new token to sequences in the active tensor.
///
/// This function concatenates newly generated tokens to their respective sequences
/// in the active tensor, extending each sequence by one token.
///
/// # Parameters
///
/// * `input` - Tensor of shape `(batch, seq, ...)` containing current sequences
/// * `output` - Tensor of shape `(batch, ...)` containing newly generated tokens
///
/// # Returns
///
/// A new tensor of shape `(batch, seq+1, ...)` with the new tokens appended
/// to each sequence in the batch.
///
/// # Implementation Notes
///
/// The function:
/// 1. Unsqueezes the output tensor to add a sequence dimension of size 1
/// 2. Concatenates the original input and the reshaped output along the sequence dimension
#[allow(dead_code)]
pub(crate) fn concat_output<B>(input: &B::Unsqueezed, output: &B) -> B::Unsqueezed
where B: Backend + Unsqueezable
{
    B::Unsqueezed::cat(
        &[ input.clone(), output.clone().unsqueeze(SEQ_DIM)], SEQ_DIM
    )
}


/// Trims sequences to a specified maximum length.
///
/// When sequences grow beyond a certain length, this function trims them by
/// removing tokens from the beginning while preserving the most recent tokens
/// up to the specified maximum length.
///
/// # Parameters
///
/// * `tensor` - Optional tensor that may contain sequences to trim
/// * `max_sequence_length` - Maximum number of tokens to retain in each sequence
///
/// # Returns
///
/// An optional tensor with sequences trimmed to the specified maximum length.
/// If the input tensor is `None`, returns `None`. If `max_sequence_length` is 0,
/// returns a tensor with empty sequences.
///
/// # Implementation Notes
///
/// The function:
/// 1. Handles the case when `tensor` is `None` by returning `None`
/// 2. Handles zero length by returning an empty slice
/// 3. Otherwise, calculates the starting index to keep tokens from the end
/// 4. Slices the tensor along the sequence dimension to retain only recent tokens
#[allow(dead_code)]
pub fn trim_sequence<B>(
    tensor: &Option<B>,
    max_sequence_length: usize
) -> Option<B>
where B: Backend
{
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


/// Pads all sequences in the active tensor.
///
/// When new sequences longer than existing ones are added to the batch,
/// this function pads all existing sequences to match the new length.
///
/// # Parameters
///
/// * `active_tensor` - Tensor of shape `(batch, seq, ...)` containing existing sequences
/// * `amount` - Number of padding tokens to add to each sequence
/// * `padding_token` - Token to use for padding
///
/// # Returns
///
/// A new tensor with all sequences padded by the specified amount.
///
/// # Implementation Notes
///
/// The function:
/// 1. Creates a padding tensor by repeating the padding token
/// 2. Concatenates the padding to all sequences along the sequence dimension
#[allow(dead_code)]
pub(crate) fn pad_all_sequences<B>(
    active_tensor: &B::Unsqueezed,
    amount: usize,
    padding_token: &B,
) -> B::Unsqueezed
where B: Backend + Unsqueezable
{
    let padding = padding_token.unsqueeze(0).repeat(0, amount);
    B::Unsqueezed::cat(&[active_tensor.clone(), padding], 1)
}

/// Adds a new sequence to the batch.
///
/// This function appends a new sequence to the existing batch of sequences
/// in the active tensor, increasing the batch size by one.
///
/// # Parameters
///
/// * `active_tensor` - Tensor of shape `(batch, seq, ...)` containing existing sequences
/// * `sequence` - Tensor of shape `(seq, ...)` representing the new sequence to add
///
/// # Returns
///
/// A new tensor with the additional sequence added to the batch.
///
/// # Implementation Notes
///
/// The function:
/// 1. Unsqueezes the sequence to add a batch dimension of size 1
/// 2. Concatenates the original tensor and the reshaped sequence along the batch dimension
#[allow(dead_code)]
pub(crate) fn add_sequence_to_outside_of_slot<B>(
    active_tensor: &B::Unsqueezed,
    sequence: &B,
) -> B::Unsqueezed
where B: Backend + Unsqueezable,
{
    B::Unsqueezed::cat(&[active_tensor.clone(), sequence.unsqueeze(0)], 0)
}


/// Removes a sequence from the batch.
///
/// When a sequence completes (by generating a stop token), this function
/// removes it from the active tensor to free up space in the batch.
///
/// # Parameters
///
/// * `active_tensor` - Optional tensor that may contain sequences
/// * `index` - Batch index of the sequence to remove
///
/// # Returns
///
/// An optional tensor with the specified sequence removed. If removing the last
/// sequence, returns `None` to indicate an empty batch.
///
/// # Implementation Notes
///
/// The function:
/// 1. Handles the case when `active_tensor` is `None` by returning `None`
/// 2. Checks if this is the last sequence in the batch (returns `None` if so)
/// 3. Otherwise, uses the `pop` operation to remove the sequence at the specified index
#[allow(dead_code)]
pub(crate) fn pop_sequence_from_slot<B>(
    active_tensor: &mut Option<B>,
    index: usize,
) -> Option<B>
where B: Backend
{
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

/// Pads a single sequence to match the batch sequence length.
///
/// When adding a shorter sequence to a batch with longer sequences,
/// this function pads the sequence to match the required length.
///
/// # Parameters
///
/// * `sequence` - Tensor of shape `(seq, ...)` representing the sequence to pad
/// * `amount` - Number of padding tokens to add
/// * `padding_token` - Token to use for padding
///
/// # Returns
///
/// A new tensor with the sequence padded to the required length.
///
/// # Implementation Notes
///
/// The function:
/// 1. Creates a padding tensor by repeating the padding token
/// 2. Concatenates the padding with the original sequence along the sequence dimension
///
/// Note: The padding is added at the beginning of the sequence, preserving the
/// most recent tokens at the end.
#[allow(dead_code)]
pub(crate) fn pad_single_sequence<B>(
    sequence: &B,
    amount: usize,
    padding_token: &B
) -> B
where B: Backend
{
    let dim_to_use = 0;
    let mut shape = padding_token.shape();
    shape[dim_to_use] = amount;
    let padding = padding_token.repeat(0, amount);//.broadcast_as(&shape);
    B::cat(&[padding, sequence.clone()], dim_to_use)
}

/// Splits a tensor into individual samples along the batch dimension.
///
/// This function divides a batched tensor into a vector of individual tensors,
/// one for each element in the batch.
///
/// # Parameters
///
/// * `tensor` - Tensor of shape `(batch, ...)` to split along the batch dimension
///
/// # Returns
///
/// A vector of tensors, each representing one element from the original batch.
///
/// # Implementation Notes
///
/// The function uses the `vectorize_dim` operation provided by the `Backend` trait
/// to split the tensor along the batch dimension (dimension 0).
///
/// This is typically used to distribute model outputs to the corresponding result streams.
#[allow(dead_code)]
pub(crate) fn slice_tensor_by_batch_dimension<B>(
    tensor: B
) -> Vec<B>
where B: Backend
{
    tensor.vectorize_dim(BATCH_DIM)
}
