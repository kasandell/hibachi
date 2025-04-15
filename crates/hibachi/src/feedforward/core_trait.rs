use async_trait::async_trait;
use crate::backend::{Backend, Unsqueezable};
use super::item::Item;

#[async_trait]
pub trait Feedforward<B, O> where B: Backend, O: Backend
{
    async fn forward(&self, tensor: B) -> O;
}

#[async_trait]
pub trait FeedforwardBatcher<T, Q> {
    async fn run(&self, item: T) -> Item<Q>;
}
