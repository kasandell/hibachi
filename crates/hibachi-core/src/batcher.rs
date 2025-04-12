use async_trait::async_trait;
use crate::communcation::ItemStream;

#[async_trait]
pub trait AutoregressiveBatcher<T, Q> {
    async fn run(&self, item: T) -> ItemStream<Q>;
}
