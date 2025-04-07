use async_trait::async_trait;
use crate::item_stream::AsyncItemStream;

#[async_trait]
pub trait Batcher<T, Q> {
    async fn run(&self, item: T) -> AsyncItemStream<Q>;
}
