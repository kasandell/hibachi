use async_trait::async_trait;
use crate::r#async::item_stream::AsyncItemStream;

#[async_trait]
pub trait Batcher<T, Q> {
    async fn run(&self, item: T) -> AsyncItemStream<Q>;
}
