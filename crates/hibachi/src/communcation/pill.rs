use std::thread;

/// A poison pill, used internally to surface panics from child backends to the main thread
/// and prevent main threads from hanging infinitely for a child thread which will never join
pub struct Pill {}

impl Drop for Pill {
    /// Drop is able to check why we are dropping. If we discover we dropped because of a panic
    /// (in the child thread), we can panic in the main thread as well to kill the program effectively.
    /// Since panics are non-recoverable, a child panic should trigger a parent panic
    fn drop(&mut self) {
        if thread::panicking() {
            panic!("Thread panic")
        }
    }
}
