use std::thread;


/*
since i know someone is going to see this and wonder why:
candle does not panic, but burn does. i wrote this around burn first, so there's no results,
if the panic happens in the background thread, it will just hang. this at least causes escalation to the
main thread. its a hack for now, but it does what it needs to.
ideally, panic in the background, we'd just close out all our senders and poison all mutex.

 */
pub struct Pill {}

impl Drop for Pill {
    fn drop(&mut self) {
        if thread::panicking() {
            panic!("Thread panic")
        }
    }
}
