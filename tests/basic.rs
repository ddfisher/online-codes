extern crate online_codes;

use online_codes::encode::OnlineCoder;
use online_codes::decode::Decoder;
use rand::{thread_rng, Rng};
use rand::distributions::Alphanumeric;

#[test]
fn basic_test() {

    let total_len = 128;
    let s: String = thread_rng().sample_iter(&Alphanumeric).take(total_len).collect();
    let buf = s.clone().into_bytes();
    let len = buf.len();
    let _to_compare = buf.clone();

    let coder = OnlineCoder::new(8);
    let encoded = coder.encode(&buf, 0);
    let num_blocks = 16;
    let mut decoder = coder.decode(num_blocks as usize, 0);

    println!("coder: {:?}", coder);
    println!();
    println!("encoded: {:?}", encoded);
    println!();
    println!("decoder: {:?}", decoder);

    assert_eq!(1, 1);
}
