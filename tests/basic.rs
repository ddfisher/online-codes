extern crate online_codes;

use online_codes::encode::OnlineCoder;
use rand::{thread_rng, Rng};
use rand::distributions::Alphanumeric;

#[test]
fn basic_test() {

    let buf_len = 128;
    let block_size = 16;
    let check_block_id = 0;

    let s: String = thread_rng().sample_iter(&Alphanumeric).take(buf_len).collect();
    let buf = s.clone().into_bytes();

    let coder = OnlineCoder::new(block_size);
    let encoded = coder.encode(&buf, check_block_id);
    let num_blocks = buf_len / block_size;
    let mut decoder = coder.decode(num_blocks as usize, check_block_id);

    for (block_id, block) in encoded {
        match decoder.decode_block(block_id, &block) {
            None => continue,
            Some(res) => {
                assert_eq!(buf, res);
                break
            }
        }
    }
}
