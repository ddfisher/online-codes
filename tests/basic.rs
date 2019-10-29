extern crate online_codes;

use online_codes::encode::OnlineCoder;
use online_codes::types::CheckBlockId;
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10000))]
    #[test]
    fn test_identity(s in ".*") {
        // Generate some data
        // NOTE: We will not mangle it here, separate test for that
        // We'll just see if we can encode <=> decode
        let buf = s.clone().into_bytes();
        let buf_len = buf.len();
        let check_block_id = 0;
        // Not ideal but okay for now to make things work
        if buf_len > 4 {
            let block_size = buf_len/4;
            let num_blocks = buf_len / block_size;
            // This _needs_ to be satisfied for it to work?
            // I think we expect it to fail otherwise anyway?
            if buf_len % block_size == 0 {
                // The real test is here
                if let Some(foo) = check_encode_decode(buf.clone(), num_blocks, block_size, check_block_id) {
                    println!("buf: {:?}", buf);
                    println!("foo: {:?}", foo);
                    println!();
                    assert_eq!(foo, buf);
                }
            }
        }
    }
}

fn check_encode_decode(
    buf: Vec<u8>,
    num_blocks: usize,
    block_size: usize,
    check_block_id: CheckBlockId) -> Option<Vec<u8>> {
    let coder = OnlineCoder::new(block_size);
    let encoded = coder.encode(&buf, check_block_id);
    let mut decoder = coder.decode(num_blocks, check_block_id);

    for (block_id, block) in encoded {
        match decoder.decode_block(block_id, &block) {
            None => continue,
            Some(res) => {
                return Some(res)
            }
        }
    }
    None
}
