extern crate online_codes;

use online_codes::encode::OnlineCoder;
use online_codes::types::CheckBlockId;
use proptest::prelude::*;
use rand::{thread_rng, Rng};

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

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10000))]
    #[test]
    fn test_with_known_loss(s in ".*") {
        let buf = s.clone().into_bytes();
        let buf_len = buf.len();
        let check_block_id = 0;
        if buf_len > 4 {
            let block_size = buf_len/4;
            let num_blocks = buf_len / block_size;
            if buf_len % block_size == 0 {
                for loss in vec![0.1, 0.3, 0.5, 0.9] {
                    if let Some((decoded, block_id, rand)) = check_encode_decode_with_loss(buf.clone(), num_blocks, block_size, check_block_id, loss) {
                        // NOTE: I'm pretty sure the higher the loss, the higher the returned block_id
                        // (our counter) would be. Looking at the output below sort of justifies
                        // that as well, but we probably should have more thorough checks.
                        // There ARE inconsistencies though.
                        println!("block_id: {:?}, rand: {:?}, loss: {:?}", block_id, rand, loss);
                        assert_eq!(decoded, buf);
                    }
                }
                println!()
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

fn check_encode_decode_with_loss(
    buf: Vec<u8>,
    num_blocks: usize,
    block_size: usize,
    check_block_id: CheckBlockId,
    loss: f64) -> Option<(Vec<u8>, CheckBlockId, f64)> {
    let coder = OnlineCoder::new(block_size);
    let encoded = coder.encode(&buf, check_block_id);
    let mut decoder = coder.decode(num_blocks, check_block_id);
    let mut loss_rng = thread_rng();

    for (block_id, block) in encoded {
        // This basically just simulates not actually decoding the block
        // at hand randomly. Could do better here presumably?
        let rand: f64 = loss_rng.gen();
        if rand > loss {
            match decoder.decode_block(block_id, &block) {
                None => continue,
                Some(res) => {
                    return Some((res, block_id, rand))
                }
            }

        }

    }
    None
}
