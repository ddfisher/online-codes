extern crate online_codes;

use online_codes::types::StreamId;
use online_codes::{decode_block, new_decoder, new_encoder, next_block};
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
        // Not ideal but okay for now to make things work
        if buf_len > 4 {
            let block_size = buf_len/4;
            // This _needs_ to be satisfied for it to work?
            // I think we expect it to fail otherwise anyway?
            if buf_len % block_size == 0 {
                // The real test is here
                if let Some(decoded) = check_encode_decode(buf.clone()) {
                    println!("decoded: {:?}", decoded);
                    assert_eq!(decoded, buf);
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
        if buf_len > 4 {
            let block_size = buf_len/4;
            if buf_len % block_size == 0 {
                for loss in vec![0.1, 0.3, 0.5, 0.9] {
                    if let Some((decoded, loss_counter, total_counter)) = check_encode_decode_with_loss(buf.clone(), loss) {
                        // NOTE: I'm pretty sure the higher the loss, the higher the returned block_id
                        // (our counter) would be. Looking at the output below sort of justifies
                        // that as well, but we probably should have more thorough checks.
                        // There ARE inconsistencies though.
                        println!("loss: {:?}, loss_counter: {:?}, total_counter: {:?}", loss, loss_counter, total_counter);
                        println!("decoded: {:?}", decoded);
                        assert_eq!(decoded, buf);
                    }
                }
                println!()
            }
        }
    }
}

fn check_encode_decode(buf: Vec<u8>) -> Option<Vec<u8>> {
    println!("buffer: {:?}", buf);

    let buf_len = buf.len();
    let mut encoder = new_encoder(buf.clone(), buf_len / 4, 0);
    let mut decoder = new_decoder(buf_len, buf_len / 4, 0);

    // TODO: Should we put a limit or loop infinitely?
    loop {
        match next_block(&mut encoder) {
            Some(block) => {
                println!("block: {:?}", block);
                match decode_block(block, &mut decoder) {
                    None => continue,
                    Some(res) => return Some(res),
                }
            }
            None => continue,
        }
    }
}

fn check_encode_decode_with_loss(buf: Vec<u8>, loss: f64) -> Option<(Vec<u8>, StreamId, u32)> {
    let mut loss_rng = thread_rng();

    let mut total_counter = 0;
    let mut loss_counter = 0;

    println!("buffer: {:?}", buf);
    let buf_len = buf.len();
    let mut encoder = new_encoder(buf.clone(), buf_len / 4, 0);
    let mut decoder = new_decoder(buf_len, buf_len / 4, 0);

    // TODO: Should we put a limit or loop infinitely?
    loop {
        total_counter += 1;
        match next_block(&mut encoder) {
            Some(block) => {
                let rand: f64 = loss_rng.gen::<f64>();
                println!("block: {:?}", block);
                if rand > loss {
                    match decode_block(block, &mut decoder) {
                        None => continue,
                        Some(res) => return Some((res, loss_counter, total_counter)),
                    }
                } else {
                    // Ignore this block and do nothing
                    loss_counter += 1
                }
            }

            None => continue,
        }
    }
}
