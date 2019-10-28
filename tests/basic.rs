extern crate online_codes;

use online_codes::encode::online_coder::OnlineCoder;
use rand::{thread_rng, Rng};
use rand::distributions::Alphanumeric;

#[test]
fn basic_test() {

    let total_len = 128;
    let s: String = thread_rng().sample_iter(&Alphanumeric).take(total_len).collect();
    let buf = s.clone().into_bytes();
    let len = buf.len();
    let to_compare = buf.clone();

    let online_coder = OnlineCoder::new(8);
    let encoded = online_coder.encode(&buf, 0);
    let num_blocks = 16;
    let mut decoder = online_coder.decode(num_blocks as usize, 0);

    println!("total_len: {:?}", total_len);
    println!("s: {:?}", s);
    println!("buf: {:?}", buf);
    println!("len: {:?}", len);
    println!("online_coder: {:?}", online_coder);

    let mut check_block_id = 0;
    for check_block in encoded {

        match check_block {
            None => break,
            Some(block) => {
                match decoder.decode_block(check_block_id, &check_block) {
                    None => break,
                    Some(res) => {
                        println!("res: {:?}", res);
                    }
                }
                // println!("check_block_id: {:?}, check_block: {:?}", check_block_id, check_block);
                if check_block_id == 10 {
                    break
                }
                check_block_id += 1;

            }
        }

    }


    // println!("encoded: {:?}", encoded);
    // println!("decoder: {:?}", decoder);
    assert_eq!(1, 2);
}
