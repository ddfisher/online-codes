// use OnLineCodes;

fn main() {
    let mut data: Vec<u8> = "this is a test".as_bytes().iter().map(|x| *x).collect();
    let seed = 0;
    let block_size = 1;
    let num_blocks = data.len();
    let mut encoded_data: Vec<u8> = Vec::new();
    for mut chunk in on_line_codes::encode(&mut data, block_size, seed).take(num_blocks + 5) {
        encoded_data.append(&mut chunk);
    }
    let decoded_message = on_line_codes::decode(&encoded_data, num_blocks, block_size, seed);

    println!("{:?}", std::str::from_utf8(&decoded_message));
}
