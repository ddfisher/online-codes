use on_line_codes::OnlineCoder;
fn main() {
    let message =
        "Gormenghast, that is the main massing of the original stone, taken by itself \
         would have displayed a certain ponderous architectural quality were it not \
         for the circumfusion of mean dwellings that swarmed like an epidemic around \
         its outer walls. They sprawled over the sloping earth, each one have way \
         over its neighbor until, held back by the castle ramparts, the innermost of \
         these hovels laid hold on the great walls, clamping themselves thereto like \
         limpets to a rock. These dwellings, by ancient law, were granted this chill \
         intimacy with the stronghold that loomed above them. Over their irregular \
         roofs would fall, thoughout the seasons, the shadows of time-eaten buttresses, \
         of broken and lofty turrets, and-most enormous of all-the shadow of the Tower of \
         Flints. This tower, patched uneavenly with black ivy, arose like a mutilated \
         finger from among the fists of knuckled masonry and pointed blasphemously at heaven. \
         At night the owls made of it an echoing throat; by day it stood voiceless and cast \
         its long shadow.";
    // let message = "this is a test";
    // let message = "01";
    let data: Vec<u8> = message.repeat(100).as_bytes().iter().map(|x| *x).collect();
    // let data: Vec<u8> = (0..100).collect();
    // dbg!(&data);
    let coder = OnlineCoder::with_parameters(0.01, 3);
    let seed = 0xDEADBEEF;
    let block_size = 1;
    let num_blocks = data.len() / block_size;
    dbg!(num_blocks);
    let mut encoded_data: Vec<u8> = Vec::new();
    for mut chunk in coder
        .encode(&data, block_size, seed)
        .take(num_blocks + 3000)
    {
        encoded_data.append(&mut chunk);
    }
    let decoded_message = coder.decode(&encoded_data, num_blocks, block_size, seed);
    // let _: Vec<(usize, u8)> = dbg!(decoded_message.into_iter().enumerate().collect());

    println!("{:?}", std::str::from_utf8(&decoded_message));
}
