use block_iter::BlockIter;
use decode::Decoder;

mod block_iter;
pub mod decode;
mod encode;
pub mod types;
mod util;

// TODO: use larger seeds for the PRNG
// TODO: allow specification of starting block_id
// TODO: write more tests with proptest
// TODO: write benchmarks with criterion
// TODO: profile and fix low-hanging fruit
// TODO: write docs

#[derive(Clone)]
pub struct Encoder {
    // NOTE: BlockIter MUST not leak, maybe put in a Box?
    block_iter: BlockIter,
}

pub type Block = (u64, Vec<u8>);

pub fn new_encoder(buf: Vec<u8>, block_size: usize, stream_id: u64) -> Encoder {
    let coder = encode::OnlineCoder::new(block_size);
    let block_iter = coder.encode(buf, stream_id);
    Encoder { block_iter }
}

pub fn new_decoder(buf_len: usize, block_size: usize, stream_id: u64) -> Decoder {
    Decoder::new(buf_len / block_size, block_size, stream_id)
}

pub fn next_block(encoder: &mut Encoder) -> Option<Block> {
    encoder.block_iter.next()
}

pub fn decode_block(block: Block, decoder: &mut Decoder) -> Option<Vec<u8>> {
    decoder.decode_block(block.0, &block.1)
}
