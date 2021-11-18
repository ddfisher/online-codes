use block_iter::BlockIter;
use decode::Decoder;
use types::{CheckBlockId, StreamId};

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

pub type Block = (CheckBlockId, Vec<u8>);

pub fn new_encoder(mut buf: Vec<u8>, block_size: usize, stream_id: StreamId) -> Encoder {
    let len = buf.len();
    let rem = len % block_size;
    let pad: i64 = block_size as i64 - rem as i64;
    buf.resize_with(len + pad.abs() as usize, || 0);

    let coder = encode::OnlineCoder::new(block_size);
    let block_iter = coder.encode(buf, stream_id);
    Encoder { block_iter }
}

pub fn new_decoder(buf_len: usize, block_size: usize, stream_id: StreamId) -> Decoder {
    let len = buf_len;
    let rem = len % block_size;
    let pad: i64 = (block_size as i64 - rem as i64).abs();
    Decoder::new(
        (buf_len + pad as usize) / block_size,
        block_size,
        stream_id,
        Some(pad),
    )
}

pub fn next_block(encoder: &mut Encoder) -> Option<Block> {
    encoder.block_iter.next()
}

pub fn decode_block(block: Block, decoder: &mut Decoder) -> Option<Vec<u8>> {
    match decoder.decode_block(block.0, &block.1) {
        Some(mut block) => match decoder.pad {
            Some(pad) => {
                let len = block.len();
                block.resize(len - pad as usize, 0);
                Some(block)
            }
            None => Some(block),
        },
        None => None,
    }
}
