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
    let pad: usize = match rem {
        0 => 0,
        r => block_size - r,
    };
    assert!(pad < block_size);
    buf.resize_with(len + pad, || 0);
    let coder = encode::OnlineCoder::new(block_size);
    let block_iter = coder.encode(buf, stream_id);
    Encoder { block_iter }
}

pub fn new_decoder(buf_len: usize, block_size: usize, stream_id: StreamId) -> Decoder {
    let len = buf_len;
    let rem = len % block_size;
    let pad: usize = match rem {
        0 => 0,
        r => block_size - r,
    };
    assert!(pad < block_size);
    Decoder::new((buf_len + pad) / block_size, block_size, stream_id, pad)
}

pub fn next_block(encoder: &mut Encoder) -> Option<Block> {
    encoder.block_iter.next()
}

pub fn decode_block(block: Block, decoder: &mut Decoder) -> Option<Vec<u8>> {
    match decoder.decode_block(block.0, &block.1) {
        Some(mut block) => {
            let pad = decoder.pad;
            let len = block.len();
            block.resize(len - pad as usize, 0);
            Some(block)
        }
        None => None,
    }
}
