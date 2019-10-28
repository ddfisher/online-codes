use rand::distributions::WeightedIndex;

// TODO: use larger seeds for the PRNG
// TODO: allow specification of starting block_id
// TODO: write tests with proptest
// TODO: write benchmarks with criterion
// TODO: profile and fix low-hanging fruit
// TODO: reorder functions
// TODO: make a minor code cleanup pass
// TODO: write docs
// TODO: remove main.rs

use crate::typedef::types::{StreamId, CheckBlockId};
use crate::util::helpers::{get_adjacent_blocks, xor_block};

pub struct BlockIter<'a> {
    pub data: &'a [u8],
    pub aux_data: Vec<u8>,
    pub block_size: usize,
    pub degree_distribution: WeightedIndex<f64>,
    pub check_block_id: CheckBlockId,
    pub stream_id: StreamId,
}

impl<'a> Iterator for BlockIter<'a> {
    type Item = Vec<u8>;
    fn next(&mut self) -> Option<Vec<u8>> {
        let num_blocks = self.data.len() / self.block_size;
        let num_aux_blocks = self.aux_data.len() / self.block_size;
        let mut check_block = vec![0; self.block_size];
        let adjacent_blocks = get_adjacent_blocks(
            self.check_block_id,
            self.stream_id,
            &self.degree_distribution,
            num_blocks + num_aux_blocks,
        );
        for block_index in adjacent_blocks {
            if block_index < num_blocks {
                xor_block(
                    &mut check_block,
                    &self.data[block_index * self.block_size..],
                    self.block_size,
                );
            } else {
                // Aux block.
                xor_block(
                    &mut check_block,
                    &self.aux_data[(block_index - num_blocks) * self.block_size..],
                    self.block_size,
                );
            }
        }

        self.check_block_id += 1;
        Some(check_block)
    }
}
