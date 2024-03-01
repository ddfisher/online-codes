use crate::types::{CheckBlockId, StreamId};
use crate::util::{get_adjacent_blocks, xor_block};
use rand::distributions::WeightedIndex;

#[derive(Clone, Debug)]
pub struct BlockIter {
    pub data: Vec<u8>,
    pub aux_data: Vec<u8>,
    pub block_size: usize,
    pub degree_distribution: WeightedIndex<f64>,
    pub check_block_id: CheckBlockId,
    pub stream_id: StreamId,
}

impl Iterator for BlockIter {
    type Item = (CheckBlockId, Vec<u8>);
    fn next(&mut self) -> Option<(CheckBlockId, Vec<u8>)> {
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
        Some((self.check_block_id - 1, check_block))
    }
}
