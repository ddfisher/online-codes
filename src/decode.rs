use std::collections::{hash_map::Entry, HashMap};
use rand::distributions::WeightedIndex;
use crate::types::{StreamId, BlockIndex, CheckBlockId};
use crate::util::{get_adjacent_blocks, xor_block};

pub enum DecodeResult<'a> {
    Complete(Vec<u8>),
    InProgress(Decoder<'a>),
}

enum UndecodedDegree {
    Zero,
    One(BlockIndex), // id of single block which hasn't yet been decoded
    Many(usize),     // number of blocks that haven't yet been decoded
}

#[derive(Debug)]
pub struct Decoder<'a> {
    pub num_blocks: usize,
    pub num_augmented_blocks: usize,
    pub block_size: usize,
    pub degree_distribution: WeightedIndex<f64>,
    pub stream_id: StreamId,
    pub unused_aux_block_adjacencies: HashMap<BlockIndex, (usize, Vec<BlockIndex>)>,

    pub augmented_data: Vec<u8>,
    pub blocks_decoded: Vec<bool>,
    pub num_undecoded_data_blocks: usize,
    pub unused_check_blocks: HashMap<CheckBlockId, (usize, &'a [u8])>,
    pub adjacent_check_blocks: HashMap<BlockIndex, Vec<CheckBlockId>>,
    pub decode_stack: Vec<(CheckBlockId, &'a [u8])>,
    pub aux_decode_stack: Vec<(BlockIndex, Vec<BlockIndex>)>,
}

impl<'a> DecodeResult<'a> {
    pub fn complete(self) -> Option<Vec<u8>> {
        match self {
            DecodeResult::Complete(v) => Some(v),
            DecodeResult::InProgress(_) => None,
        }
    }
}

impl<'a> Decoder<'a> {
    pub fn decode_block(
        &mut self,
        check_block_id: CheckBlockId,
        check_block: &'a [u8],
    ) -> Option<Vec<u8>> {
        if self.num_undecoded_data_blocks == 0 {
            // Decoding has already finished and the decoded data has already been returned.
            return None;
        }

        // TODO: don't immediately push then pop off the decode stack
        self.decode_stack.push((check_block_id, check_block));

        while let Some((check_block_id, check_block)) = self.decode_stack.pop() {
            let adjacent_blocks = get_adjacent_blocks(
                check_block_id,
                self.stream_id,
                &self.degree_distribution,
                self.num_augmented_blocks,
            );
            match undecoded_degree(&adjacent_blocks, &self.blocks_decoded) {
                UndecodedDegree::Zero => { /* This check block contains no new information. */ }
                UndecodedDegree::One(target_block_index) => {
                    decode_from_check_block(
                        target_block_index,
                        check_block,
                        &adjacent_blocks,
                        &mut self.augmented_data,
                        self.block_size,
                    );
                    self.blocks_decoded[target_block_index] = true;
                    if target_block_index < self.num_blocks {
                        self.num_undecoded_data_blocks -= 1;
                    } else {
                        // Decoded an aux block.
                        // If that aux block can be used to decode a data block, schedule it for
                        // decoding.
                        if let Entry::Occupied(mut unused_aux_entry) =
                            self.unused_aux_block_adjacencies.entry(target_block_index)
                        {
                            let remaining_degree = &mut unused_aux_entry.get_mut().0;
                            *remaining_degree -= 1;
                            if *remaining_degree == 1 {
                                self.aux_decode_stack
                                    .push((target_block_index, unused_aux_entry.remove().1));
                            }
                        }
                    }
                    if let Some(adjacent_check_block_ids) =
                        self.adjacent_check_blocks.remove(&target_block_index)
                    {
                        for check_block_id in adjacent_check_block_ids {
                            if let Entry::Occupied(mut unused_block_entry) =
                                self.unused_check_blocks.entry(check_block_id)
                            {
                                let remaining_degree = &mut unused_block_entry.get_mut().0;
                                *remaining_degree -= 1;
                                if *remaining_degree == 1 {
                                    self.decode_stack
                                        .push((check_block_id, unused_block_entry.remove().1));
                                }
                            }
                        }
                    };
                }
                UndecodedDegree::Many(degree) => {
                    self.unused_check_blocks
                        .insert(check_block_id, (degree, check_block));
                    for block_index in adjacent_blocks {
                        self.adjacent_check_blocks
                            .entry(block_index)
                            .or_default()
                            .push(check_block_id)
                    }
                }
            }
        }

        while let Some((aux_block_index, adjacent_blocks)) = self.aux_decode_stack.pop() {
            if let Some(decoded_block_id) = decode_aux_block(
                aux_block_index,
                &adjacent_blocks,
                &mut self.augmented_data,
                self.block_size,
                &self.blocks_decoded,
            ) {
                self.blocks_decoded[decoded_block_id] = true;
                self.num_undecoded_data_blocks -= 1;
            }
        }

        if self.num_undecoded_data_blocks == 0 {
            // Decoding finished -- return decoded data.
            let mut decoded_data = std::mem::replace(&mut self.augmented_data, Vec::new());
            decoded_data.truncate(self.block_size * self.num_blocks);
            Some(decoded_data)
        } else {
            // Decoding not yet complete.
            None
        }
    }

    pub fn from_iter<T>(mut self, iter: T) -> DecodeResult<'a>
    where
        T: IntoIterator<Item = (CheckBlockId, &'a [u8])>,
    {
        for (check_block_id, check_block) in iter {
            if let Some(decoded_data) = self.decode_block(check_block_id, check_block) {
                return DecodeResult::Complete(decoded_data);
            }
        }
        DecodeResult::InProgress(self)
    }

    pub fn get_incomplete_result(&self) -> (&[bool], &[u8]) {
        (
            &self.blocks_decoded[0..self.num_blocks],
            &self.augmented_data[0..self.block_size * self.num_blocks],
        )
    }

    pub fn into_incomplete_result(mut self) -> (Vec<bool>, Vec<u8>) {
        self.blocks_decoded.truncate(self.num_blocks);
        self.augmented_data
            .truncate(self.num_blocks * self.block_size);
        (self.blocks_decoded, self.augmented_data)
    }
}

fn decode_from_check_block(
    target_block_index: BlockIndex,
    check_block: &[u8],
    adjacent_blocks: &[BlockIndex],
    augmented_data: &mut [u8],
    block_size: usize,
) {
    xor_block(
        &mut augmented_data[target_block_index * block_size..],
        check_block,
        block_size,
    );
    xor_adjacent_blocks(
        target_block_index,
        adjacent_blocks,
        augmented_data,
        block_size,
    );
}

fn decode_aux_block(
    index: BlockIndex,
    adjacent_blocks: &[BlockIndex],
    augmented_data: &mut [u8],
    block_size: usize,
    blocks_decoded: &[bool],
) -> Option<BlockIndex> {
    block_to_decode(adjacent_blocks, blocks_decoded).map(|target_block_index| {
        for i in 0..block_size {
            augmented_data[target_block_index * block_size + i] ^=
                augmented_data[index * block_size + i];
        }
        xor_adjacent_blocks(
            target_block_index,
            adjacent_blocks,
            augmented_data,
            block_size,
        );
        target_block_index
    })
}

fn xor_adjacent_blocks(
    target_block_index: BlockIndex,
    adjacent_blocks: &[BlockIndex],
    augmented_data: &mut [u8],
    block_size: usize,
) {
    for block_index in adjacent_blocks {
        if *block_index != target_block_index {
            for i in 0..block_size {
                augmented_data[target_block_index * block_size + i] ^=
                    augmented_data[block_index * block_size + i];
            }
        }
    }
}

fn undecoded_degree(adjacent_block_ids: &[BlockIndex], blocks_decoded: &[bool]) -> UndecodedDegree {
    // If exactly one of the adjacent blocks is not yet decoded, return the id of that block.
    let mut degree = UndecodedDegree::Zero;
    for block_index in adjacent_block_ids {
        if !blocks_decoded[*block_index] {
            degree = match degree {
                UndecodedDegree::Zero => UndecodedDegree::One(*block_index),
                UndecodedDegree::One(_) => UndecodedDegree::Many(2),
                UndecodedDegree::Many(n) => UndecodedDegree::Many(n + 1),
            }
        }
    }

    degree
}

fn block_to_decode(adjacent_blocks: &[BlockIndex], block_decoded: &[bool]) -> Option<BlockIndex> {
    // If exactly one of the adjacent blocks is not yet decoded, return the id of that block.
    let mut to_decode = None;
    for block_index in adjacent_blocks {
        if !block_decoded[*block_index] {
            if to_decode.is_some() {
                return None;
            }
            to_decode = Some(*block_index)
        }
    }

    to_decode
}

