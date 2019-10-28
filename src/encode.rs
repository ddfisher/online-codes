use std::collections::HashMap;
use rand::distributions::WeightedIndex;
use rand_xoshiro::Xoshiro256StarStar;
use crate::decode::Decoder;
use crate::util::{sample_with_exclusive_repeats, xor_block, seed_block_rng, get_adjacent_blocks};
use crate::types::{StreamId, BlockIndex, CheckBlockId};

#[derive(Debug)]
pub struct OnlineCoder {
    block_size: usize,
    epsilon: f64,
    q: usize,
}

pub struct BlockIter<'a> {
    pub data: &'a [u8],
    pub aux_data: Vec<u8>,
    pub block_size: usize,
    pub degree_distribution: WeightedIndex<f64>,
    pub check_block_id: CheckBlockId,
    pub stream_id: StreamId,
}

impl OnlineCoder {
    pub fn new(block_size: usize) -> OnlineCoder {
        Self::with_parameters(block_size, 0.01, 3)
    }

    pub fn with_parameters(block_size: usize, epsilon: f64, q: usize) -> OnlineCoder {
        OnlineCoder {
            block_size,
            epsilon,
            q,
        }
    }

    pub fn encode<'a>(&self, data: &'a [u8], stream_id: StreamId) -> BlockIter<'a> {
        assert!(data.len() % self.block_size == 0);
        let aux_data = self.outer_encode(data, stream_id);
        self.inner_encode(data, aux_data, stream_id)
    }

    pub fn decode<'a>(&self, num_blocks: usize, stream_id: StreamId) -> Decoder {
        let num_aux_blocks = self.num_aux_blocks(num_blocks);
        let num_augmented_blocks = num_blocks + num_aux_blocks;
        let unused_aux_block_adjacencies =
            self.get_aux_block_adjacencies(stream_id, num_blocks, num_aux_blocks);
        Decoder {
            num_blocks,
            num_augmented_blocks: num_blocks + num_aux_blocks,
            block_size: self.block_size,
            unused_aux_block_adjacencies,
            degree_distribution: make_degree_distribution(self.epsilon),
            stream_id,

            augmented_data: vec![0; num_augmented_blocks * self.block_size],
            blocks_decoded: vec![false; num_augmented_blocks],
            num_undecoded_data_blocks: num_blocks,
            unused_check_blocks: HashMap::new(),
            adjacent_check_blocks: HashMap::new(),
            decode_stack: Vec::new(),
            aux_decode_stack: Vec::new(),
        }
    }

    fn num_aux_blocks(&self, num_blocks: usize) -> usize {
        (0.55f64 * self.q as f64 * self.epsilon * num_blocks as f64).ceil() as usize
    }

    fn outer_encode(&self, data: &[u8], stream_id: StreamId) -> Vec<u8> {
        let num_blocks = data.len() / self.block_size;
        let num_aux_blocks = self.num_aux_blocks(num_blocks);
        let mut aux_data = vec![0; num_aux_blocks * self.block_size];
        let mut rng = seed_stream_rng(stream_id);
        for block in data.chunks_exact(self.block_size) {
            for aux_index in sample_with_exclusive_repeats(&mut rng, num_aux_blocks, self.q) {
                xor_block(
                    &mut aux_data[aux_index * self.block_size..],
                    block,
                    self.block_size,
                );
            }
        }
        aux_data
    }

    fn inner_encode<'a>(
        &self,
        data: &'a [u8],
        aux_data: Vec<u8>,
        stream_id: StreamId,
    ) -> BlockIter<'a> {
        BlockIter {
            data,
            aux_data,
            block_size: self.block_size,
            degree_distribution: make_degree_distribution(self.epsilon),
            check_block_id: 0,
            stream_id,
        }
    }

    fn get_aux_block_adjacencies(
        &self,
        stream_id: StreamId,
        num_blocks: usize,
        num_auxiliary_blocks: usize,
    ) -> HashMap<BlockIndex, (usize, Vec<BlockIndex>)> {
        let mut mapping: HashMap<BlockIndex, (usize, Vec<BlockIndex>)> = HashMap::new();
        let mut rng = seed_stream_rng(stream_id);
        for i in 0..num_blocks {
            for aux_index in sample_with_exclusive_repeats(&mut rng, num_auxiliary_blocks, self.q) {
                // TODO: clean up a bit
                let (num, ids) = &mut mapping.entry(aux_index + num_blocks).or_default();
                *num += 1;
                ids.push(i);
            }
        }
        mapping
    }
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

fn make_degree_distribution(epsilon: f64) -> WeightedIndex<f64> {
    // See section 3.2 of the Maymounkov-Mazières paper.
    let f = ((f64::ln(epsilon * epsilon / 4.0)) / f64::ln(1.0 - epsilon / 2.0)).ceil() as usize;
    let mut p = Vec::with_capacity(f);
    let p1 = 1.0 - ((1.0 + 1.0 / f as f64) / (1.0 + epsilon));
    p.push(p1);
    // Extracted unchanging constant from p_i's.
    let c = (1.0 - p1) * f as f64 / (f - 1) as f64;
    for i in 2..=f {
        p.push(c / (i * (i - 1)) as f64);
    }
    WeightedIndex::new(&p).expect("serious probability calculation error")
}

fn seed_stream_rng(stream_id: StreamId) -> Xoshiro256StarStar {
    seed_block_rng(stream_id, 0)
}
