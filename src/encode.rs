use crate::block_iter::BlockIter;
use crate::decode::Decoder;
use crate::types::StreamId;
use crate::util::{
    get_aux_block_adjacencies, make_degree_distribution, sample_with_exclusive_repeats,
    seed_stream_rng, xor_block,
};
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct OnlineCoder {
    block_size: usize,
    epsilon: f64,
    q: usize,
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

    pub fn encode(&self, data: Vec<u8>, stream_id: StreamId) -> BlockIter {
        assert!(data.len() % self.block_size == 0);
        let aux_data = self.outer_encode(&data, stream_id);
        self.inner_encode(data, aux_data, stream_id)
    }

    pub fn decode(&self, num_blocks: usize, stream_id: StreamId) -> Decoder {
        let num_aux_blocks = self.num_aux_blocks(num_blocks);
        let num_augmented_blocks = num_blocks + num_aux_blocks;
        let unused_aux_block_adjacencies =
            get_aux_block_adjacencies(stream_id, num_blocks, num_aux_blocks, self.q);
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

    fn inner_encode(&self, data: Vec<u8>, aux_data: Vec<u8>, stream_id: StreamId) -> BlockIter {
        BlockIter {
            data,
            aux_data,
            block_size: self.block_size,
            degree_distribution: make_degree_distribution(self.epsilon),
            check_block_id: 0,
            stream_id,
        }
    }
}
