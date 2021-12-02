use crate::block_iter::BlockIter;
use crate::types::StreamId;
use crate::util::{
    make_degree_distribution, sample_with_exclusive_repeats, seed_stream_rng, xor_block,
};

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
        let aux_data = self.outer_encode(&data, stream_id);
        self.inner_encode(data, aux_data, stream_id)
    }

    fn num_aux_blocks(&self, num_blocks: usize) -> usize {
        (0.55_f64 * self.q as f64 * self.epsilon * num_blocks as f64).ceil() as usize
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
