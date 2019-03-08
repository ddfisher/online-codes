use rand::distributions::{Distribution, Uniform, WeightedIndex};
use rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;
use std::collections::{hash_map::Entry, HashMap, HashSet};

// TODO: use larger seeds for the PRNG
// TODO: allow specification of starting block_id
// TODO: write tests with proptest
// TODO: write benchmarks with criterion
// TODO: profile and fix low-hanging fruit
// TODO: reorder functions
// TODO: make a minor code cleanup pass
// TODO: write docs
// TODO: remove main.rs

enum UndecodedDegree {
    Zero,
    One(BlockIndex), // id of single block which hasn't yet been decoded
    Many(usize),     // number of blocks that haven't yet been decoded
}

// TODO: the IDs should be u128
type StreamId = u64;
type CheckBlockId = u64;
type BlockIndex = usize;

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

    pub fn encode<'a>(&self, data: &'a [u8], stream_id: StreamId) -> BlockIter<'a> {
        assert!(data.len() % self.block_size == 0);
        let aux_data = self.outer_encode(data, stream_id);
        self.inner_encode(data, aux_data, stream_id)
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
}

pub struct BlockIter<'a> {
    data: &'a [u8],
    aux_data: Vec<u8>,
    block_size: usize,
    degree_distribution: WeightedIndex<f64>,
    check_block_id: CheckBlockId,
    stream_id: StreamId,
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

impl OnlineCoder {
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

pub enum DecodeResult<'a> {
    Complete(Vec<u8>),
    InProgress(Decoder<'a>),
}

pub struct Decoder<'a> {
    num_blocks: usize,
    num_augmented_blocks: usize,
    block_size: usize,
    degree_distribution: WeightedIndex<f64>,
    stream_id: StreamId,
    unused_aux_block_adjacencies: HashMap<BlockIndex, (usize, Vec<BlockIndex>)>,

    augmented_data: Vec<u8>,
    blocks_decoded: Vec<bool>,
    num_undecoded_data_blocks: usize,
    unused_check_blocks: HashMap<CheckBlockId, (usize, &'a [u8])>,
    adjacent_check_blocks: HashMap<BlockIndex, Vec<CheckBlockId>>,
    decode_stack: Vec<(CheckBlockId, &'a [u8])>,
    aux_decode_stack: Vec<(BlockIndex, Vec<BlockIndex>)>,
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
            return Some(decoded_data);
        } else {
            // Decoding not yet complete.
            return None;
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
        return DecodeResult::InProgress(self);
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

impl<'a> DecodeResult<'a> {
    pub fn complete(self) -> Option<Vec<u8>> {
        match self {
            DecodeResult::Complete(v) => Some(v),
            DecodeResult::InProgress(_) => None,
        }
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

fn sample_with_exclusive_repeats(
    rng: &mut Xoshiro256StarStar,
    high_exclusive: usize,
    num: usize,
) -> Vec<usize> {
    let mut selected = HashSet::with_capacity(num);
    let distribution = Uniform::new(0, high_exclusive);
    for _ in 0..num {
        let sample = distribution.sample(rng);
        if !selected.insert(sample) {
            selected.remove(&sample);
        }
    }

    return selected.into_iter().collect();
}

// TODO: optimize
fn xor_block(dest: &mut [u8], src: &[u8], block_size: usize) {
    for i in 0..block_size {
        dest[i] ^= src[i];
    }
}

fn seed_stream_rng(stream_id: StreamId) -> Xoshiro256StarStar {
    seed_block_rng(stream_id, 0)
}

// TODO: don't lose bits when combining the stream id and block id
fn seed_block_rng(stream_id: StreamId, check_block_id: CheckBlockId) -> Xoshiro256StarStar {
    // Make sure the seed is a good, even mix of 0's and 1's.
    Xoshiro256StarStar::seed_from_u64(check_block_id.wrapping_add(stream_id))
}

fn get_adjacent_blocks(
    check_block_id: CheckBlockId,
    stream_id: StreamId,
    degree_distribution: &WeightedIndex<f64>,
    num_blocks: usize,
) -> Vec<BlockIndex> {
    let mut rng = seed_block_rng(stream_id, check_block_id);
    let degree = 1 + degree_distribution.sample(&mut rng);
    sample_with_exclusive_repeats(&mut rng, num_blocks, degree)
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

    return to_decode;
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

    return degree;
}
