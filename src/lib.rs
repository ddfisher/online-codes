use log::trace;
use rand::distributions::{Distribution, Uniform, WeightedIndex};
use rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;
use std::collections::{hash_map::Entry, HashMap, HashSet};

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}

enum UndecodedDegree {
    Zero,
    One(BlockIndex), // id of single block which hasn't yet been decoded
    Many(usize),     // number of blocks that haven't yet been decoded
}

// TODO: these should be larger types
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
        trace!("data: {:X?}", data);
        let aux_data = self.outer_encode(data, stream_id);
        trace!("aux data: {:X?}", data);
        self.inner_encode(data, aux_data)
    }

    fn num_aux_blocks(&self, num_blocks: usize) -> usize {
        (0.55f64 * self.q as f64 * self.epsilon * num_blocks as f64).ceil() as usize
    }

    fn outer_encode(&self, data: &[u8], stream_id: StreamId) -> Vec<u8> {
        let num_blocks = data.len() / self.block_size;
        let num_aux_blocks = self.num_aux_blocks(num_blocks);
        let mut aux_data = vec![0; num_aux_blocks * self.block_size];
        let mut rng = Xoshiro256StarStar::seed_from_u64(stream_id);
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

    fn inner_encode<'a>(&self, data: &'a [u8], aux_data: Vec<u8>) -> BlockIter<'a> {
        BlockIter {
            data,
            aux_data,
            block_size: self.block_size,
            degree_distribution: make_degree_distribution(self.epsilon),
            check_block_id: 0,
        }
    }
}

pub struct BlockIter<'a> {
    data: &'a [u8],
    aux_data: Vec<u8>,
    block_size: usize,
    degree_distribution: WeightedIndex<f64>,
    check_block_id: CheckBlockId,
}

impl<'a> Iterator for BlockIter<'a> {
    type Item = Vec<u8>;
    fn next(&mut self) -> Option<Vec<u8>> {
        let num_blocks = self.data.len() / self.block_size;
        let num_aux_blocks = self.aux_data.len() / self.block_size;
        let mut check_block = vec![0; self.block_size];
        let associated_blocks = get_associated_blocks(
            self.check_block_id,
            &self.degree_distribution,
            num_blocks + num_aux_blocks,
        );
        trace!(
            "encoding check block from associated blocks {:?}",
            associated_blocks
        );
        for block_index in associated_blocks {
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
    // TODO: implement in a vaguely optimized way
    pub fn decode<'a>(&self, num_blocks: usize, stream_id: StreamId) -> Decoder {
        let num_aux_blocks = self.num_aux_blocks(num_blocks);
        let num_augmented_blocks = num_blocks + num_aux_blocks;
        let aux_block_associations =
            self.get_aux_block_associations(stream_id, num_blocks, num_aux_blocks);
        Decoder {
            num_blocks,
            num_augmented_blocks: num_blocks + num_aux_blocks,
            block_size: self.block_size,
            aux_block_associations,
            degree_distribution: make_degree_distribution(self.epsilon),

            augmented_data: vec![0; num_augmented_blocks * self.block_size],
            blocks_decoded: vec![false; num_augmented_blocks],
            num_undecoded_data_blocks: num_blocks,
            unused_check_blocks: HashMap::new(),
            block_dependencies: HashMap::new(),
            decode_stack: Vec::new(),
        }
    }

    fn get_aux_block_associations(
        &self,
        stream_id: StreamId,
        num_blocks: usize,
        num_auxiliary_blocks: usize,
    ) -> HashMap<BlockIndex, Vec<BlockIndex>> {
        let mut mapping: HashMap<BlockIndex, Vec<BlockIndex>> = HashMap::new();
        let mut rng = Xoshiro256StarStar::seed_from_u64(stream_id);
        for i in 0..num_blocks {
            for aux_index in sample_with_exclusive_repeats(&mut rng, num_auxiliary_blocks, self.q) {
                mapping.entry(aux_index + num_blocks).or_default().push(i);
            }
        }
        mapping
    }
}

pub struct Decoder<'a> {
    num_blocks: usize,
    num_augmented_blocks: usize,
    block_size: usize,
    degree_distribution: WeightedIndex<f64>,
    aux_block_associations: HashMap<BlockIndex, Vec<BlockIndex>>,

    augmented_data: Vec<u8>,
    blocks_decoded: Vec<bool>,
    num_undecoded_data_blocks: usize,
    unused_check_blocks: HashMap<CheckBlockId, (usize, &'a [u8])>,
    block_dependencies: HashMap<BlockIndex, Vec<CheckBlockId>>,
    decode_stack: Vec<(CheckBlockId, &'a [u8])>,
}

impl<'a> Decoder<'a> {
    pub fn decode_chunk(&mut self, check_block_id: CheckBlockId, check_block: &'a [u8]) -> bool {
        // TODO: consider if this should take in a slice or a Vec
        // TODO: consider if this function should copy the slice if need be so it's not required to
        // live for the lifetime of the decoder

        // TODO: don't immediately push then pop off the decode stack
        self.decode_stack.push((check_block_id, check_block));

        while let Some((check_block_id, check_block)) = self.decode_stack.pop() {
            let associated_block_ids = get_associated_blocks(
                check_block_id,
                &self.degree_distribution,
                self.num_augmented_blocks,
            );
            match undecoded_degree(&associated_block_ids, &self.blocks_decoded) {
                UndecodedDegree::Zero => { /* This check block contains no new information. */ }
                UndecodedDegree::One(target_block_id) => {
                    decode_from_check_block(
                        target_block_id,
                        check_block,
                        &associated_block_ids,
                        &mut self.augmented_data,
                        self.block_size,
                    );
                    self.blocks_decoded[target_block_id] = true;
                    if target_block_id < self.num_blocks {
                        self.num_undecoded_data_blocks -= 1;
                    }
                    if let Some(dependencies) = self.block_dependencies.remove(&target_block_id) {
                        for depending_block_id in dependencies {
                            if let Entry::Occupied(mut unused_block_entry) =
                                self.unused_check_blocks.entry(depending_block_id)
                            {
                                let remaining_degree = &mut unused_block_entry.get_mut().0;
                                *remaining_degree -= 1;
                                if *remaining_degree == 1 {
                                    self.decode_stack
                                        .push((depending_block_id, unused_block_entry.remove().1));
                                }
                            }
                        }
                    };
                }
                UndecodedDegree::Many(degree) => {
                    self.unused_check_blocks
                        .insert(check_block_id, (degree, check_block));
                    for associated_block_id in associated_block_ids {
                        self.block_dependencies
                            .entry(associated_block_id)
                            .or_default()
                            .push(check_block_id) // TODO: consider switching to storing pointers
                    }
                }
            }
        }
        for aux_block_id in self.num_blocks..self.num_augmented_blocks {
            if !self.blocks_decoded[aux_block_id] {
                continue;
            }

            let associated_block_ids = self.aux_block_associations.get(&aux_block_id).unwrap();
            if let Some(decoded_block_id) = decode_aux_block(
                aux_block_id,
                &associated_block_ids,
                &mut self.augmented_data,
                self.block_size,
                &self.blocks_decoded,
            ) {
                self.blocks_decoded[decoded_block_id] = true;
                self.num_undecoded_data_blocks -= 1;
            }
        }
        self.num_undecoded_data_blocks == 0
    }

    pub fn get_result(mut self) -> Option<Vec<u8>> {
        if self.num_undecoded_data_blocks == 0 {
            self.augmented_data
                .truncate(self.block_size * self.num_blocks);
            Some(self.augmented_data)
        } else {
            None
        }
    }
}

fn decode_from_check_block(
    target_block_id: BlockIndex,
    check_block: &[u8],
    associated_block_ids: &[BlockIndex],
    augmented_data: &mut [u8],
    block_size: usize,
) {
    xor_block(
        &mut augmented_data[target_block_id * block_size..],
        check_block,
        block_size,
    );
    xor_associated_blocks(
        target_block_id,
        associated_block_ids,
        augmented_data,
        block_size,
    );
}

fn decode_aux_block(
    aux_block_id: BlockIndex,
    associated_block_ids: &[BlockIndex],
    augmented_data: &mut [u8],
    block_size: usize,
    blocks_decoded: &[bool],
) -> Option<BlockIndex> {
    block_to_decode(associated_block_ids, blocks_decoded).map(|target_block_id| {
        for i in 0..block_size {
            augmented_data[target_block_id * block_size + i] ^=
                augmented_data[aux_block_id * block_size + i];
        }
        xor_associated_blocks(
            target_block_id,
            associated_block_ids,
            augmented_data,
            block_size,
        );
        target_block_id
    })
}

fn xor_associated_blocks(
    target_block_id: BlockIndex,
    associated_block_ids: &[BlockIndex],
    augmented_data: &mut [u8],
    block_size: usize,
) {
    for associated_block_id in associated_block_ids {
        if *associated_block_id != target_block_id {
            for i in 0..block_size {
                augmented_data[target_block_id * block_size + i] ^=
                    augmented_data[associated_block_id * block_size + i];
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

// TODO: return an iterator instead
fn get_associated_blocks(
    check_block_id: CheckBlockId,
    degree_distribution: &WeightedIndex<f64>,
    num_blocks: usize,
) -> Vec<BlockIndex> {
    // TODO: this should use the stream id too
    let mut rng = Xoshiro256StarStar::seed_from_u64(check_block_id);
    let degree = 1 + degree_distribution.sample(&mut rng);
    sample_with_exclusive_repeats(&mut rng, num_blocks, degree)
}

fn block_to_decode(
    associated_block_ids: &[BlockIndex],
    block_decoded: &[bool],
) -> Option<BlockIndex> {
    // If exactly one of the associated blocks is not yet decoded, return the id of that block.
    let mut to_decode = None;
    for block_id in associated_block_ids {
        if !block_decoded[*block_id] {
            if to_decode.is_some() {
                return None;
            }
            to_decode = Some(*block_id)
        }
    }

    return to_decode;
}

fn undecoded_degree(
    associated_block_ids: &[BlockIndex],
    blocks_decoded: &[bool],
) -> UndecodedDegree {
    // If exactly one of the associated blocks is not yet decoded, return the id of that block.
    let mut degree = UndecodedDegree::Zero;
    for block_id in associated_block_ids {
        if !blocks_decoded[*block_id] {
            degree = match degree {
                UndecodedDegree::Zero => UndecodedDegree::One(*block_id),
                UndecodedDegree::One(_) => UndecodedDegree::Many(2),
                UndecodedDegree::Many(n) => UndecodedDegree::Many(n + 1),
            }
        }
    }

    return degree;
}
