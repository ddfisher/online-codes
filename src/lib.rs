use log::{debug, trace};
use rand::distributions::{Distribution, Uniform, WeightedIndex};
use rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;
use std::collections::{HashMap, HashSet};

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}

enum UndecodedDegree {
    Zero,
    One(usize),  // id of single block which hasn't yet been decoded
    Many(usize), // number of blocks that haven't yet been decoded
}

pub struct OnlineCoder {
    epsilon: f64,
    q: usize,
}

impl OnlineCoder {
    pub fn new() -> OnlineCoder {
        Self::with_parameters(0.01, 3)
    }

    pub fn with_parameters(epsilon: f64, q: usize) -> OnlineCoder {
        OnlineCoder { epsilon, q }
    }

    pub fn encode<'a>(&self, data: &'a [u8], block_size: usize, seed: u64) -> BlockIter<'a> {
        assert!(data.len() % block_size == 0);
        trace!("data: {:X?}", data);
        let aux_data = self.outer_encode(data, block_size, seed);
        trace!("aux data: {:X?}", data);
        self.inner_encode(data, aux_data, block_size)
    }

    fn num_aux_blocks(&self, num_blocks: usize) -> usize {
        (0.55f64 * self.q as f64 * self.epsilon * num_blocks as f64).ceil() as usize
    }

    fn outer_encode(&self, data: &[u8], block_size: usize, seed: u64) -> Vec<u8> {
        let num_blocks = data.len() / block_size;
        let num_aux_blocks = self.num_aux_blocks(num_blocks);
        let mut aux_data = vec![0; num_aux_blocks * block_size];
        let mut rng = Xoshiro256StarStar::seed_from_u64(seed);
        for block in data.chunks_exact(block_size) {
            for aux_index in sample_with_exclusive_repeats(&mut rng, num_aux_blocks, self.q) {
                xor_block(&mut aux_data[aux_index * block_size..], block, block_size);
            }
        }
        aux_data
    }

    fn inner_encode<'a>(
        &self,
        data: &'a [u8],
        aux_data: Vec<u8>,
        block_size: usize,
    ) -> BlockIter<'a> {
        BlockIter {
            data,
            aux_data,
            block_size,
            degree_distribution: make_degree_distribution(self.epsilon),
            block_id: 0,
        }
    }
}

pub struct BlockIter<'a> {
    data: &'a [u8],
    aux_data: Vec<u8>,
    block_size: usize,
    degree_distribution: WeightedIndex<f64>,
    block_id: u64, // TODO: this should be a larger size type
}

impl<'a> Iterator for BlockIter<'a> {
    type Item = Vec<u8>;
    fn next(&mut self) -> Option<Vec<u8>> {
        let num_blocks = self.data.len() / self.block_size;
        let num_aux_blocks = self.aux_data.len() / self.block_size;
        let mut check_block = vec![0; self.block_size];
        let associated_blocks = get_associated_blocks(
            self.block_id,
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

        self.block_id += 1;
        Some(check_block)
    }
}

impl OnlineCoder {
    // TODO: implement in a vaguely optimized way
    pub fn decode<'a>(
        &self,
        encoded_data: &'a Vec<u8>,
        num_blocks: usize,
        block_size: usize,
        seed: u64,
    ) -> Vec<u8> {
        let num_aux_blocks = self.num_aux_blocks(num_blocks);
        let aux_block_associations =
            self.get_aux_block_associations(seed, num_blocks, num_aux_blocks);
        let degree_distribution = make_degree_distribution(self.epsilon);
        let mut augmented_data = vec![0; (num_blocks + num_aux_blocks) * block_size]; // includes aux blocks
        let mut blocks_decoded = vec![false; num_blocks + num_aux_blocks];
        let mut num_undecoded_data_blocks = num_blocks;
        let mut unused_check_blocks: HashMap<u64, (usize, &[u8])> = HashMap::new();
        let mut block_dependencies: HashMap<usize, Vec<u64>> = HashMap::new();
        let mut decode_stack: Vec<(u64, &[u8])> = Vec::new();

        for (i, check_block) in encoded_data.chunks_exact(block_size).enumerate() {
            decode_stack.push((i as u64, check_block));
        }
        while let Some((check_block_id, check_block)) = decode_stack.pop() {
            let associated_block_ids = get_associated_blocks(
                check_block_id,
                &degree_distribution,
                num_blocks + num_aux_blocks,
            );
            match undecoded_degree(&associated_block_ids, &blocks_decoded) {
                UndecodedDegree::Zero => { /* This block has already been decoded. */ }
                UndecodedDegree::One(target_block_id) => {
                    decode_from_check_block(
                        target_block_id,
                        check_block,
                        &associated_block_ids,
                        &mut augmented_data,
                        block_size,
                    );
                    blocks_decoded[target_block_id] = true;
                    if target_block_id < num_blocks {
                        num_undecoded_data_blocks -= 1;
                    }
                    block_dependencies
                        .remove(&target_block_id)
                        .map(|dependencies| {
                            for depending_block_id in dependencies {
                                if let Some((remaining_degree, _)) =
                                    &mut unused_check_blocks.get_mut(&depending_block_id)
                                {
                                    *remaining_degree -= 1;
                                    if *remaining_degree == 1 {
                                        decode_stack.push((
                                            depending_block_id,
                                            unused_check_blocks
                                                .remove(&depending_block_id)
                                                .unwrap()
                                                .1, //TODO: use entry
                                        ));
                                    }
                                }
                            }
                        });
                }
                UndecodedDegree::Many(degree) => {
                    unused_check_blocks.insert(check_block_id, (degree, check_block));
                    for associated_block_id in associated_block_ids {
                        block_dependencies
                            .entry(associated_block_id)
                            .or_default()
                            .push(check_block_id) // TODO: consider switching to storing pointers
                    }
                }
            }
        }
        for aux_block_id in num_blocks..(num_blocks + num_aux_blocks) {
            if !blocks_decoded[aux_block_id] {
                continue;
            }

            let associated_block_ids = aux_block_associations.get(&aux_block_id).unwrap();
            if let Some(decoded_block_id) = decode_aux_block(
                aux_block_id,
                &associated_block_ids,
                &mut augmented_data,
                block_size,
                &blocks_decoded,
            ) {
                blocks_decoded[decoded_block_id] = true;
                num_undecoded_data_blocks -= 1;
            }
        }
        if num_undecoded_data_blocks == 0 {
            augmented_data.truncate(block_size * num_blocks);
            return augmented_data;
        } else if blocks_decoded.iter().take(num_blocks).all(|b| *b) {
            let _: usize = dbg!(blocks_decoded.iter().map(|b| if *b { 1 } else { 0 }).sum());
            let _: usize = dbg!(blocks_decoded
                .iter()
                .take(num_blocks)
                .map(|b| if *b { 0 } else { 1 })
                .sum());
            panic!("SOMETHING HAS GONE VERY WRONG");
        } else {
            let _: usize = dbg!(blocks_decoded.iter().map(|b| if *b { 1 } else { 0 }).sum());
            let _: usize = dbg!(blocks_decoded
                .iter()
                .take(num_blocks)
                .map(|b| if *b { 0 } else { 1 })
                .sum());
            dbg!(unused_check_blocks.len());
            panic!("could not complete decoding!")
        }
    }

    fn get_aux_block_associations(
        &self,
        seed: u64,
        num_blocks: usize,
        num_auxiliary_blocks: usize,
    ) -> HashMap<usize, Vec<usize>> {
        let mut mapping: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut rng = Xoshiro256StarStar::seed_from_u64(seed);
        for i in 0..num_blocks {
            for aux_index in sample_with_exclusive_repeats(&mut rng, num_auxiliary_blocks, self.q) {
                mapping.entry(aux_index + num_blocks).or_default().push(i);
            }
        }
        mapping
    }
}

fn decode_from_check_block(
    block_id: usize,
    check_block: &[u8],
    associated_block_ids: &[usize],
    augmented_data: &mut [u8],
    block_size: usize,
) {
    // eprintln!(
    //     "using check block #{} to decode block #{}: {:?}",
    //     i, target_block_id, &associated_block_ids
    // );
    xor_block(
        &mut augmented_data[block_id * block_size..],
        check_block,
        block_size,
    );
    xor_associated_blocks(block_id, associated_block_ids, augmented_data, block_size);
}

fn decode_aux_block(
    aux_block_id: usize,
    associated_block_ids: &[usize],
    augmented_data: &mut [u8],
    block_size: usize,
    blocks_decoded: &[bool],
) -> Option<usize> {
    // eprintln!(
    //     "using AUX block  #{} to decode block #{}: {:?}",
    //     i, target_block_id, &associated_block_ids
    // );
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
    target_block_id: usize,
    associated_block_ids: &[usize],
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
// TODO: there should be more involved with block_id
fn get_associated_blocks(
    block_id: u64,
    degree_distribution: &WeightedIndex<f64>,
    num_blocks: usize,
) -> Vec<usize> {
    let mut rng = Xoshiro256StarStar::seed_from_u64(block_id);
    let degree = 1 + degree_distribution.sample(&mut rng);
    sample_with_exclusive_repeats(&mut rng, num_blocks, degree)
}

fn block_to_decode(associated_block_ids: &[usize], block_decoded: &[bool]) -> Option<usize> {
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

fn undecoded_degree(associated_block_ids: &[usize], blocks_decoded: &[bool]) -> UndecodedDegree {
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
