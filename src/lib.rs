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
        let aux_data = self.outer_encode(data, block_size, seed);
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
        let mut data = vec![0; (num_blocks + num_aux_blocks) * block_size]; // includes aux blocks
        let mut block_decoded = vec![false; num_blocks + num_aux_blocks];
        while !block_decoded.iter().all(|x| *x) {
            let mut progress_made = false;
            for (i, encoded_block) in encoded_data.chunks_exact(block_size).enumerate() {
                let associated_block_ids = get_associated_blocks(
                    i as u64,
                    &degree_distribution,
                    num_blocks + num_aux_blocks,
                );
                if let Some(target_block_id) =
                    block_to_decode(associated_block_ids.as_slice(), &block_decoded)
                {
                    eprintln!(
                        "using check block #{} to decode block #{}: {:?}",
                        i, target_block_id, &associated_block_ids
                    );
                    xor_block(
                        &mut data[target_block_id * block_size..],
                        encoded_block,
                        block_size,
                    );
                    for associated_block_id in associated_block_ids {
                        if associated_block_id != target_block_id {
                            for i in 0..block_size {
                                data[target_block_id * block_size + i] ^=
                                    data[associated_block_id * block_size + i];
                            }
                        }
                    }
                    block_decoded[target_block_id] = true;
                    progress_made = true;
                }
            }
            let (s_data, aux_data) = data.split_at_mut(num_blocks * block_size);
            for (i, aux_block) in aux_data.chunks_exact(block_size).enumerate() {
                let aux_id = i + num_blocks;
                if !block_decoded[aux_id] {
                    continue;
                }

                let associated_block_ids = aux_block_associations.get(&aux_id).unwrap();
                if let Some(target_block_id) =
                    block_to_decode(associated_block_ids.as_slice(), &block_decoded)
                {
                    eprintln!(
                        "using AUX block  #{} to decode block #{}: {:?}",
                        i, target_block_id, &associated_block_ids
                    );
                    xor_block(
                        &mut s_data[target_block_id * block_size..],
                        aux_block,
                        block_size,
                    );
                    for associated_block_id in associated_block_ids {
                        if *associated_block_id != target_block_id {
                            for i in 0..block_size {
                                s_data[target_block_id * block_size + i] ^=
                                    s_data[associated_block_id * block_size + i];
                            }
                        }
                    }
                    block_decoded[target_block_id] = true;
                    progress_made = true;
                }
            }
            if !progress_made {
                let _: usize = dbg!(block_decoded.iter().map(|b| if *b { 1 } else { 0 }).sum());
                panic!("could not complete decoding!")
            }
        }

        data.truncate(block_size * num_blocks);
        data
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
        for block_index in get_associated_blocks(
            self.block_id,
            &self.degree_distribution,
            num_blocks + num_aux_blocks,
        ) {
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
