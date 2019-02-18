use rand::distributions::{Distribution, Uniform, WeightedIndex};
use rand_core::{RngCore, SeedableRng};
use rand_xoshiro::Xoshiro256StarStar;
use std::collections::{HashMap, HashSet};

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}

// TODO: allow user to change these
const q: usize = 3;
const epsilon: f64 = 0.01;

// TODO: optimize
fn sample_without_replacement(
    rng: &mut Xoshiro256StarStar,
    high_exclusive: usize,
    num: usize,
) -> Vec<usize> {
    // TODO: this might have bad behavior when called by inner_encode with a high degree
    let mut selected = HashSet::with_capacity(num);
    let distribution = Uniform::new(0, high_exclusive);
    let mut sample = distribution.sample(rng);
    while selected.contains(&sample) {
        sample = distribution.sample(rng);
    }
    selected.insert(sample);

    selected.into_iter().collect()
}

// TODO: optimize
fn xor_block(dest: &mut [u8], src: &[u8], block_size: usize) {
    for i in 0..block_size {
        dest[i] ^= src[i];
    }
}

pub fn encode<'a>(data: &'a mut Vec<u8>, block_size: usize, seed: u64) -> BlockIter<'a> {
    outer_encode(data, block_size, seed);
    inner_encode(data, block_size)
}

fn outer_encode(data: &mut Vec<u8>, block_size: usize, seed: u64) {
    assert!(data.len() % block_size == 0);
    let num_blocks = data.len() / block_size;
    let num_auxiliary_blocks = (0.55f64 * q as f64 * epsilon * num_blocks as f64).ceil() as usize;
    data.resize((num_blocks + num_auxiliary_blocks) * block_size, 0u8);
    let (blocks, aux_blocks) = data.split_at_mut(num_blocks * block_size);
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed);
    for block in blocks.chunks_exact(block_size) {
        for aux_index in sample_without_replacement(&mut rng, num_auxiliary_blocks, q) {
            xor_block(
                &mut aux_blocks[aux_index * block_size..(aux_index + 1) * block_size],
                block,
                block_size,
            );
        }
    }
}

fn get_degree_distibution() -> WeightedIndex<f64> {
    // TODO: only compute this once
    // See section 3.2 of the Maymounkov-Mazières paper.
    let f = ((f64::ln(epsilon * epsilon / 4.0)) / f64::ln(1.0 - epsilon / 2.0)).ceil() as usize;
    let mut p = Vec::with_capacity(f);
    let p1 = 1.0 - ((1.0 + 1.0 / f as f64) / 1.0 + epsilon);
    p.push(p1);
    // Extracted unchanging constant from p_i's.
    let c = (1.0 - p1) * f as f64 / (f - 1) as f64;
    for i in 2..=f {
        p.push(c / (i * (i - 1)) as f64);
    }
    WeightedIndex::new(&p).expect("serious probability calculation error")
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
    sample_without_replacement(&mut rng, num_blocks, degree)
}

// TODO: yikes this is not efficient
fn add_aux_associations(
    associated_block_ids: Vec<usize>,
    aux_associations: &HashMap<usize, Vec<usize>>,
) -> Vec<usize> {
    let assoc_set: HashSet<usize> = associated_block_ids.into_iter().collect();
    let mut change_set = HashSet::new();
    for assoc_id in assoc_set.iter() {
        if let Some(block_ids) = aux_associations.get(assoc_id) {
            for block_id in block_ids {
                if !change_set.remove(block_id) {
                    change_set.insert(*block_id);
                }
            }
        }
    }
    assoc_set
        .symmetric_difference(&change_set)
        .map(|x| *x)
        .collect()
}

fn get_aux_block_associations(
    seed: u64,
    num_blocks: usize,
    num_auxiliary_blocks: usize,
) -> HashMap<usize, Vec<usize>> {
    let mut mapping: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed);
    for i in 0..num_blocks {
        for aux_index in sample_without_replacement(&mut rng, num_auxiliary_blocks, q) {
            mapping.entry(aux_index).or_default().push(i);
        }
    }
    mapping
}

pub struct BlockIter<'a> {
    data: &'a Vec<u8>,
    block_size: usize,
    degree_distribution: WeightedIndex<f64>,
    block_id: u64, // TODO: this should be a larger size type
}

impl<'a> Iterator for BlockIter<'a> {
    type Item = Vec<u8>;
    fn next(&mut self) -> Option<Vec<u8>> {
        let num_blocks = self.data.len() / self.block_size;
        let mut check_block = vec![0; self.block_size];
        for block_index in
            get_associated_blocks(self.block_id, &self.degree_distribution, num_blocks)
        {
            xor_block(
                &mut check_block,
                &self.data[block_index * self.block_size..(block_index + 1) * self.block_size],
                self.block_size,
            );
        }

        self.block_id += 1;
        Some(check_block)
    }
}

fn inner_encode<'a>(data: &'a mut Vec<u8>, block_size: usize) -> BlockIter<'a> {
    assert!(data.len() % block_size == 0);

    BlockIter {
        data,
        block_size,
        degree_distribution: get_degree_distibution(),
        block_id: 0,
    }
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

// TODO: implement in a vaguely optimized way
pub fn decode<'a>(
    encoded_data: &'a Vec<u8>,
    num_blocks: usize,
    block_size: usize,
    seed: u64,
) -> Vec<u8> {
    let num_auxiliary_blocks = (0.55f64 * q as f64 * epsilon * num_blocks as f64).ceil() as usize;
    let aux_block_associations = get_aux_block_associations(seed, num_blocks, num_auxiliary_blocks);
    let mut data = vec![0; num_blocks * block_size];
    let mut block_decoded = vec![false; num_blocks];
    let degree_distribution = get_degree_distibution();
    while !block_decoded.iter().all(|x| *x) {
        let mut progress_made = false;
        for (i, encoded_block) in encoded_data.chunks_exact(block_size).enumerate() {
            let associated_block_ids = add_aux_associations(
                get_associated_blocks(i as u64, &degree_distribution, num_blocks),
                &aux_block_associations,
            );
            if let Some(target_block_id) =
                block_to_decode(associated_block_ids.as_slice(), &block_decoded)
            {
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
        if !progress_made {
            panic!("could not complete decoding!")
        }
    }

    data
}
