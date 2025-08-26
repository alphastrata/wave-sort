//! # Wave Sort
//!
//! A Rust implementation of Wave Sort (W-Sort), a novel in-place sorting algorithm
//! It is in no way compettivie with any high performance sorting algos like quicksort or mergesort etc.
//! Implemented here as the paper was interesting: https://www.alphaxiv.org/abs/2505.13552
//! A serial and parallel (slightly better than the serial but nothing special) implementation is provided.

use std::cmp::Ordering;

#[cfg(feature = "parallel")]
mod parallel;

/// Trait providing Wave Sort functionality for slices and vectors.
pub trait WaveSort<T> {
    fn wave_sort(&mut self)
    where
        T: Ord;

    fn wave_sort_by<F>(&mut self, compare: F)
    where
        F: FnMut(&T, &T) -> Ordering;

    fn wave_sort_by_key<K, F>(&mut self, f: F)
    where
        F: FnMut(&T) -> K,
        K: Ord;

    /// Parallel version of wave_sort
    #[cfg(feature = "parallel")]
    fn par_wave_sort(&mut self)
    where
        T: Ord + Send + Sync;

    /// Parallel version of wave_sort_by
    #[cfg(feature = "parallel")]
    fn par_wave_sort_by<F>(&mut self, compare: F)
    where
        F: FnMut(&T, &T) -> Ordering + Send + Sync + Clone;

    /// Parallel version of wave_sort_by_key
    #[cfg(feature = "parallel")]
    fn par_wave_sort_by_key<K, F>(&mut self, f: F)
    where
        F: FnMut(&T) -> K + Send + Sync + Clone,
        K: Ord;

    /// Parallel wave sort with custom threshold
    #[cfg(feature = "parallel")]
    fn par_wave_sort_with_threshold(&mut self, threshold: usize)
    where
        T: Ord + Send + Sync;
}

impl<T: Send + Sync> WaveSort<T> for [T] {
    fn wave_sort(&mut self)
    where
        T: Ord,
    {
        if self.len() < 2 {
            return;
        }
        self.wave_sort_by(T::cmp);
    }

    fn wave_sort_by<F>(&mut self, mut compare: F)
    where
        F: FnMut(&T, &T) -> Ordering,
    {
        if self.len() < 2 {
            return;
        }

        let mut sorter = WaveSorter {
            compare: &mut compare,
        };
        sorter.upwave(self, 0, self.len() - 1);
    }

    fn wave_sort_by_key<K, F>(&mut self, mut f: F)
    where
        F: FnMut(&T) -> K,
        K: Ord,
    {
        self.wave_sort_by(|a, b| f(a).cmp(&f(b)));
    }

    #[cfg(feature = "parallel")]
    fn par_wave_sort(&mut self)
    where
        T: Ord + Send + Sync,
    {
        if self.len() < 2 {
            return;
        }
        self.par_wave_sort_by(T::cmp);
    }

    #[cfg(feature = "parallel")]
    fn par_wave_sort_by<F>(&mut self, compare: F)
    where
        F: FnMut(&T, &T) -> Ordering + Send + Sync + Clone,
    {
        if self.len() < 2 {
            return;
        }

        let mut sorter = parallel::ParallelWaveSorter::new(compare);
        sorter.par_upwave(self, 0, self.len() - 1);
    }

    #[cfg(feature = "parallel")]
    fn par_wave_sort_by_key<K, F>(&mut self, mut f: F)
    where
        F: FnMut(&T) -> K + Send + Sync + Clone,
        K: Ord,
    {
        self.par_wave_sort_by(move |a, b| f(a).cmp(&f(b)));
    }

    #[cfg(feature = "parallel")]
    fn par_wave_sort_with_threshold(&mut self, threshold: usize)
    where
        T: Ord + Send + Sync,
    {
        if self.len() < 2 {
            return;
        }

        let mut sorter = parallel::ParallelWaveSorter::with_threshold(T::cmp, threshold);
        sorter.par_upwave(self, 0, self.len() - 1);
    }
}

impl<T: Send + Sync> WaveSort<T> for Vec<T> {
    fn wave_sort(&mut self)
    where
        T: Ord,
    {
        self.as_mut_slice().wave_sort();
    }

    fn wave_sort_by<F>(&mut self, compare: F)
    where
        F: FnMut(&T, &T) -> Ordering,
    {
        self.as_mut_slice().wave_sort_by(compare);
    }

    fn wave_sort_by_key<K, F>(&mut self, f: F)
    where
        F: FnMut(&T) -> K,
        K: Ord,
    {
        self.as_mut_slice().wave_sort_by_key(f);
    }

    #[cfg(feature = "parallel")]
    fn par_wave_sort(&mut self)
    where
        T: Ord + Send + Sync,
    {
        self.as_mut_slice().par_wave_sort();
    }

    #[cfg(feature = "parallel")]
    fn par_wave_sort_by<F>(&mut self, compare: F)
    where
        F: FnMut(&T, &T) -> Ordering + Send + Sync + Clone,
    {
        self.as_mut_slice().par_wave_sort_by(compare);
    }

    #[cfg(feature = "parallel")]
    fn par_wave_sort_by_key<K, F>(&mut self, f: F)
    where
        F: FnMut(&T) -> K + Send + Sync + Clone,
        K: Ord,
    {
        self.as_mut_slice().par_wave_sort_by_key(f);
    }

    #[cfg(feature = "parallel")]
    fn par_wave_sort_with_threshold(&mut self, threshold: usize)
    where
        T: Ord + Send + Sync,
    {
        self.as_mut_slice().par_wave_sort_with_threshold(threshold);
    }
}

struct WaveSorter<F> {
    compare: F,
}

impl<F> WaveSorter<F> {
    /// Up-wave operation following the paper's algorithm
    fn upwave<T>(&mut self, slice: &mut [T], start: usize, end: usize)
    where
        F: FnMut(&T, &T) -> Ordering,
    {
        if start >= end {
            return;
        }

        // Initialise sorted region at the right end (as per paper)
        let mut sorted_start = end;
        let mut sorted_length = 1;
        let length = end - start + 1;

        // Main upwave loop - exactly as in the paper
        loop {
            let left_bound = if sorted_length * 2 < length {
                end + 1 - (sorted_length * 2)
            } else {
                start
            };

            self.downwave(slice, left_bound, sorted_start, end);
            sorted_start = left_bound;
            sorted_length = end - sorted_start + 1;

            // Check termination condition
            if length < (sorted_length * 2) || left_bound <= start {
                break;
            }
        }

        // Final downwave to sort remaining portion if needed
        if start < sorted_start {
            self.downwave(slice, start, sorted_start, end);
        }
    }

    /// Down-wave operation following the paper's algorithm
    fn downwave<T>(&mut self, slice: &mut [T], start: usize, sorted_start: usize, end: usize)
    where
        F: FnMut(&T, &T) -> Ordering,
    {
        if sorted_start <= start {
            return;
        }

        // Select pivot as median of sorted portion
        let p = sorted_start + (end - sorted_start) / 2;

        // Partition unsorted portion using pivot
        let m = self.partition(slice, start, sorted_start, p);

        if m == sorted_start {
            if p == sorted_start {
                if start > 0 {
                    self.upwave(slice, start, sorted_start - 1);
                }
                return;
            }
            self.downwave(slice, start, sorted_start, p - 1);
            return;
        }

        // Perform block swap
        self.block_swap(slice, m, sorted_start, p);

        if m == start {
            if p == sorted_start {
                if m < end {
                    self.upwave(slice, m + 1, end);
                }
                return;
            }
            let new_start = m + (p - sorted_start) + 1;
            if new_start <= end {
                self.downwave(slice, new_start, p + 1, end);
            }
            return;
        }

        if p == sorted_start {
            if start < m {
                self.upwave(slice, start, m - 1);
            }
            if m < end {
                self.upwave(slice, m + 1, end);
            }
            return;
        }

        // General recursive case
        let boundary = m + (p - sorted_start);
        if start < m && boundary > 0 {
            self.downwave(slice, start, m, boundary - 1);
        }
        if boundary < end && p < end {
            self.downwave(slice, boundary + 1, p + 1, end);
        }
    }

    /// Partition function similar to quicksort's Lomuto partition
    fn partition<T>(&mut self, slice: &mut [T], start: usize, end: usize, pivot: usize) -> usize
    where
        F: FnMut(&T, &T) -> Ordering,
    {
        if start >= end {
            return start;
        }

        let mut i = start;
        let mut j = start;

        // Use pivot element for comparison
        while j < end {
            if (self.compare)(&slice[j], &slice[pivot]) == Ordering::Less {
                if i != j {
                    slice.swap(i, j);
                }
                i += 1;
            }
            j += 1;
        }

        i
    }

    /// Safe block swap implementation using rotations
    fn block_swap<T>(&mut self, slice: &mut [T], m: usize, r: usize, p: usize) {
        if m >= r || r > p || p >= slice.len() {
            return;
        }

        let left_len = r - m; // Length of left block
        let right_len = p - r + 1; // Length of right block

        if left_len == 0 || right_len == 0 {
            return;
        }

        // Use slice rotation for safe block swapping
        // This rotates the elements so that [left_block][right_block] becomes [right_block][left_block]
        slice[m..=p].rotate_right(right_len);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_and_single() {
        let mut empty: Vec<i32> = vec![];
        empty.wave_sort();
        assert_eq!(empty, vec![]);

        let mut single = vec![42];
        single.wave_sort();
        assert_eq!(single, vec![42]);
    }

    #[test]
    fn basic_sorting() {
        let mut data = vec![3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5];
        let mut expected = data.clone();
        expected.sort();

        data.wave_sort();
        assert_eq!(data, expected);
    }

    #[test]
    fn two_elements() {
        let mut data = vec![2, 1];
        data.wave_sort();
        assert_eq!(data, vec![1, 2]);

        let mut data = vec![1, 2];
        data.wave_sort();
        assert_eq!(data, vec![1, 2]);
    }

    #[test]
    fn reverse_sorted() {
        let mut data: Vec<i32> = (0..10).rev().collect();
        let mut expected = data.clone();
        expected.sort();

        data.wave_sort();
        assert_eq!(data, expected);
    }

    #[test]
    fn custom_comparison() {
        let mut data = vec![3, 1, 4, 1, 5];
        data.wave_sort_by(|a, b| b.cmp(a));
        assert_eq!(data, vec![5, 4, 3, 1, 1]);
    }

    #[test]
    fn already_sorted() {
        let mut data: Vec<i32> = (0..10).collect();
        let expected = data.clone();

        data.wave_sort();
        assert_eq!(data, expected);
    }

    #[test]
    fn duplicates() {
        let mut data = vec![5, 1, 3, 1, 5, 9, 5, 6, 5, 3, 1];
        let mut expected = data.clone();
        expected.sort();

        data.wave_sort();
        assert_eq!(data, expected);
    }
}
