use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;
use std::hint::black_box;
use wave_sort::WaveSort;

const SEED: u64 = 1234567890;

fn generate_data(size: usize) -> Vec<i64> {
    let mut rng = Pcg64::seed_from_u64(SEED);
    let mut data = Vec::with_capacity(size);
    for _ in 0..size {
        data.push(rng.random());
    }
    data
}

fn benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("wave_sort_vs_std_sort");

    [8, 16].iter().map(|p| 1 << p).for_each(|size| {
        let data = generate_data(size);

        group.bench_with_input(BenchmarkId::new("std_sort", size), &data, |b, d| {
            b.iter(|| {
                let mut vec = d.clone();

                // A variant of quicksort from the std lib -- the stable one is not quicksort based.
                /*
                   The paper compares against quicksort -- so we bench against this.
                */
                black_box(vec.sort_unstable())
            })
        });

        group.bench_with_input(
            BenchmarkId::new("wave_sort_sequential", size),
            &data,
            |b, d| {
                b.iter(|| {
                    let mut vec = d.clone();
                    black_box(vec.wave_sort())
                })
            },
        );

        #[cfg(feature = "parallel")]
        group.bench_with_input(
            BenchmarkId::new("wave_sort_parallel", size),
            &data,
            |b, d| {
                b.iter(|| {
                    let mut vec = d.clone();
                    black_box(vec.par_wave_sort())
                })
            },
        );
    });

    group.finish();
}

#[cfg(feature = "parallel")]
fn parallel_performance_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_performance");

    // Test on larger datasets where parallelization should shine
    [28, 32].iter().map(|p| 1 << p).for_each(|size| {
        // I believe we'll NOT see appreciable gains till 2^64...
        let data = generate_data(size);

        group.bench_with_input(BenchmarkId::new("std_sort", size), &data, |b, d| {
            b.iter(|| {
                let mut vec = d.clone();
                black_box(vec.sort_unstable())
            })
        });

        group.bench_with_input(
            BenchmarkId::new("wave_sort_parallel", size),
            &data,
            |b, d| {
                b.iter(|| {
                    let mut vec = d.clone();
                    black_box(vec.par_wave_sort())
                })
            },
        );
    });

    group.finish();
}

#[cfg(feature = "parallel")]
criterion_group!(benches, benchmark, parallel_performance_benchmark);

#[cfg(not(feature = "parallel"))]
criterion_group!(benches, benchmark);

criterion_main!(benches);
