use wave_sort::WaveSort;

// Simple demo
fn main() {
    let mut data = vec![64, 34, 25, 12, 22, 11, 90];
    println!("Before: {:?}", data);
    data.wave_sort();
    println!("After:  {:?}", data);
}
