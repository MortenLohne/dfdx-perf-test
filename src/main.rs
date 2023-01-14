use std::time;

use dfdx::{
    prelude::{Linear, Module, ModuleBuilder, ReLU, Tanh},
    shapes::Rank1,
    tensor::{AsArray, Cpu, Tensor, ZerosTensor},
};

const RUNS: usize = 1_000_000;

pub type Model<const N: usize> = (
    (Linear<N, N>, ReLU),
    (Linear<N, N>, ReLU),
    (Linear<N, 1>, Tanh),
);

fn main() {
    bench::<1>();
    bench::<2>();
    bench::<4>();
    bench::<8>();
    bench::<16>();
    bench::<32>();
    bench::<64>();
}

fn bench<const N: usize>() {
    let cpu: Cpu = Default::default();

    let model: Model<N> = cpu.build_module();

    let start_time = time::Instant::now();
    let mut result = 0.0;

    for _ in 0..RUNS {
        let input_tensor: Tensor<Rank1<N>> = cpu.zeros();
        let output = model.forward(input_tensor);
        result += output.array()[0];
    }

    let elapsed = start_time.elapsed().as_secs_f64();
    println!(
        "Network width {}: {:.1} runs/s ({:.1})",
        N,
        (RUNS as f64) / elapsed,
        result
    );
}
