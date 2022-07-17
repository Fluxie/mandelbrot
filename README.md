# Mandelbrot Visualizer

Creates a graysacle image visualization of the [Mandelbrot set](https://en.wikipedia.org/wiki/Mandelbrot_set).

## Usage

```
Arguments for the Mandelbrot set visualizer

USAGE:
    mandelbrot [OPTIONS]

OPTIONS:
    -f, --filename <FILENAME>    Target path to the image (PNG) [default: mandelbrot.png]
    -h, --height <HEIGHT>        Height of the image [default: 1024]
    --help                   Print help information
    -p, --parallel               Enable parallel execution
    -V, --version                Print version information
    -w, --width <WIDTH>          Width of the image [default: 1024]
```

## Installation

This assumes no previous knowledge of Rust. 

1. Install [Rust toolchain](https://rustup.rs/)
2. Clone this repository "git clone https://github.com/fluxie/mandelbrot.git"
3. Got to the root of the repository
4. Run tests: "cargo test"
5. Run the Mandlebrot visualizer "cargo run --release"

## TODO
* Increase the color depth to 16-bits.

## Remarks

This project was an exercise to Rust SIMD for the author.
The non-generalized implementation of the [SIMD algorithm](./src/simd/f64/mod.rs)
is preserved as a reference for the viewers. 