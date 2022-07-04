#![feature(portable_simd)]
#![feature(test)]

extern crate test;
use std::fs::File;
use std::io::BufWriter;
use clap::Parser;
use png;
use rayon::prelude::*;

mod scalar;
mod simd;

/// Arguments for the Mandelbrot set visualizer.
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {

    /// Width of the image.
    #[clap(short, long, value_parser, default_value_t = 1024)]
    width: u32,

    /// Height of the image.
    #[clap(short, long, value_parser, default_value_t = 1024)]
    height: u32,

    /// Target path to the image (PNG).
    #[clap(short, long, value_parser, default_value_t = String::from( "mandelbrot.png" ))]
    filename: String,

    /// Enable parallel execution.
    #[clap(short, long, value_parser)]
    parallel: bool,
}

fn main() {

    // Parse the arguments.
    let args = Args::parse();

    // Prepare output file.
    let file = File::create( args.filename ).expect( "Creating the target file failed." );
    let ref mut w = BufWriter::new( file );

    // Prepare the image.
    let width = args.width;
    let height = args.height;
    let mut encoder = png::Encoder::new(w, width, height);
    encoder.set_color( png::ColorType::Grayscale );
    encoder.set_depth( png::BitDepth::Eight );
    let mut writer = encoder.write_header().expect( "Preparing image failed." );

    // Calculate color for each pixel.
    let simd_mandelbrot = simd::Mandelbrot::<f64, 4>::new(width, height ).unwrap();
    let mut pixel_index = 0;
    let simd_image_data= simd_mandelbrot.iter()

        // Specify index for each individual pixel.
        .map( |pixel| {
            pixel_index += 1;
            ( pixel_index, pixel )
        } );

    // Calculate the results in parallel or sequentially.
    let mut simd_image_data = if args.parallel {

        // Enable parallelism
        simd_image_data.par_bridge()

            // Calculate color for each pixel.
            .map(|pixel| (pixel.0, simd::calculate_color( pixel.1 ) ) )
            .collect::<Vec<_>>()
    }
     else {
         // Calculate color for each pixel.
         simd_image_data.map(|pixel| (pixel.0, simd::calculate_color( pixel.1 ) ) )
             .collect::<Vec<_>>()
     };

    // The parallel calculation above trashes the original order.
    // Sort the result to generate the final output.
    simd_image_data.par_sort_by( |a, b| a.0.partial_cmp( &b.0 ).unwrap() );
    let simd_image_data = simd_image_data.into_iter()
        .flat_map( |mpixel| mpixel.1 )
        .collect::<Vec<_>>();

    // Finalize the image.
    writer.write_image_data(&simd_image_data).expect( "Invalid data");
    writer.finish().expect( "Finalized" );
}


#[cfg(test)]
mod tests {

    use super::*;
    use test::Bencher;

    /// Ensure both SIMD and scalar versions produce identical output.
    #[test]
    fn pixels_are_identical()
    {
        // Test with different image sizes.
        let test_data = [ ( 16, 16 ), ( 32, 16 ), ( 16, 32 )];
        for image_size in test_data {

            let width = image_size.0;
            let height = image_size.1;
            const LANES: usize = 4;
            let simd_mandelbrot =
                    crate::simd::Mandelbrot::<f64, LANES>::new(width, height ).unwrap();
            let scalar_mandelbrot = crate::scalar::Mandelbrot::new(width, height );
            let mut scalar_iterator = scalar_mandelbrot.iter();
            for simd_pixel in simd_mandelbrot.iter() {

                let mut scalar_colors: [u8; LANES] = [0, 0, 0, 0];
                for i in 0..LANES {

                    // Ensure the properties of the pixels are identical.
                    let scalar_pixel = scalar_iterator.next().unwrap();
                    assert_eq!( simd_pixel.x[ i ], scalar_pixel.x );
                    assert_eq!( simd_pixel.x_scaled[ i ], scalar_pixel.x_scaled );
                    assert_eq!( simd_pixel.y[ i ], scalar_pixel.y );
                    assert_eq!( simd_pixel.y_scaled[ i ], scalar_pixel.y_scaled );

                    scalar_colors[ i ] = crate::scalar::calculate_color( scalar_pixel );
                }
                let simd_colors = crate::simd::calculate_color( simd_pixel );
                assert_eq!( simd_colors, scalar_colors);
            }
        }
    }

    #[bench]
    fn simd_f32x2_sequential_benchmark(
        b: &mut Bencher
    ) {
        b.iter( || {
            // Benchmark the SIMD mandelbrot
            let width = 256;
            let height = 256;
            let simd_mandelbrot =
                    simd::Mandelbrot::<f32, 2>::new(width, height).unwrap();
            let simd_image_data = simd_mandelbrot.iter()
                .flat_map(|mpixel| simd::calculate_color(mpixel))
                .collect::<Vec<_>>();

            // Require to value to prevent compiler optimizations.
            test::black_box(simd_image_data);
        } );
    }

    #[bench]
    fn simd_f32x4_sequential_benchmark(
        b: &mut Bencher
    ) {
        b.iter( || {
            // Benchmark the SIMD mandelbrot
            let width = 256;
            let height = 256;
            let simd_mandelbrot =
                    simd::Mandelbrot::<f32, 4>::new(width, height).unwrap();
            let simd_image_data = simd_mandelbrot.iter()
                .flat_map(|mpixel| simd::calculate_color(mpixel))
                .collect::<Vec<_>>();

            // Require to value to prevent compiler optimizations.
            test::black_box(simd_image_data);
        } );
    }

    #[bench]
    fn simd_f32x4_parallel_benchmark(
        b: &mut Bencher
    ) {
        b.iter( || {
            // Benchmark the SIMD mandelbrot
            let width = 256;
            let height = 256;
            let simd_mandelbrot =
                    simd::Mandelbrot::<f32, 4>::new(width, height).unwrap();
            let mut pixel_index = 0;
            let mut simd_image_data= simd_mandelbrot.iter()

                    // Specify index for each individual pixel.
                    .map( |pixel| {
                        pixel_index += 1;
                        ( pixel_index, pixel )
                    } )

                    // Enable parallelism
                    .par_bridge()

                    // Calculate color for each pixel.
                    .map(|pixel| (pixel.0, simd::calculate_color( pixel.1 ) ) )
                    .collect::<Vec<_>>();

            // The parallel calculation above trashes the original order.
            // Sort the result to generate the final output.
            simd_image_data.par_sort_by( |a, b| a.0.partial_cmp( &b.0 ).unwrap() );
            let simd_image_data = simd_image_data.into_iter()
                    .flat_map( |mpixel| mpixel.1 )
                    .collect::<Vec<_>>();

            // Require to value to prevent compiler optimizations.
            test::black_box(simd_image_data);
        } );
    }

    #[bench]
    fn simd_f32x8_sequential_benchmark(
        b: &mut Bencher
    ) {
        b.iter( || {
            // Benchmark the SIMD mandelbrot
            let width = 256;
            let height = 256;
            let simd_mandelbrot =
                    simd::Mandelbrot::<f32, 4>::new(width, height).unwrap();
            let simd_image_data = simd_mandelbrot.iter()
                .flat_map(|mpixel| simd::calculate_color(mpixel))
                .collect::<Vec<_>>();

            // Require to value to prevent compiler optimizations.
            test::black_box(simd_image_data);
        } );
    }

    #[bench]
    fn simd_f64x2_sequential_benchmark(
        b: &mut Bencher
    ) {
        b.iter( || {
            // Benchmark the SIMD mandelbrot
            let width = 256;
            let height = 256;
            let simd_mandelbrot =
                    simd::Mandelbrot::<f64, 2>::new(width, height).unwrap();
            let simd_image_data = simd_mandelbrot.iter()
                .flat_map(|mpixel| simd::calculate_color(mpixel))
                .collect::<Vec<_>>();

            // Require to value to prevent compiler optimizations.
            test::black_box(simd_image_data);
        } );
    }

    #[bench]
    fn simd_f64x4_sequential_benchmark(
        b: &mut Bencher
    ) {
        b.iter( || {
            // Benchmark the SIMD mandelbrot
            let width = 256;
            let height = 256;
            let simd_mandelbrot = simd::Mandelbrot::<f64, 4>::new(width, height).unwrap();
            let simd_image_data = simd_mandelbrot.iter()
                .flat_map(|mpixel| simd::calculate_color(mpixel))
                .collect::<Vec<_>>();

            // Require to value to prevent compiler optimizations.
            test::black_box(simd_image_data);
        } );
    }

    #[bench]
    fn scalar_f64_sequential_benchmark(
        b: &mut Bencher
    ) {
        b.iter( || {

            // Benchmark the SIMD mandelbrot
            let width = 256;
            let height = 256;
            let scalar_mandelbrot = scalar::Mandelbrot::new(width, height);
            test::black_box(scalar_mandelbrot.iter()
                .map(|mpixel| scalar::calculate_color(mpixel))
                .collect::<Vec<u8>>());
        } );
    }
}
