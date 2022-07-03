#![feature(portable_simd)]
#![feature(test)]

extern crate test;
use std::path::Path;
use std::fs::File;
use std::io::BufWriter;
use std::simd::{f64x4, Mask, u32x4, i64x4};
use png;
use rayon::prelude::*;

mod scalar;
mod simd;


fn main() {

    // let path = std::env::args()
    //     .nth(1)
    //     .expect("Expected a filename to output to.");
    let file = File::create( "/home/fluxie/mandlebrot.png").unwrap();
    let ref mut w = BufWriter::new( file );

    // Prepare the image.
    let width = 10240;
    let height = 10240;
    let mut encoder = png::Encoder::new(w, width, height);
    encoder.set_color( png::ColorType::Grayscale );
    encoder.set_depth( png::BitDepth::Eight );
    let mut writer = encoder.write_header().expect( "Preparing image failed." );

    // Calculate color for each pixel.
    let simd_mandlebrot = simd::Mandlebrot::new(width, height );
    let mut pixel_index = 0;
    let mut simd_image_data= simd_mandlebrot.iter()

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
        let width = 16;
        let height = 16;
        let simd_mandlebrot = crate::simd::Mandlebrot::new(width, height );
        let scalar_mandlebrot = crate::scalar::Mandlebrot::new(width, height );
        let mut scalar_iterator = scalar_mandlebrot.iter();
        for simd_pixel in simd_mandlebrot.iter() {

            let mut scalar_colors: [u8; 4] = [0, 0, 0, 0];
            for i in 0..4 {

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

    #[bench]
    fn simd_benchmark(
        b: &mut Bencher
    ) {
        b.iter( || {
            // Benchmark the SIMD mandlebrot
            let width = 256;
            let height = 256;
            let mut simd_mandlebrot = simd::Mandlebrot::new(width, height);
            let simd_image_data = simd_mandlebrot.iter()
                .flat_map(|mpixel| simd::calculate_color(mpixel))
                .collect::<Vec<_>>();

            // Require to value to prevent compiler optimizations.
            test::black_box(simd_image_data);
        } );
    }

    #[bench]
    fn scalar_benchmark(
        b: &mut Bencher
    ) {
        b.iter( || {

            // Benchmark the SIMD mandlebrot
            let width = 256;
            let height = 256;
            let mut scalar_mandlebrot = scalar::Mandlebrot::new(width, height);
            test::black_box(scalar_mandlebrot.iter()
                .map(|mpixel| scalar::calculate_color(mpixel))
                .collect::<Vec<u8>>());
        } );
    }
}
