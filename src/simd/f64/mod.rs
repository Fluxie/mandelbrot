use std::simd::{f64x4, u32x4, i64x4};

/// The contents in this file are preserved to showcase the Rust SIMD without
/// the complexities of generalizing the solution to different floating point types and lane counts.

pub struct Mandelbrot {

    /// Width of the final image in pixels.
    width: u32,

    /// Height of the final image in pixels.
    height: u32,

    /// The size of the increment of each pixel in Mandelbrot scale in X-axis.
    x_step: f64x4,

    /// The size of the increment of each pixel in Mandelbrot scale in Y-axis.
    y_step: f64x4,

    /// Holds a scaled vector with initial values for the X-axis when the algorithm starts a new row.
    x_scale_start: f64x4,
}

/// A pixel in Mandelbrot scale.
pub struct MandelbrotPixel {

    /// A vector defining the position of lanes in X-axis.
    x: u32x4,

    /// A vector defining the position of lanes in Y-axis.
    y: u32x4,

    /// A vector defining the position of lanes in X-axis in Mandelbrot coordinates.
    x_scaled: f64x4,

    /// A vector defining the position of lanes in Y-axis in Mandelbrot coordinates.
    y_scaled: f64x4,
}

pub struct MandelbrotIterator<'a> {

    /// Reference to the Mandelbrot parameters.
    mandelbrot: &'a Mandelbrot,

    /// Next pixel returned by the iterator.
    next_pixel: MandelbrotPixel,
}

impl Mandelbrot {

    /// Minimum starting value of the set in the X-axis that can satisfy the Mandelbrot condition.
    const X_MIN: f64 = -2.0;

    /// A vector of minimum starting values of the set in the X-axis that can satisfy the Mandelbrot condition.
    const X_MIN_SIMD: f64x4 = f64x4::splat( -2.0 );

    /// Minimum starting value of the set in the Y-axis that can satisfy the Mandelbrot condition.
    const Y_MIN: f64 = -1.12;

    /// Initializes new image with the given parameters.
    #[allow(dead_code)]
    pub fn new(
        width: u32,
        height: u32,
    ) -> Result<Mandelbrot, String> {

        // Ensure the size of the requested image is compatible with the specified lane count.
        if width as usize % 4 != 0 {
            return Err( format!( "Width {} must be a multiple of {}", width, 4 ) );
        }
        else if height as usize % 4 != 0 {
            return Err( format!( "Height {} must be a multiple of {}", height, 4 ) );
        }

        // Scale the pixels.
        let x_step = ( f64::abs( Mandelbrot::X_MIN ) + f64::abs( 0.47  ) ) / width as f64;
        let y_step = ( f64::abs( Mandelbrot::Y_MIN ) + f64::abs( 1.12  ) ) / height as f64;
        let x_scale_start = f64x4::from_array(
            [Mandelbrot::X_MIN,
                Mandelbrot::X_MIN + x_step * 1.0,
                Mandelbrot::X_MIN + x_step * 2.0,
                Mandelbrot::X_MIN + x_step * 3.0] );

        let x_step = f64x4::splat( x_step );
        let y_step = f64x4::splat( y_step );

        // Start.
        Ok( Mandelbrot {
            width, height,
            x_step, y_step,
            x_scale_start
        } )
    }

    /// Returns an iterator to the pixels in the Mandelbrot.
    #[allow(dead_code)]
    pub fn iter(
        &self
    ) -> MandelbrotIterator {

        MandelbrotIterator {
            mandelbrot: self,
            next_pixel: MandelbrotPixel {
                x: MandelbrotIterator::X_START, x_scaled: self.x_scale_start.clone(),
                y: u32x4::splat( 0 ), y_scaled: f64x4::splat( Mandelbrot::Y_MIN )
            },
        }
    }
}

impl<'a> MandelbrotIterator<'a> {
    const X_START: u32x4 = u32x4::from_array( [ 0, 1, 2, 3 ]  );
    const X_PIXER_STEP: u32x4 = u32x4::splat( u32x4::LANES as u32 );
    const Y_PIXER_STEP: u32x4 = u32x4::splat( 1 );
}

impl<'a> Iterator for MandelbrotIterator<'a> {
    type Item = MandelbrotPixel;


    /// Gets a vector of the next pixels that fills the image grid.
    fn next(&mut self) -> Option<Self::Item> {

        // Last pixel returned?
        if self.next_pixel.y == u32x4::splat( self.mandelbrot.height ) {
            return None;
        };

        // Which axis to increment?
        let mut pixel;
        if self.next_pixel.x[ 3 ] == self.mandelbrot.width - 1 {

            // Next row, increase Y-axis.
            pixel = MandelbrotPixel {
                x: MandelbrotIterator::X_START,
                x_scaled: self.mandelbrot.x_scale_start.clone(),
                y: &self.next_pixel.y + MandelbrotIterator::Y_PIXER_STEP,
                y_scaled: f64x4::splat( Mandelbrot::Y_MIN ) + to_float( &self.next_pixel.y + MandelbrotIterator::Y_PIXER_STEP ) * &self.mandelbrot.y_step,
            }
        }
        else {

            // Continue on the same row. Increase X-axis.
            pixel = MandelbrotPixel {
                x: &self.next_pixel.x + MandelbrotIterator::X_PIXER_STEP,
                x_scaled: &Mandelbrot::X_MIN_SIMD + to_float( &self.next_pixel.x + MandelbrotIterator::X_PIXER_STEP )  * &self.mandelbrot.x_step,
                y: self.next_pixel.y.clone(),
                y_scaled: self.next_pixel.y_scaled.clone()
            }
        }

        // Grab the pixel we will return.
        // By storing the value of the next pixel instead of the previous pixel it was possible
        // to initialize the generator without Option.
        std::mem::swap( &mut pixel, &mut self.next_pixel );

        // Return the next pixel.
        return Some( pixel );
    }
}

/// Calculates the colors for the pixel represented by the input.
#[allow(dead_code)]
pub fn calculate_color(
    pixel: MandelbrotPixel
) -> [u8; f64x4::LANES] {
    let mut x = f64x4::splat( 0.0 );
    let mut y = f64x4::splat( 0.0 );
    let mandelbrot_condition = f64x4::splat( 2.0 * 2.0 );
    const MAX_ITERATIONS: i64 = 255;
    const ITERATION_INCREMENT: i64x4 = i64x4::splat( 1 );
    const TWO: f64x4 = f64x4::splat( 2.0 );
    let mut iteration = i64x4::splat( 0 );
    loop  {

        // Iterations exhausted?
        // Only some of the lanes may reach this point if the associated pixel fails to
        // satisfy the Mandelbrot condition.
        if iteration.reduce_max() >= MAX_ITERATIONS {
            break;
        }

        // Increment only those lanes that satisfy the Mandelbrot condition.
        let condition_mask = ( x*x + y*y ).lanes_lt( mandelbrot_condition );
        if condition_mask.any() == false {
            break;
        }
        let previous_iteration = iteration.clone();
        iteration = condition_mask.select::<i64>( iteration + &ITERATION_INCREMENT, previous_iteration );

        // Next value in the series for each pixel.
        let temp: f64x4 = x*x - y*y + &pixel.x_scaled;
        y = &TWO*x*y + &pixel.y_scaled;
        x = temp;
    }

    return iteration.to_array().map( |value| value as u8 );
}

/// Converts integer SIMD type into floating-point SIMD type.
fn to_float(
    value: u32x4
) -> f64x4 {
    let integers = value.to_array();
    f64x4::from_slice( &integers.map( |i| i as f64 ) )
}