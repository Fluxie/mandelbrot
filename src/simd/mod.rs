use std::simd::{f64x4, Mask, u32x4, i64x4};

pub struct Mandlebrot {

    width: u32,
    height: u32,
    x_step: f64x4,
    y_step: f64x4,
    x_scale_start: f64x4,
}

/// A pixel in Mandlebrot scale.
pub struct MandlebrotPixel {

    pub x: u32x4,
    pub y: u32x4,
    pub x_scaled: f64x4,
    pub y_scaled: f64x4,
}

pub struct MandlebrotIterator<'a> {

    mandlebrot: &'a Mandlebrot,
    next_pixel: MandlebrotPixel,
}

impl Mandlebrot {

    const X_MIN: f64 = -2.0;
    const X_MIN_SIMD: f64x4 = f64x4::splat( -2.0 );
    const Y_MIN: f64 = -1.12;

    pub fn new(
        width: u32,
        height: u32,
    ) -> Mandlebrot {

        // Scale the pixels.
        let x_step = ( f64::abs( Mandlebrot::X_MIN ) + f64::abs( 0.47  ) ) / width as f64;
        let y_step = ( f64::abs( Mandlebrot::Y_MIN ) + f64::abs( 1.12  ) ) / height as f64;
        let x_scale_start = f64x4::from_array(
            [Mandlebrot::X_MIN,
                Mandlebrot::X_MIN + x_step * 1.0,
                Mandlebrot::X_MIN + x_step * 2.0,
                Mandlebrot::X_MIN + x_step * 3.0] );

        let x_step = f64x4::splat( x_step );
        let y_step = f64x4::splat( y_step );

        // Start.
        Mandlebrot {
            width, height,
            x_step, y_step,
            x_scale_start
        }
    }

    pub fn iter(
        &self
    ) -> MandlebrotIterator {


        MandlebrotIterator {
            mandlebrot: self,
            next_pixel: MandlebrotPixel {
                x: MandlebrotIterator::X_START, x_scaled: self.x_scale_start.clone(),
                y: u32x4::splat( 0 ), y_scaled: f64x4::splat( Mandlebrot::Y_MIN )
            },
        }
    }
}

impl<'a> MandlebrotIterator<'a> {
    const X_START: u32x4 = u32x4::from_array( [ 0, 1, 2, 3 ]  );
    const X_PIXER_STEP: u32x4 = u32x4::splat( u32x4::LANES as u32 );
    const Y_PIXER_STEP: u32x4 = u32x4::splat( 1 );
}

impl<'a> Iterator for MandlebrotIterator<'a> {
    type Item = MandlebrotPixel;


    /// Gets next pixel that fills the grid.
    fn next(&mut self) -> Option<Self::Item> {

        // Last pixel returned?
        if self.next_pixel.y == u32x4::splat( self.mandlebrot.height ) {
            return None;
        };

        // Increment x axis first.
        let mut pixel;
        if self.next_pixel.x[ 3 ] == self.mandlebrot.width - 1 {
            pixel = MandlebrotPixel {
                x: MandlebrotIterator::X_START,
                x_scaled: self.mandlebrot.x_scale_start.clone(),
                y: &self.next_pixel.y + MandlebrotIterator::Y_PIXER_STEP,
                y_scaled: f64x4::splat( Mandlebrot::Y_MIN ) + to_float( &self.next_pixel.y + MandlebrotIterator::Y_PIXER_STEP ) * &self.mandlebrot.y_step,
            }
        }
        else {
            pixel = MandlebrotPixel {
                x: &self.next_pixel.x + MandlebrotIterator::X_PIXER_STEP,
                x_scaled: &Mandlebrot::X_MIN_SIMD + to_float( &self.next_pixel.x + MandlebrotIterator::X_PIXER_STEP )  * &self.mandlebrot.x_step,
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

pub fn calculate_color(
    pixel: MandlebrotPixel
) -> [u8; f64x4::LANES] {
    let mut x = f64x4::splat( 0.0 );
    let mut y = f64x4::splat( 0.0 );
    let mandlebrot_condition = f64x4::splat( 2.0 * 2.0 );
    const MAX_ITERATIONS: i64 = 255;
    const ITERATION_INCREMENT: i64x4 = i64x4::splat( 1 );
    const TWO: f64x4 = f64x4::splat( 2.0 );
    let mut iteration = i64x4::splat( 0 );
    loop  {

        // Iterations exhausted?
        // Only some of the lanes may reach this point if the associated pixel fails to
        // satisfy the Mandlebrot condition.
        if iteration.reduce_max() >= MAX_ITERATIONS {
            break;
        }

        // Increment only those lanes that satisfy the Mandlebrot condition.
        let condition_mask = ( x*x + y*y ).lanes_lt( mandlebrot_condition );
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