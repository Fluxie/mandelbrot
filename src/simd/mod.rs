use std::ops::{Add, Mul, Sub};
use std::simd::{Mask, MaskElement, Simd, SimdElement, LaneCount, SupportedLaneCount};

mod f64;

/// Defines the parameters for the Mandelbrot image.
pub struct Mandelbrot<T, const LANES: usize>
    where
        T: MandelbrotGrid<LANES, GridType = T>,
        T: SimdElement + PartialOrd,
        LaneCount<LANES>: SupportedLaneCount,

    // Require Add, Mul and Sub SIMD operator support
        Simd<T, LANES>: Mul< Output  = Simd<T, LANES>> + Add< Output  = Simd<T, LANES>> + Sub< Output  = Simd<T, LANES>>
{
    /// Width of the final image in pixels.
    width: u32,

    /// Height of the final image in pixels.
    height: u32,

    /// The size of the increment of each pixel in Mandelbrot scale in X-axis.
    x_step: Simd<T, LANES>,

    /// The size of the increment of each pixel in Mandelbrot scale in Y-axis.
    y_step: Simd<T, LANES>,

    /// Holds a scaled vector with initial values for the X-axis when the algorithm starts a new row.
    x_scale_start: Simd<T, LANES>,
}

/// A pixel in Mandelbrot scale.
pub struct MandelbrotPixel<T, const LANES: usize>
    where
        T: MandelbrotGrid<LANES, GridType = T, IteratorType = <T as SimdElement>::Mask>,
        T: SimdElement + PartialOrd,
        LaneCount<LANES>: SupportedLaneCount,

    // Require Add, Mul and Sub SIMD operator support
        Simd<T, LANES>: Mul< Output  = Simd<T, LANES>> + Add< Output  = Simd<T, LANES>> + Sub< Output  = Simd<T, LANES>>
{

    /// A vector defining the position of lanes in X-axis.
    pub x: Simd<u32, LANES>,

    /// A vector defining the position of lanes in Y-axis.
    pub y: Simd<u32, LANES>,

    /// A vector defining the position of lanes in X-axis in Mandelbrot coordinates.
    pub x_scaled: Simd<T, LANES>,

    /// A vector defining the position of lanes in Y-axis in Mandelbrot coordinates.
    pub y_scaled: Simd<T, LANES>,
}

/// An iterator that returns all the pixel for the Mandelbrot image.
pub struct MandelbrotIterator<'a, T, const LANES: usize>
    where
        T: MandelbrotGrid<LANES, GridType = T, IteratorType = <T as SimdElement>::Mask>,
        T: SimdElement + PartialOrd,
        LaneCount<LANES>: SupportedLaneCount,
        Simd<T, LANES>: Mul< Output  = Simd<T, LANES>> + Add< Output  = Simd<T, LANES>> + Sub< Output  = Simd<T, LANES>>,
        Simd<<T as SimdElement>::Mask, LANES>: Add<Output=Simd<<T as SimdElement>::Mask, LANES>>
{
    /// Reference to the Mandelbrot parameters.
    mandelbrot: &'a Mandelbrot<T, LANES>,

    /// Next pixel returned by the iterator.
    next_pixel: MandelbrotPixel<T, LANES>,
}

/// Defines the properties of the Mandelbrot grid for a number type used in the calculations.
pub trait MandelbrotGrid<const LANES: usize>
where
    LaneCount<LANES>: SupportedLaneCount
{
    /// The type of the type used in the calculations.
    type GridType: SimdElement;

    /// The type of the unit used as the iteration counter .
    /// The size of this unit is based on the GridType.
    type IteratorType: MaskElement + SimdElement;

    /// Minimum starting value of the set in the X-axis that can satisfy the Mandelbrot condition.
    const X_MIN: Self::GridType;

    /// Maximum starting value of the set in the X-axis that can satisfy the Mandelbrot condition.
    const X_MAX: Self::GridType;

    /// Minimum starting value of the set in the Y-axis that can satisfy the Mandelbrot condition.
    const Y_MIN: Self::GridType;

    /// Maximum starting value of the set in the Y-axis that can satisfy the Mandelbrot condition.
    const Y_MAX: Self::GridType;

    /// A vector of minimum starting values of the set in the X-axis that can satisfy the Mandelbrot condition.
    const X_MIN_SIMD: Simd<Self::GridType, LANES>;

    /// A vector of zeros. Helper for calculations.
    const ZERO: Simd<Self::GridType, LANES>;

    /// A vector of twos. Helper for calculations.
    const TWO: Simd<Self::GridType, LANES>;

    /// Vector of lanes each defining the value of the Mandelbrot condition.
    const MANDELBROT_CONDITION: Simd<Self::GridType, LANES>;

    /// The maximum number of iterations to to calculate the series.
    const MAX_ITERATIONS: Self::IteratorType;

    /// A vector of zeros for starting the iteration.
    const ZERO_ITERATIONS: Simd<Self::IteratorType, LANES>;

    /// A vector of lanes for incrementing the iteration countes.
    const ITERATION_INCREMENT: Simd<Self::IteratorType, LANES>;

    /// Calculates the size of a single step in X-axis in the Mandelbrot grid.
    fn x_step(
        pixel_width: u32
    ) -> Self::GridType;

    /// Calculates the size of a single step in Y-axis in the Mandelbrot grid.
    fn y_step(
        pixel_height: u32
    ) -> Self::GridType;

    /// Converts the integer pixel grid value into a representation in the actual grid.
    fn to_grid(
        value: &Simd<u32, LANES>
    ) -> Simd<Self::GridType, LANES>;

    /// Converts the reached iteration value into a grayscale color value.
    fn iteration_to_color(
        iteration: <Self::GridType as SimdElement>::Mask
    ) -> u8;

    /// Returns the maximum value of a lane in the given iteration.
    fn reduce_max_iteration(
        iteration: &Simd<Self::IteratorType, LANES>
    ) -> Self::IteratorType;

    /// The lanes product X and Y axis pixels to the lanes of the Mandelbrot condition.
    fn lanes_lt(
        product: Simd<Self::GridType, LANES>,
        condition: Simd<Self::GridType, LANES>,
    ) -> Mask<<Self::GridType as SimdElement>::Mask, LANES>;

    /// Selects the values for the next iteration
    fn select_iteration(
        mask: &Mask<<Self::GridType as SimdElement>::Mask, LANES>,
        iteration_proposal: Simd<Self::IteratorType, LANES>,
        previous_iteration: Simd<Self::IteratorType, LANES>,
    ) -> Simd<Self::IteratorType, LANES>;
}

impl<T, const LANES: usize> Mandelbrot<T, LANES>
    where
        T: MandelbrotGrid<LANES, GridType = T, IteratorType = <T as SimdElement>::Mask>,
        T: SimdElement + PartialOrd,
            LaneCount<LANES>: SupportedLaneCount,

        // Require Add, Mul and Sub SIMD operator support
        Simd<T, LANES>: Mul< Output  = Simd<T, LANES>> + Add< Output  = Simd<T, LANES>> + Sub< Output  = Simd<T, LANES>>,
        Simd<<T as SimdElement>::Mask, LANES>: Add<Output=Simd<<T as SimdElement>::Mask, LANES>>
{
    const X_START: Simd<u32, LANES> = Simd::from_array( get_x_start() );

    /// Initializes new image with the given parameters.
    pub fn new(
        width: u32,
        height: u32,
    ) -> Result<Mandelbrot<T, LANES>, String> {

        // Ensure the size of the requested image is compatible with the specified lane count.
        if width as usize % LANES != 0 {
            return Err( format!( "Width {} must be a multiple of {}", width, LANES ) );
        }
        else if height as usize % LANES != 0 {
            return Err( format!( "Height {} must be a multiple of {}", height, LANES ) );
        }

        // Scale the pixels.
        let x_step = <T as MandelbrotGrid<LANES>>::x_step( width );
        let y_step = <T as MandelbrotGrid<LANES>>::y_step( height );

        let x_step = Simd::splat( x_step );
        let y_step = Simd::splat( y_step );
        let x_scale_start= <T as MandelbrotGrid<LANES>>::X_MIN_SIMD + <T as MandelbrotGrid<LANES>>::to_grid( &Mandelbrot::X_START ) * x_step;

        // Start.
       Ok(  Mandelbrot {
            width, height,
            x_step, y_step,
            x_scale_start
        } )
    }

    /// Returns an iterator to the pixels in the Mandelbrot.
    pub fn iter(
        &self
    ) -> MandelbrotIterator<T, LANES> {

        MandelbrotIterator {
            mandelbrot: self,
            next_pixel: MandelbrotPixel {
                x: Mandelbrot::X_START, x_scaled: self.x_scale_start.clone(),
                y: Simd::splat( 0 ), y_scaled: Simd::<T, LANES>::splat( <T as MandelbrotGrid<LANES>>::Y_MIN )
            },
        }
    }
}

impl<'a, T, const LANES: usize> MandelbrotIterator<'a, T, LANES>
    where
        T: MandelbrotGrid<LANES, GridType = T, IteratorType = <T as SimdElement>::Mask>,
        T: SimdElement + PartialOrd,
        LaneCount<LANES>: SupportedLaneCount,

        // Require Add, Mul and Sub SIMD operator support
        Simd<T, LANES>: Mul< Output  = Simd<T, LANES>> + Add< Output  = Simd<T, LANES>> + Sub< Output  = Simd<T, LANES>>,
        Simd<<T as SimdElement>::Mask, LANES>: Add<Output=Simd<<T as SimdElement>::Mask, LANES>>
{
    const X_PIXER_STEP: Simd<u32, LANES>  = Simd::splat( LANES as u32 );
    const Y_PIXER_STEP: Simd<u32, LANES>  = Simd::splat( 1 );
}

impl<'a, T, const LANES: usize> Iterator for MandelbrotIterator<'a, T, LANES>
    where
        T: MandelbrotGrid<LANES, GridType = T, IteratorType = <T as SimdElement>::Mask>,
        T: SimdElement + PartialOrd,
        LaneCount<LANES>: SupportedLaneCount,

        // Require Add, Mul and Sub SIMD operator support
        Simd<T, LANES>: Mul< Output  = Simd<T, LANES>> + Add< Output  = Simd<T, LANES>> + Sub< Output  = Simd<T, LANES>>,
        Simd<<T as SimdElement>::Mask, LANES>: Add<Output=Simd<<T as SimdElement>::Mask, LANES>>
{
    type Item = MandelbrotPixel<T, LANES>;


    /// Gets a vector of the next pixels that fills the image grid.
    fn next(&mut self) -> Option<Self::Item> {

        // Last pixel returned?
        if self.next_pixel.y == Simd::splat( self.mandelbrot.height ) {
            return None;
        };

        // Which axis to increment?
        let mut pixel;
        if self.next_pixel.x[ LANES - 1 ] == self.mandelbrot.width - 1 {

            // Next row, increase Y-axis.
            pixel = MandelbrotPixel {
                x: Mandelbrot::X_START,
                x_scaled: self.mandelbrot.x_scale_start.clone(),
                y: &self.next_pixel.y + MandelbrotIterator::Y_PIXER_STEP,
                y_scaled: Simd::splat( <T as MandelbrotGrid<LANES>>::Y_MIN ) + <T as MandelbrotGrid<LANES>>::to_grid( &(&self.next_pixel.y + MandelbrotIterator::Y_PIXER_STEP )) * self.mandelbrot.y_step.clone(),
            }
        }
        else {

            // Continue on the same row. Increase X-axis.
            pixel = MandelbrotPixel {
                x: &self.next_pixel.x + MandelbrotIterator::X_PIXER_STEP,
                x_scaled: <T as MandelbrotGrid<LANES>>::X_MIN_SIMD + <T as MandelbrotGrid<LANES>>::to_grid( &(&self.next_pixel.x + MandelbrotIterator::X_PIXER_STEP ) ) * self.mandelbrot.x_step.clone(),
                y: self.next_pixel.y.clone(),
                y_scaled: self.next_pixel.y_scaled.clone()
            }
        }

        // Grab the pixel we will return.
        // By storing the value of the next pixel instead of the previous pixel it was possible
        // to initialize the generator without Option.
        std::mem::swap( &mut pixel, &mut self.next_pixel );

        // Return the pixel.
        return Some( pixel );
    }
}

/// Implements the properties of the grid for f32 floating-point type.
impl<const LANES: usize> MandelbrotGrid<LANES> for f64
    where
        LaneCount<LANES>: SupportedLaneCount
{
    type GridType = f64;
    type IteratorType = i64;

    // Grid limits.
    const X_MIN: f64 = -2.0;
    const X_MAX: f64 = 0.47;
    const Y_MIN: f64 = -1.12;
    const Y_MAX: f64 = 1.12;

    // Other constants.
    const X_MIN_SIMD: Simd<f64, LANES> = Simd::splat( -2.0 );
    const ZERO: Simd<f64, LANES> = Simd::splat( 0.0 );
    const TWO: Simd<f64, LANES> = Simd::splat( 2.0 );
    const MANDELBROT_CONDITION: Simd<f64, LANES> = Simd::splat( 2.0 * 2.0 );

    const MAX_ITERATIONS: i64 = 255;
    const ZERO_ITERATIONS: Simd<i64, LANES> = Simd::splat( 0 );
    const ITERATION_INCREMENT: Simd<i64, LANES> = Simd::splat( 1 );

    fn x_step(
        pixel_width: u32
    ) -> f64 {
        ( f64::abs( Self::X_MIN ) + f64::abs( Self::X_MAX  ) ) / pixel_width as f64
    }

    fn y_step(
        pixel_height: u32
    ) -> f64 {
        ( f64::abs( Self::Y_MIN ) + f64::abs( Self::Y_MAX  ) ) / pixel_height as f64
    }

    fn to_grid(
        value: &Simd<u32, LANES>
    ) -> Simd<f64, LANES>
    {
        let integers = value.to_array();
        Simd::from_slice( &integers.map( |i| i as f64 ) )
    }
    fn iteration_to_color(
        iteration: i64
    ) -> u8 {
        iteration as u8
    }

    fn reduce_max_iteration(
        iteration: &Simd<i64, LANES>
    ) -> i64 {
        iteration.reduce_max()
    }

    fn lanes_lt(
        product: Simd<f64, LANES>,
        condition: Simd<f64, LANES>,
    ) -> Mask<i64, LANES> {
        product.lanes_lt( condition )
    }

    fn select_iteration(
        mask: &Mask<i64, LANES>,
        iteration_proposal: Simd<i64, LANES>,
        previous_iteration: Simd<i64, LANES>,
    ) -> Simd<i64, LANES> {
        mask.select( iteration_proposal, previous_iteration )
    }
}

/// Implements the properties of the grid for floating-point type.
impl<const LANES: usize> MandelbrotGrid<LANES> for f32
    where
        LaneCount<LANES>: SupportedLaneCount
{
    type GridType = f32;
    type IteratorType = i32;

    // Grid limits.
    const X_MIN: f32 = -2.0;
    const X_MAX: f32 = 0.47;
    const Y_MIN: f32 = -1.12;
    const Y_MAX: f32 = 1.12;

    // Other constants.
    const X_MIN_SIMD: Simd<f32, LANES> = Simd::splat( -2.0 );
    const ZERO: Simd<f32, LANES> = Simd::splat( 0.0 );
    const TWO: Simd<f32, LANES> = Simd::splat( 2.0 );
    const MANDELBROT_CONDITION: Simd<f32, LANES> = Simd::splat( 2.0 * 2.0 );

    const MAX_ITERATIONS: i32 = 255;
    const ZERO_ITERATIONS: Simd<i32, LANES> = Simd::splat( 0 );
    const ITERATION_INCREMENT: Simd<i32, LANES> = Simd::splat( 1 );

    fn x_step(
        pixel_width: u32
    ) -> f32 {
        ( f32::abs( Self::X_MIN ) + f32::abs( Self::X_MAX  ) ) / pixel_width as f32
    }

    fn y_step(
        pixel_height: u32
    ) -> f32 {
        ( f32::abs( Self::Y_MIN ) + f32::abs( Self::Y_MAX  ) ) / pixel_height as f32
    }

    fn to_grid(
        value: &Simd<u32, LANES>
    ) -> Simd<f32, LANES>
    {
        let integers = value.to_array();
        Simd::from_slice( &integers.map( |i| i as f32 ) )
    }
    fn iteration_to_color(
        iteration: i32
    ) -> u8 {
        iteration as u8
    }

    fn reduce_max_iteration(
        iteration: &Simd<i32, LANES>
    ) -> i32 {
        iteration.reduce_max()
    }

    fn lanes_lt(
        product: Simd<f32, LANES>,
        condition: Simd<f32, LANES>,
    ) -> Mask<i32, LANES> {
        product.lanes_lt( condition )
    }

    fn select_iteration(
        mask: &Mask<i32, LANES>,
        iteration_proposal: Simd<i32, LANES>,
        previous_iteration: Simd<i32, LANES>,
    ) -> Simd<i32, LANES> {
        mask.select( iteration_proposal, previous_iteration )
    }
}

/// Calculates the colors for the pixel represented by the input.
pub fn calculate_color<T, const LANES: usize>(
    pixel: MandelbrotPixel<T, LANES>
) -> [u8; LANES]
    where
        T: MandelbrotGrid<LANES, GridType = T, IteratorType = <T as SimdElement>::Mask>,
        T: SimdElement + PartialOrd,
        LaneCount<LANES>: SupportedLaneCount,

        // Require Add, Mul and Sub SIMD operator support
        Simd<T, LANES>: Mul< Output  = Simd<T, LANES>> + Add< Output  = Simd<T, LANES>> + Sub< Output  = Simd<T, LANES>>,
        Simd<<T as SimdElement>::Mask, LANES>: Add<Output=Simd<<T as SimdElement>::Mask, LANES>>,
        <T as SimdElement>::Mask: Eq + Ord
{
    let mut x: Simd<T, LANES> = <T as MandelbrotGrid<LANES>>::ZERO.clone();
    let mut y: Simd<T, LANES> = <T as MandelbrotGrid<LANES>>::ZERO.clone();
    let mut iteration = <T as MandelbrotGrid<LANES>>::ZERO_ITERATIONS;
    loop  {

        // Iterations exhausted?
        // Only some of the lanes may reach this point if the series of the associated pixel goes past
        // the Mandelbrot condition.
        if <T as MandelbrotGrid<LANES>>::reduce_max_iteration( &iteration ) >= <T as MandelbrotGrid<LANES>>::MAX_ITERATIONS {
            break;
        }

        // Increment only those lanes that satisfy the Mandelbrot condition.
        let condition_mask = <T as MandelbrotGrid<LANES>>::lanes_lt( x*x + y*y, <T as MandelbrotGrid<LANES>>::MANDELBROT_CONDITION );
        if condition_mask.any() == false {
            break;
        }
        let previous_iteration = iteration.clone();
        iteration = <T as MandelbrotGrid<LANES>>::select_iteration(
                &condition_mask, iteration + <T as MandelbrotGrid<LANES>>::ITERATION_INCREMENT, previous_iteration );

        // Next value in the series for each pixel.
        let temp: Simd<T, LANES> = x*x - y*y + pixel.x_scaled.clone();
        y = <T as MandelbrotGrid<LANES>>::TWO*x*y + pixel.y_scaled.clone();
        x = temp;
    }

    return iteration.to_array().map( |value| <T as MandelbrotGrid<LANES>>::iteration_to_color( value ) );
}

/// Initializes first lane for the x pixel at the beginning of of a row.
const fn get_x_start<const LANES: usize>() -> [u32; LANES] {
    let mut output: [u32; LANES] = [0; LANES];
    let mut l  = 0;
    while l < LANES {
        output[ l ] = l as u32;
        l += 1;
    }
    output
}