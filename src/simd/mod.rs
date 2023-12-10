use std::array::IntoIter;
use std::ops::{Add, Mul, Sub};
use std::simd::prelude::*;
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

    /// Defines the size of a step in the mandelbrot grid.
    step: MandelbrotGridStep<T, LANES>,

    /// Holds a scaled vector with initial values for the X-axis when the algorithm starts a new row.
    x_scale_start: Simd<T, LANES>,
}

pub struct Color<C, const LANES: usize>
    where
        C: ColorDepth<ColorType=C>
{

    /// Position of the color value in the grid.
    pub position: u64,

    /// The color.
    pub color: [C; LANES]
}

struct MandelbrotGridStep<T, const LANES: usize>
    where
        T: MandelbrotGrid<LANES, GridType = T>,
        T: SimdElement + PartialOrd,
        LaneCount<LANES>: SupportedLaneCount,

        // Require Add, Mul and Sub SIMD operator support
        Simd<T, LANES>: Mul< Output  = Simd<T, LANES>> + Add< Output  = Simd<T, LANES>> + Sub< Output  = Simd<T, LANES>>
{

    /// The size of the increment of each pixel in Mandelbrot scale in X-axis.
    x: Simd<T, LANES>,

    /// The size of the increment of each pixel in Mandelbrot scale in Y-axis.
    y: Simd<T, LANES>,
}

/// A tile or a window to the Mandelbrot grid.
pub struct MandelbrotTile<'a, T, const LANES: usize>
    where
        T: MandelbrotGrid<LANES, GridType = T, IteratorType = <T as SimdElement>::Mask>,
        T: SimdElement + PartialOrd,
        LaneCount<LANES>: SupportedLaneCount,

    // Require Add, Mul and Sub SIMD operator support
    Simd<T, LANES>: Mul< Output  = Simd<T, LANES>> + Add< Output  = Simd<T, LANES>> + Sub< Output  = Simd<T, LANES>>
{
    /// Reference to the Mandelbrot parameters.
    mandelbrot: &'a Mandelbrot<T, LANES>,

    /// Width of the window in pixels.
    width: u32,

    /// X-coordinate of the relative bottom-left corner of the tile in the original grid.
    x: u32,

    /// Height of the window in image in pixels.
    height: u32,

    /// Y-coordinate of the relative bottom-left corner of the tile in the original grid.
    y: u32,
}

/// An iterator that returns all the pixel for the Mandelbrot image.
pub struct MandelbrotTileIterator<'a, T, B, const LANES: usize>
    where
        T: MandelbrotGrid<LANES, GridType = T, IteratorType = <T as SimdElement>::Mask>,
        T: SimdElement + PartialOrd,
        LaneCount<LANES>: SupportedLaneCount,
        Simd<T, LANES>: Mul< Output  = Simd<T, LANES>> + Add< Output  = Simd<T, LANES>> + Sub< Output  = Simd<T, LANES>>,
        Simd<<T as SimdElement>::Mask, LANES>: Add<Output=Simd<<T as SimdElement>::Mask, LANES>>,
        B: MandelbrotBounds<T, LANES>,
{
    /// The area that is iterated over.
    bounds: &'a B,

    /// Next window returned by the iterator.
    next_tile: MandelbrotTile<'a, T, LANES>,
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

/// An iterator that returns all the pixel for Mandelbrot image.
pub struct MandelbrotPixelIterator<'a, T, BOUNDS, const LANES: usize>
    where
        T: MandelbrotGrid<LANES, GridType = T, IteratorType = <T as SimdElement>::Mask>,
        T: SimdElement + PartialOrd,
        LaneCount<LANES>: SupportedLaneCount,
        Simd<T, LANES>: Mul< Output  = Simd<T, LANES>> + Add< Output  = Simd<T, LANES>> + Sub< Output  = Simd<T, LANES>>,
        Simd<<T as SimdElement>::Mask, LANES>: Add<Output=Simd<<T as SimdElement>::Mask, LANES>>,
        BOUNDS: MandelbrotBounds<T, LANES>,
{
    /// Reference to the Mandelbrot parameters.
    step: &'a MandelbrotGridStep<T, LANES>,

    /// Reference to the Mandelbrot parameters.
    bounds: &'a BOUNDS,

    /// Next pixel returned by the iterator.
    next_pixel: MandelbrotPixel<T, LANES>,
}

/// Defines information about the depth of the color.
pub trait ColorDepth {

    type ColorType;

    /// Maximum number of iterations this color supports.
    const MAX_SUPPORTED_ITERATIONS: u32;

    /// The default scaling for the color depth.
    const DEFAULT_SCALING_FACTOR: u32;

    /// Scales the color.
    fn scale_color(
        unscaled_color: u32,
    ) -> Self::ColorType;
}

pub trait MandelbrotBounds<T, const LANES: usize>
    where
        T: MandelbrotGrid<LANES, GridType = T, IteratorType = <T as SimdElement>::Mask>,
        T: SimdElement + PartialOrd,
        LaneCount<LANES>: SupportedLaneCount,
        Simd<T, LANES>: Mul< Output  = Simd<T, LANES>> + Add< Output  = Simd<T, LANES>> + Sub< Output  = Simd<T, LANES>>,
        Simd<<T as SimdElement>::Mask, LANES>: Add<Output=Simd<<T as SimdElement>::Mask, LANES>>
{
    /// Returns X-coordinate of the bounded area.
    fn x(
        &self,
    ) -> u32;

    /// Returns the width of the bounded area.
    fn width(
        &self,
    ) -> u32;

    /// Returns the y-coordinate of the bounded area.
    fn y(
        &self,
    ) -> u32;

    /// Returns the height of the bounded area.
    fn height(
        &self,
    ) -> u32;

    /// Returns a scaled vector with initial values for the X-axis when the algorithm starts a new row.
    fn x_scale_start(
      &self
    ) -> Simd<T, LANES>;
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

    /// Converts the given input vector into iteration type.
    fn to_iteration(
        value: &Simd<u32, LANES>
    ) -> Simd<Self::IteratorType, LANES>;

    /// Converts the reached iteration value into a grayscale color value.
    fn iteration_to_unscaled_color(
        iteration: <Self::GridType as SimdElement>::Mask
    ) -> u32;

    /// Returns the maximum value of a lane in the given iteration.
    fn reduce_max_iteration(
        iteration: &Simd<Self::IteratorType, LANES>
    ) -> u32;

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
    const X_START: Simd<u32, LANES> = Simd::from_array( get_x_start( 0 ) );

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
            step: MandelbrotGridStep {
                x: x_step,
                y: y_step,
            },
            x_scale_start
        } )
    }

    /// Returns an iterator to the pixels in the Mandelbrot.
    #[allow(dead_code)]
    pub fn iter_pixels(
        &self
    ) -> MandelbrotPixelIterator<T, Mandelbrot<T, LANES>, LANES> {

        MandelbrotPixelIterator {
            step: &self.step,
            bounds: &self,
            next_pixel: MandelbrotPixel {
                x: Mandelbrot::X_START, x_scaled: self.x_scale_start.clone(),
                y: Simd::splat( 0 ), y_scaled: Simd::<T, LANES>::splat( <T as MandelbrotGrid<LANES>>::Y_MIN )
            },
        }
    }

    /// Returns an iterator to the pixels in the Mandelbrot.
    #[allow(dead_code)]
    pub fn iter_tiles(
        &self
    ) -> MandelbrotTileIterator<T, Mandelbrot<T, LANES>, LANES> {

        // Determine the maximum tile size.
        // The tile size must be divisible with the size of the image to make it fit.
        // Ideally each CPU will handle one tile.
        let total_size = self.width * self.height;
        let proposed_tile_size: u32 = total_size / num_cpus::get() as u32;
        let total_tile_size = num::integer::gcd( total_size, proposed_tile_size );

        // Calculate the number of tiles in each direction.
        let tile_width = num::integer::gcd( self.width, total_tile_size );
        let x_tiles = self.width / tile_width;
        let y_tiles = total_tile_size / ( x_tiles * tile_width );
        let y_tiles = num::integer::gcd( self.height, y_tiles );
        let tile_height = self.height / y_tiles;

        MandelbrotTileIterator {
            bounds: self,
            next_tile: MandelbrotTile {
                mandelbrot: self,
                x: 0, width: tile_width,
                y: 0, height: tile_height,
            },
        }
    }
}

impl<T, const LANES: usize> MandelbrotBounds<T, LANES>for Mandelbrot<T, LANES>
    where
        T: MandelbrotGrid<LANES, GridType = T, IteratorType = <T as SimdElement>::Mask>,
        T: SimdElement + PartialOrd,
        LaneCount<LANES>: SupportedLaneCount,

        // Require Add, Mul and Sub SIMD operator support
        Simd<T, LANES>: Mul< Output  = Simd<T, LANES>> + Add< Output  = Simd<T, LANES>> + Sub< Output  = Simd<T, LANES>>,
        Simd<<T as SimdElement>::Mask, LANES>: Add<Output=Simd<<T as SimdElement>::Mask, LANES>>
{
    /// Returns X-coordinate of the bounded area.
    fn x(
        &self,
    ) -> u32 {
        0
    }

    /// Returns the width of the bounded area.
    fn width(
        &self,
    ) -> u32 {
        self.width
    }

    /// Returns the y-coordinate of the bounded area.
    fn y(
        &self,
    ) -> u32 {
        0
    }

    /// Returns the height of the bounded area.
    fn height(
        &self,
    ) -> u32 {
        self.height
    }

    // Returns a scaled vector with initial values for the X-axis when the algorithm starts a new row.
    fn x_scale_start(
        &self
    ) -> Simd<T, LANES> {
        self.x_scale_start.clone()
    }
}

impl<C, const LANES: usize> IntoIterator for Color<C, LANES>
    where
        C: ColorDepth<ColorType=C>
{
    type Item = C;
    type IntoIter = IntoIter<C, LANES>;

    fn into_iter(self) -> Self::IntoIter {
        self.color.into_iter()
    }
}

impl<'a, T, const LANES: usize> MandelbrotTile<'a, T, LANES>
    where
        T: MandelbrotGrid<LANES, GridType = T, IteratorType = <T as SimdElement>::Mask>,
        T: SimdElement + PartialOrd,
        LaneCount<LANES>: SupportedLaneCount,

        // Require Add, Mul and Sub SIMD operator support
        Simd<T, LANES>: Mul< Output  = Simd<T, LANES>> + Add< Output  = Simd<T, LANES>> + Sub< Output  = Simd<T, LANES>>,
        Simd<<T as SimdElement>::Mask, LANES>: Add<Output=Simd<<T as SimdElement>::Mask, LANES>>
{
    /// Iterates the pixels of this tile.
    pub fn iter_pixels(
        &self,
    ) -> MandelbrotPixelIterator<T, MandelbrotTile<'a, T, LANES>, LANES > {

        // Calculate X and Y coordinates.
        let x = Simd::from_array( get_x_start( self.x ) );
        let y_min = Simd::splat( <T as MandelbrotGrid<LANES>>::Y_MIN );
        let y = Simd::splat( self.y );

        let next_pixel = MandelbrotPixel {
            x,
            y,
            x_scaled: <T as MandelbrotGrid<LANES>>::X_MIN_SIMD + <T as MandelbrotGrid<LANES>>::to_grid(&x) * self.mandelbrot.step.x.clone(),
            y_scaled: y_min + <T as MandelbrotGrid<LANES>>::to_grid(&y) * self.mandelbrot.step.y.clone(),
        };

        MandelbrotPixelIterator {
            step: &self.mandelbrot.step,
            bounds: self,
            next_pixel,
        }

    }
}

impl<'a, T, const LANES: usize> MandelbrotBounds<T, LANES>for MandelbrotTile<'a, T, LANES>
    where
        T: MandelbrotGrid<LANES, GridType = T, IteratorType = <T as SimdElement>::Mask>,
        T: SimdElement + PartialOrd,
        LaneCount<LANES>: SupportedLaneCount,

    // Require Add, Mul and Sub SIMD operator support
        Simd<T, LANES>: Mul< Output  = Simd<T, LANES>> + Add< Output  = Simd<T, LANES>> + Sub< Output  = Simd<T, LANES>>,
        Simd<<T as SimdElement>::Mask, LANES>: Add<Output=Simd<<T as SimdElement>::Mask, LANES>>
{
    /// Returns X-coordinate of the bounded area.
    fn x(
        &self,
    ) -> u32 {
        self.x
    }

    /// Returns the width of the bounded area.
    fn width(
        &self,
    ) -> u32 {
        self.width
    }

    /// Returns the y-coordinate of the bounded area.
    fn y(
        &self,
    ) -> u32 {
        self.y
    }

    /// Returns the height of the bounded area.
    fn height(
        &self,
    ) -> u32 {
        self.height
    }

    // Returns a scaled vector with initial values for the X-axis when the algorithm starts a new row.
    fn x_scale_start(
        &self
    ) -> Simd<T, LANES> {
        &self.mandelbrot.x_scale_start + ( &self.mandelbrot.step.x * <T as MandelbrotGrid<LANES>>::to_grid( &Simd::splat( self.x ) ) )
    }
}

impl<'a, T, B, const LANES: usize> Iterator for MandelbrotTileIterator<'a, T, B, LANES>
    where
        T: MandelbrotGrid<LANES, GridType = T, IteratorType = <T as SimdElement>::Mask>,
        T: SimdElement + PartialOrd,
        LaneCount<LANES>: SupportedLaneCount,

        // Require Add, Mul and Sub SIMD operator support
        Simd<T, LANES>: Mul< Output  = Simd<T, LANES>> + Add< Output  = Simd<T, LANES>> + Sub< Output  = Simd<T, LANES>>,
        Simd<<T as SimdElement>::Mask, LANES>: Add<Output=Simd<<T as SimdElement>::Mask, LANES>>,
        B: MandelbrotBounds<T, LANES>,
{
    type Item = MandelbrotTile<'a, T, LANES>;


    /// Gets a vector of the next tiles that fills the image grid.
    fn next(&mut self) -> Option<Self::Item> {

        // Last tile  returned?
        if self.next_tile.y == self.bounds.y() + self.bounds.height() {
            return None;
        };

        // Which axis to increment?
        let mut tile;
        if self.next_tile.x + self.next_tile.width == self.bounds.x() + self.bounds.width() {

            // Next row, increase Y-axis.
            tile = MandelbrotTile {
                mandelbrot: self.next_tile.mandelbrot,
                x: 0,
                width: self.next_tile.width,
                y: self.next_tile.y + self.next_tile.height,
                height: self.next_tile.height,
            }
        }
        else {

            // Continue on the same row. Increase X-axis.
            tile = MandelbrotTile {
                mandelbrot: self.next_tile.mandelbrot,
                x: self.next_tile.x + self.next_tile.width,
                width: self.next_tile.width,
                y: self.next_tile.y,
                height: self.next_tile.height,
            }
        }

        // Grab the tile we will return.
        // By storing the value of the next tile instead of the previous tile it was possible
        // to initialize the generator without Option.
        std::mem::swap(&mut tile, &mut self.next_tile );

        // Return the tile.
        return Some(tile);
    }
}

impl<'a, T, BOUNDS, const LANES: usize> MandelbrotPixelIterator<'a, T, BOUNDS, LANES>
    where
        T: MandelbrotGrid<LANES, GridType = T, IteratorType = <T as SimdElement>::Mask>,
        T: SimdElement + PartialOrd,
        LaneCount<LANES>: SupportedLaneCount,

        // Require Add, Mul and Sub SIMD operator support
        Simd<T, LANES>: Mul< Output  = Simd<T, LANES>> + Add< Output  = Simd<T, LANES>> + Sub< Output  = Simd<T, LANES>>,
        Simd<<T as SimdElement>::Mask, LANES>: Add<Output=Simd<<T as SimdElement>::Mask, LANES>>,

        BOUNDS: MandelbrotBounds<T, LANES>,
{
    const X_PIXER_STEP: Simd<u32, LANES> = array_splat( LANES as u32 );
    const Y_PIXER_STEP: Simd<u32, LANES> = array_splat( 1 );
}

impl<'a, T, BOUNDS, const LANES: usize> Iterator for MandelbrotPixelIterator<'a, T, BOUNDS, LANES>
    where
        T: MandelbrotGrid<LANES, GridType = T, IteratorType = <T as SimdElement>::Mask>,
        T: SimdElement + PartialOrd,
        LaneCount<LANES>: SupportedLaneCount,

        // Require Add, Mul and Sub SIMD operator support
        Simd<T, LANES>: Mul< Output  = Simd<T, LANES>> + Add< Output  = Simd<T, LANES>> + Sub< Output  = Simd<T, LANES>>,
        Simd<<T as SimdElement>::Mask, LANES>: Add<Output=Simd<<T as SimdElement>::Mask, LANES>>,

        BOUNDS: MandelbrotBounds<T, LANES>,
{
    type Item = MandelbrotPixel<T, LANES>;


    /// Gets a vector of the next pixels that fills the image grid.
    fn next(&mut self) -> Option<Self::Item> {

        // Last pixel returned?
        if self.next_pixel.y == Simd::splat( self.bounds.y() + self.bounds.height() ) {
            return None;
        };

        // Which axis to increment?
        let mut pixel;
        let x_boundary = self.bounds.x() + self.bounds.width();
        assert!( self.next_pixel.x[ LANES - 1 ] < x_boundary );
        if self.next_pixel.x[ LANES - 1 ] == x_boundary - 1 {

            // Next row, increase Y-axis.
            pixel = MandelbrotPixel {
                x: Simd::from_array( get_x_start( self.bounds.x() ) ),
                x_scaled: self.bounds.x_scale_start(),
                y: &self.next_pixel.y + MandelbrotPixelIterator::<T, BOUNDS, LANES>::Y_PIXER_STEP,
                y_scaled: Simd::splat( <T as MandelbrotGrid<LANES>>::Y_MIN ) + <T as MandelbrotGrid<LANES>>::to_grid(
                        &(&self.next_pixel.y + MandelbrotPixelIterator::<T, BOUNDS, LANES>::Y_PIXER_STEP )) * self.step.y.clone(),
            }
        }
        else {

            // Continue on the same row. Increase X-axis.
            pixel = MandelbrotPixel {
                x: &self.next_pixel.x + MandelbrotPixelIterator::<T, BOUNDS, LANES>::X_PIXER_STEP,
                x_scaled: <T as MandelbrotGrid<LANES>>::X_MIN_SIMD + <T as MandelbrotGrid<LANES>>::to_grid(
                        &(&self.next_pixel.x + MandelbrotPixelIterator::<T, BOUNDS, LANES>::X_PIXER_STEP ) ) * self.step.x.clone(),
                y: self.next_pixel.y.clone(),
                y_scaled: self.next_pixel.y_scaled.clone()
            }
        }

        // Grab the pixel we will return.
        // By storing the value of the next pixel instead of the previous pixel it was possible
        // to initialize the generator without Option.
        std::mem::swap( &mut pixel, &mut self.next_pixel );

        assert!( pixel.x[ 0 ] != pixel.x[ 1 ] );

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
    const X_MIN_SIMD: Simd<f64, LANES> = array_splat( -2.0 );
    const ZERO: Simd<f64, LANES> = array_splat( 0.0 );
    const TWO: Simd<f64, LANES> = array_splat( 2.0 );
    const MANDELBROT_CONDITION: Simd<f64, LANES> = array_splat( 2.0 * 2.0 );

    const ZERO_ITERATIONS: Simd<i64, LANES> = array_splat( 0 );
    const ITERATION_INCREMENT: Simd<i64, LANES> = array_splat( 1 );

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

    fn to_iteration(
        value: &Simd<u32, LANES>
    ) -> Simd<i64, LANES> {
        let integers = value.to_array();
        Simd::from_slice( &integers.map( |i| i as i64 ) )
    }

    fn iteration_to_unscaled_color(
        iteration: i64
    ) -> u32 {
        iteration as u32
    }

    fn reduce_max_iteration(
        iteration: &Simd<i64, LANES>
    ) -> u32 {
        iteration.reduce_max() as u32
    }

    fn lanes_lt(
        product: Simd<f64, LANES>,
        condition: Simd<f64, LANES>,
    ) -> Mask<i64, LANES> {
        product.simd_lt( condition )
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
    const X_MIN_SIMD: Simd<f32, LANES> = array_splat( -2.0 );
    const ZERO: Simd<f32, LANES> = array_splat( 0.0 );
    const TWO: Simd<f32, LANES> = array_splat( 2.0 );
    const MANDELBROT_CONDITION: Simd<f32, LANES> = array_splat( 2.0 * 2.0 );

    const ZERO_ITERATIONS: Simd<i32, LANES> = array_splat( 0 );
    const ITERATION_INCREMENT: Simd<i32, LANES> = array_splat( 1 );

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

    fn to_iteration(
        value: &Simd<u32, LANES>
    ) -> Simd<i32, LANES> {
        let integers = value.to_array();
        Simd::from_slice( &integers.map( |i| i as i32 ) )
    }

    fn iteration_to_unscaled_color(
        iteration: i32
    ) -> u32 {
        iteration as u32
    }

    fn reduce_max_iteration(
        iteration: &Simd<i32, LANES>
    ) -> u32 {
        iteration.reduce_max() as u32
    }

    fn lanes_lt(
        product: Simd<f32, LANES>,
        condition: Simd<f32, LANES>,
    ) -> Mask<i32, LANES> {
        product.simd_lt( condition )
    }

    fn select_iteration(
        mask: &Mask<i32, LANES>,
        iteration_proposal: Simd<i32, LANES>,
        previous_iteration: Simd<i32, LANES>,
    ) -> Simd<i32, LANES> {
        mask.select( iteration_proposal, previous_iteration )
    }
}

impl ColorDepth for u8 {

    type ColorType = u8;

    const MAX_SUPPORTED_ITERATIONS: u32 = ( u8::MAX ) as u32;

    /// The default scaling for the color depth.
    const DEFAULT_SCALING_FACTOR: u32 = 1;

    /// Scales the color.
    fn scale_color(
        unscaled_color: u32,
    ) -> u8 {
        std::cmp::min( unscaled_color, u8::MAX as u32  ) as u8
    }
}

impl ColorDepth for u16 {

    type ColorType = u16;

    const MAX_SUPPORTED_ITERATIONS: u32 = ( u16::MAX ) as u32;

    /// The default scaling for the color depth.
    /// TODO: For prettier images the color scaling should probably be larger at the beginning of the series.
    /// This way even small increase in the number of iterations would be visible in the final image.
    /// I.e. the scaling should not be linear.
    const DEFAULT_SCALING_FACTOR: u32 = 64;

    /// Scales the color.
    fn scale_color(
        unscaled_color: u32,
    ) -> u16 {
        std::cmp::min( unscaled_color, u16::MAX as u32  ) as u16
    }
}

/// Calculates the colors for the pixel represented by the input.
pub fn calculate_color<C, T, const LANES: usize>(
    pixel: MandelbrotPixel<T, LANES>,
) -> Color<C, LANES>
    where
        T: MandelbrotGrid<LANES, GridType = T, IteratorType = <T as SimdElement>::Mask>,
        T: SimdElement + PartialOrd,
        LaneCount<LANES>: SupportedLaneCount,
        C: ColorDepth<ColorType = C>,

    // Require Add, Mul and Sub SIMD operator support
        Simd<T, LANES>: Mul< Output  = Simd<T, LANES>> + Add< Output  = Simd<T, LANES>> + Sub< Output  = Simd<T, LANES>>,
        Simd<<T as SimdElement>::Mask, LANES>: Add<Output=Simd<<T as SimdElement>::Mask, LANES>>,
        <T as SimdElement>::Mask: Eq + Ord
{
    let max_iterations: u32 = <C as ColorDepth>::MAX_SUPPORTED_ITERATIONS / <C as ColorDepth>::DEFAULT_SCALING_FACTOR;
    calculate_color_with_iterations( pixel, max_iterations )
}

/// Calculates the colors for the pixel represented by the input.
pub fn calculate_color_with_iterations<C, T, const LANES: usize>(
    pixel: MandelbrotPixel<T, LANES>,
    max_iterations: u32,
) -> Color<C, LANES>
    where
        T: MandelbrotGrid<LANES, GridType = T, IteratorType = <T as SimdElement>::Mask>,
        T: SimdElement + PartialOrd,
        LaneCount<LANES>: SupportedLaneCount,
        C: ColorDepth<ColorType = C>,

        // Require Add, Mul and Sub SIMD operator support
        Simd<T, LANES>: Mul< Output  = Simd<T, LANES>> + Add< Output  = Simd<T, LANES>> + Sub< Output  = Simd<T, LANES>>,
        Simd<<T as SimdElement>::Mask, LANES>: Add<Output=Simd<<T as SimdElement>::Mask, LANES>>,
        <T as SimdElement>::Mask: Eq + Ord
{
    let mut x: Simd<T, LANES> = <T as MandelbrotGrid<LANES>>::ZERO.clone();
    let mut y: Simd<T, LANES> = <T as MandelbrotGrid<LANES>>::ZERO.clone();
    let mut iteration = <T as MandelbrotGrid<LANES>>::ZERO_ITERATIONS;
    let scaling_factor = <C as ColorDepth>::MAX_SUPPORTED_ITERATIONS / max_iterations;
    let iteration_increment = Simd::splat( scaling_factor );
    let iteration_increment = <T as MandelbrotGrid<LANES>>::to_iteration( &iteration_increment );
    loop  {

        // Iterations exhausted?
        // Only some of the lanes may reach this point if the series of the associated pixel goes past
        // the Mandelbrot condition.
        if <T as MandelbrotGrid<LANES>>::reduce_max_iteration( &iteration ) >= <C as ColorDepth>::MAX_SUPPORTED_ITERATIONS {
            break;
        }

        // Increment only those lanes that satisfy the Mandelbrot condition.
        let condition_mask = <T as MandelbrotGrid<LANES>>::lanes_lt( x*x + y*y, <T as MandelbrotGrid<LANES>>::MANDELBROT_CONDITION );
        if condition_mask.any() == false {
            break;
        }
        let previous_iteration = iteration.clone();
        iteration = <T as MandelbrotGrid<LANES>>::select_iteration(
                &condition_mask, iteration + iteration_increment, previous_iteration );

        // Next value in the series for each pixel.
        let temp: Simd<T, LANES> = x*x - y*y + pixel.x_scaled.clone();
        y = <T as MandelbrotGrid<LANES>>::TWO*x*y + pixel.y_scaled.clone();
        x = temp;
    }

    // Calculate the position and the final color.
    let position: u64 = ( pixel.y[ 0 ] as u64 ) << 32;
    let position = position + ( pixel.x [ 0 ] as u64 );
    let unscaled_color = iteration.to_array().map( |value| <T as MandelbrotGrid<LANES>>::iteration_to_unscaled_color( value ) );
    let color = unscaled_color.map( |color| <C as ColorDepth>::scale_color( color ) );

    return Color {
        position,
        color
    }
}

/// Initializes first lane for the x pixel at the beginning of of a row.
const fn get_x_start<const LANES: usize>(
    x: u32
) -> [u32; LANES] {
    let mut output: [u32; LANES] = [0; LANES];
    let mut l  = x as usize;
    while l < LANES + x as usize {
        output[ l - x as usize ] = l as u32;
        l += 1;
    }
    output
}

const fn array_splat<Scalar, const LANES: usize>(
    value: Scalar
) -> Simd<Scalar, LANES>
    where
        LaneCount<LANES>: SupportedLaneCount,
        Scalar: SimdElement
{
    Simd::from_array( [value; LANES] )
}

#[cfg(test)]
mod tests {


    use super::*;

    #[test]
    fn get_x_start_returns_appropriate_vector()
    {
        let test_vectors = [ ( 0, [ 0, 1, 2, 3 ] ), ( 4, [ 4, 5, 6, 7 ] )];
        for test in test_vectors {

            let result = get_x_start( test.0 );
            assert_eq!( result, test.1 );
        }
    }

    #[test]
    fn center_is_white()
    {
        // Ensures the center of the grid is white.
        let x_min = <f32 as MandelbrotGrid<4>>::X_MIN_SIMD.clone();
        let y_min = <f32 as MandelbrotGrid<4>>::Y_MIN.clone();
        let mandelbrot = Mandelbrot::<f32, 4>::new( 32, 32 ).unwrap();
        let pixel = MandelbrotPixel::<f32, 4> {
            x: Simd::splat( 16 ), y: Simd::splat( 16 ),
            x_scaled: x_min + mandelbrot.step.x * Simd::splat( 16.0 ),
            y_scaled: Simd::splat( y_min ) + mandelbrot.step.y * Simd::splat( 16.0 )
        };

        let color: Color::<u16, 4> = calculate_color( pixel );
        assert_eq!( color.color.to_vec(), vec![ u16::MAX; 4 ] );
    }
}