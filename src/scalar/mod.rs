
/// Defines the parameters for the Mandelbrot image.
pub struct Mandelbrot {

    /// Width of the final image in pixels.
    width: u32,

    /// Height of the final image in pixels.
    height: u32,

    /// The size of the increment of each pixel in Mandelbrot scale in X-axis.
    x_step: f64,

    /// The size of the increment of each pixel in Mandelbrot scale in Y-axis.
    y_step: f64,
}

/// A pixel in Mandelbrot scale.
pub struct MandelbrotPixel {

    /// Defines the position of the pixel in the final image in the X-acis.
    pub x: u32,

    /// Defines the position of the pixel in the final image in the Y-acis.
    pub y: u32,

    /// Pixel scaled to the area where the starting points may fill the Mandelbrot condition.
    pub x_scaled: f64,

    /// Pixel scaled to the area where the starting points may fill the Mandelbrot condition.
    pub y_scaled: f64,
}

/// Iterator that returns all the pixels that fit into the Mandelbrot.
pub struct MandelbrotIterator<'a> {

    /// Reference to the Mandelbrot parameters.
    mandelbrot: &'a Mandelbrot,

    /// Next pixel returned by the iterator.
    next_pixel: MandelbrotPixel,
}

impl Mandelbrot {

    /// Minimum starting value of the set in the X-axis that can satisfy the Mandelbrot condition.
    const X_MIN: f64 = -2.0;

    /// Minimum starting value of the set in the Y-axis that can satisfy the Mandelbrot condition.
    const Y_MIN: f64 = -1.12;

    /// Initializes new image with the given parameters.
    #[allow(dead_code)]
    pub fn new(
        width: u32,
        height: u32,
    ) -> Mandelbrot {

        // Scale the pixels.
        let x_step = ( f64::abs( Mandelbrot::X_MIN ) + f64::abs( 0.47  ) ) / width as f64;
        let y_step = ( f64::abs( Mandelbrot::Y_MIN ) + f64::abs( 1.12  ) ) / height as f64;

        // Start.
        Mandelbrot {
            width, height,
            x_step, y_step,

        }
    }

    /// Returns an iterator to the pixels in the Mandelbrot.
    #[allow(dead_code)]
    pub fn iter(
        &self
    ) -> MandelbrotIterator {

        MandelbrotIterator {
            mandelbrot: self,
            next_pixel: MandelbrotPixel { x: 0, x_scaled: Mandelbrot::X_MIN, y: 0, y_scaled: Mandelbrot::Y_MIN },
        }
    }
}

impl<'a> Iterator for MandelbrotIterator<'a> {
    type Item = MandelbrotPixel;


    /// Gets next pixel that fills the image grid.
    fn next(&mut self) -> Option<Self::Item> {

        // Last pixel returned?
        if self.next_pixel.y == self.mandelbrot.height {
            return None;
        };

        // Increment x axis first.
        let mut pixel;
        if self.next_pixel.x == self.mandelbrot.width - 1 {
            pixel = MandelbrotPixel {
                x: 0,
                x_scaled: Mandelbrot::X_MIN,
                y: self.next_pixel.y + 1,
                y_scaled: Mandelbrot::Y_MIN + ( self.next_pixel.y + 1 ) as f64 * self.mandelbrot.y_step,
            }
        }
        else {
            pixel = MandelbrotPixel {
                x: self.next_pixel.x + 1,
                x_scaled: Mandelbrot::X_MIN + ( self.next_pixel.x + 1 ) as f64 * self.mandelbrot.x_step,
                y: self.next_pixel.y,
                y_scaled: self.next_pixel.y_scaled
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

/// Calculates the color for a pixel.
#[allow(dead_code)]
pub fn calculate_color(
    pixel: MandelbrotPixel
) -> u8 {
    let mut x = 0.0;
    let mut y = 0.0;
    const MAX_ITERATIONS: u8 = 255;
    let mut iteration = 0;
    while x*x + y*y < 2.0*2.0 && iteration < MAX_ITERATIONS {

        // Next value in the series for each pixel.
        let temp: f64 = x*x - y*y + pixel.x_scaled;
        y = 2.0*x*y + pixel.y_scaled;
        x = temp;

        iteration =  iteration + 1;
    }
    return iteration;
}