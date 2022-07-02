use std::cmp::Ordering;
use std::convert::Infallible;
use std::iter::{Chain, Cloned, Copied, Cycle, Enumerate, Filter, FilterMap, FlatMap, Flatten, Fuse, Inspect, Map, MapWhile, Peekable, Product, Rev, Scan, Skip, SkipWhile, StepBy, Sum, Take, TakeWhile, Zip};
use std::path::Path;
use std::fs::File;
use std::io::BufWriter;
use png;
use png::text_metadata::{ITXtChunk, ZTXtChunk};
use rayon::prelude::*;


struct Mandlebrot {

    width: u32,
    height: u32,
    x_step: f64,
    y_step: f64,
}

struct MandlebrotIterator<'a> {

    mandlebrot: &'a Mandlebrot,
    next_pixel: MandlebrotPixel,
}

impl Mandlebrot {

    const X_MIN: f64 = -2.0;
    const Y_MIN: f64 = -1.12;

    pub fn new(
        width: u32,
        height: u32,
    ) -> Mandlebrot {

        // Scale the pixels.
        let x_step = ( f64::abs( Mandlebrot::X_MIN ) + f64::abs( 0.47  ) ) / width as f64;
        let y_step = ( f64::abs( Mandlebrot::Y_MIN ) + f64::abs( 1.12  ) ) / height as f64;

        // Start.
        Mandlebrot {
            width, height,
            x_step, y_step,

        }
    }

    pub fn iter(
        &self
    ) -> MandlebrotIterator {
        MandlebrotIterator {
            mandlebrot: self,
            next_pixel: MandlebrotPixel { x: 0, x_scaled: Mandlebrot::X_MIN, y: 0, y_scaled: Mandlebrot::Y_MIN },
        }
    }
}

impl<'a> Iterator for MandlebrotIterator<'a> {
    type Item = MandlebrotPixel;


    /// Gets next pixel that fills the grid.
    fn next(&mut self) -> Option<Self::Item> {

        // Last pixel returned?
        if self.next_pixel.y == self.mandlebrot.height {
            return None;
        };

        // Increment x axis first.
        let mut pixel;
        if self.next_pixel.x == self.mandlebrot.width - 1 {
            pixel = MandlebrotPixel {
                x: 0,
                x_scaled: Mandlebrot::X_MIN,
                y: self.next_pixel.y + 1,
                y_scaled: Mandlebrot::Y_MIN + ( self.next_pixel.y + 1 ) as f64 * self.mandlebrot.y_step,
            }
        }
        else {
            pixel = MandlebrotPixel {
                x: self.next_pixel.x + 1,
                x_scaled: Mandlebrot::X_MIN + ( self.next_pixel.x + 1 ) as f64 * self.mandlebrot.x_step,
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

/// A pixel in Mandlebrot scale.
struct MandlebrotPixel {

    x: u32,
    y: u32,
    x_scaled: f64,
    y_scaled: f64,
}

fn main() {

    // let path = std::env::args()
    //     .nth(1)
    //     .expect("Expected a filename to output to.");
    let file = File::create( "/home/fluxie/mandlebrot.png").unwrap();
    let ref mut w = BufWriter::new( file );

    let width = 10000;
    let height = 10000;
    let mut encoder = png::Encoder::new(w, width, height);
    encoder.set_color( png::ColorType::Grayscale );
    encoder.set_depth( png::BitDepth::Eight );
    let mut writer = encoder.write_header().expect( "Preparing image failed." );

    let mut mandlebrot = Mandlebrot::new(width, height );

    // Calculate color for each pixel.
    let image_data: Vec<u8> = mandlebrot.iter().par_bridge()
        .map( |mpixel| { calculate_color( mpixel ) } )
        .collect();
    writer.write_image_data(& image_data).expect( "Invalid data");
    writer.finish().expect( "Finalized" );

    println!("Hello, world!");
}

fn calculate_color(
    pixel: MandlebrotPixel
) -> u8 {
    let mut x = 0.0;
    let mut y = 0.0;
    const max_iterations: u8 = 255;
    let mut iteration = 0;
    while x*x + y*y < 2.0*2.0 && iteration < max_iterations {

        // Next pixel.
        let temp: f64 = x*x - y*y + pixel.x_scaled;
        y = 2.0*x*y + pixel.y_scaled;
        x = temp;

        iteration =  iteration + 1;
    }
    return iteration;
}