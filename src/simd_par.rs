//! Vectorized parallel Mandelbrot implementation
#![allow(non_camel_case_types)]

use crate::*;
use packed_simd::*;

type u32s = u32x8;
type f32s = f32x8;
type m32s = m32x8;

/// Storage for complex numbers in SIMD format.
/// The real and imaginary parts are kept in separate registers.
#[derive(Copy, Clone)]
struct Complex {
    real: f32s,
    imag: f32s,
}

const THRESHOLD: f32 = 4.0;

impl Complex {
    /// Returns a mask describing which members of the Mandelbrot sequence
    /// haven't diverged yet
    #[inline]
    fn undiverged(&self) -> m32s {
        let Self { real: x, imag: y } = *self;

        let xx = x * x;
        let yy = y * y;
        let sum = xx + yy;

        sum.le(f32s::splat(THRESHOLD))
    }
}

/// Mandelbrot sequence iterator using SIMD.
struct MandelbrotIter {
    /// Initial value which generated this sequence
    start: Complex,
    /// Current iteration value
    current: Complex,
}

impl MandelbrotIter {
    /// Creates a new Mandelbrot sequence iterator for a given starting point
    fn new(start: Complex) -> Self {
        Self { start, current: start }
    }

    /// Returns the number of iterations it takes for each member of the
    /// Mandelbrot sequence to diverge at this point, or `ITER_LIMIT` if
    /// they don't diverge.
    ///
    /// This function will operate on N complex numbers at once, where N is the
    /// number of lanes in a SIMD vector of doubles.
    fn count(mut self, iters: u32) -> u32s {
        let mut z = self.start;
        let mut count = u32s::splat(0);
        for _ in 0..iters {
            // Keep track of those lanes which haven't diverged yet. The other
            // ones will be masked off.
            let undiverged = z.undiverged();

            // Stop the iteration if they all diverged. Note that we don't do
            // this check every iteration, since a branch
            // misprediction can hurt more than doing some extra
            // calculations.
            if undiverged.none() {
                break;
            }

            count += undiverged.select(u32s::splat(1), u32s::splat(0));

            z = self.next().unwrap();
        }
        count.cast()
    }
}

impl Iterator for MandelbrotIter {
    type Item = Complex;

    /// Generates the next values in the sequence
    #[inline]
    fn next(&mut self) -> Option<Complex> {
        let Complex { real: c_x, imag: c_y } = self.start;
        let Complex { real: x, imag: y } = self.current;

        let xx = x * x;
        let yy = y * y;
        let xy = x * y;

        let new_x = c_x + (xx - yy);
        let new_y = c_y + (xy + xy);

        self.current = Complex { real: new_x, imag: new_y };

        Some(self.current)
    }
}

pub fn generate(min_x: f32, max_x: f32, min_y: f32, max_y: f32, width: usize, height: usize, iters: u32) -> Vec<u32> {
    let block_size = f32s::lanes();

    assert_eq!(
        width % block_size,
        0,
        "image width = {} is not divisible by the number of vector lanes = {}",
        width,
        block_size,
    );

    let width_in_blocks = width / block_size;

    // The initial X values are the same for every row.
    let xs = unsafe {
        let dx = (max_x - min_x) / (width as f32);
        let mut buf: Vec<f32s> = vec![f32s::splat(0.); width_in_blocks];

        std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut f32, width)
            .iter_mut()
            .enumerate()
            .for_each(|(j, x)| {
                *x = min_x + dx * (j as f32);
            });

        buf
    };

    let dy = (max_y - min_y) / (height as f32);

    let len = width_in_blocks * height;
    let mut out = Vec::with_capacity(len);
    unsafe {
        out.set_len(len);
    }

    out.par_chunks_mut(width_in_blocks).enumerate().for_each(|(i, row)| {
        let y = f32s::splat(min_y + dy * (i as f32));
        row.iter_mut().enumerate().for_each(|(j, count)| {
            let x = xs[j];
            let z = Complex { real: x, imag: y };
            *count = MandelbrotIter::new(z).count(iters);
        });
    });

    // This is safe, we're transmuting from a more-aligned type to a
    // less-aligned one.
    #[allow(clippy::unsound_collection_transmute)]
    unsafe {
        let mut out: Vec<u32> = std::mem::transmute(out);
        out.set_len(width * height);
        out
    }
}
