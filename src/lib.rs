use pyo3::prelude::*;
use ndarray::{Dim, ArrayViewMut, IxDynImpl};
use numpy::{PyArray, PyArrayDyn, IntoPyArray};
use rayon::prelude::*;

mod simd_par;

#[pymodule]
fn mandelbrot(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_mandelbrot_rs, m)?)?;
    m.add_function(wrap_pyfunction!(compute_mandelbrot_rs_par, m)?)?;
    m.add_function(wrap_pyfunction!(compute_mandelbrot_rs_simd_par, m)?)?;
    Ok(())
}

#[pyfunction]
fn compute_mandelbrot_rs(
    _py: Python<'_>,
    min_x: f32, 
    max_x: f32, 
    min_y: f32, 
    max_y: f32, 
    width: u32, 
    height: u32, 
    t: &PyArrayDyn<u8>,
    iters: u8,
) {
    let mut a = unsafe { t.as_array_mut() }; 
    compute_mandelbrot(min_x, max_x, min_y, max_y, width, height, iters, &mut a);
}

#[pyfunction]
fn compute_mandelbrot_rs_par<'py>(
    py: Python<'py>,
    min_x: f32, 
    max_x: f32, 
    min_y: f32, 
    max_y: f32, 
    width: u32, 
    height: u32, 
    iters: u8,
) -> &'py PyArray<u8, Dim<[usize; 2]>> {
    let out = compute_mandelbrot_par(min_x, max_x, min_y, max_y, width, height, iters);
    out.into_pyarray(py).reshape([height as usize, width as usize]).unwrap()
}

#[pyfunction]
fn compute_mandelbrot_rs_simd_par<'py>(
    py: Python<'py>,
    min_x: f64, 
    max_x: f64, 
    min_y: f64, 
    max_y: f64, 
    width: usize, 
    height: usize, 
    iters: u32,
) -> &'py PyArray<u32, Dim<[usize; 2]>> {
    let out = simd_par::generate(min_x, max_x, min_y, max_y, width, height, iters);
    out.into_pyarray(py).reshape([height as usize, width as usize]).unwrap()
}


#[derive(Clone, Copy)]
struct Complex {
    pub a: f32,
    pub b: f32,
}

impl std::ops::Add for Complex {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Complex {
            a: self.a + rhs.a,
            b: self.b + rhs.b,
        }
    }
}

impl std::ops::Mul for Complex {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Complex { 
            a: self.a * rhs.a - self.b * rhs.b, 
            b: self.a * rhs.b + self.b * rhs.a ,
        }
    }
}

impl Complex {
    fn arg_sq(self) -> f32 {
        self.a * self.a + self.b * self.b
    }
}

fn mandelbrot_kernel(x: f32, y: f32, max: u8) -> u8 {
    let mut z = Complex { a: 0.0, b: 0.0 };
    let c = Complex { a: x, b: y };
    let mut i = 0u8;
    while i < max && z.arg_sq() < 4.0 {
        z = z * z + c;
        i += 1;
    }
    return i;
}

fn compute_mandelbrot(min_x: f32, max_x: f32, min_y: f32, max_y: f32, width: u32, height: u32, iters: u8, t: &mut ArrayViewMut<'_, u8, Dim<IxDynImpl>>) {
    let dx = (max_x - min_x) / width as f32;
    let dy = (max_y - min_y) / height as f32;
    let mut y = min_y;
    for row in 0..height {
        let mut x = min_x;
        for col in 0..height{
            t[[row as usize, col as usize]] = mandelbrot_kernel(x, y, iters);
            x += dx;
        }
        y += dy;
    }
}

fn compute_mandelbrot_par(min_x: f32, max_x: f32, min_y: f32, max_y: f32, width: u32, height: u32, iters: u8) -> Vec<u8> {
    let len = (width * height) as usize;
    let mut out = Vec::with_capacity(len);
    unsafe {
        out.set_len(len);
    }

    let dx = (max_x - min_x) / width as f32;
    let dy = (max_y - min_y) / height as f32;

    out.par_chunks_mut(width as usize).enumerate().for_each(|(i, row)| {
        let y = min_y + dy * i as f32;
        row.iter_mut().enumerate().for_each(|(j, col)| {
            let x = min_x + dx * j as f32;
            *col = mandelbrot_kernel(x, y, iters);
        });
    });
    out
}

