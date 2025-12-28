use crate::nn::{Dense, Module, SparseVec};

pub struct SparseLinear {
    w: Vec<f32>,
    b: Vec<f32>,
    grad_w: Vec<f32>,
    grad_b: Vec<f32>,
    in_size: usize,
    out_size: usize,
}

impl SparseLinear {
    pub fn new(in_size: usize, out_size: usize) -> Self {
        Self {
            w: (0..in_size * out_size)
                .map(|_| super::init_weight())
                .collect(),
            b: vec![0.0; out_size],
            grad_w: vec![0.0; in_size * out_size],
            grad_b: vec![0.0; out_size],
            in_size,
            out_size,
        }
    }
}

impl Module for SparseLinear {
    type Input = SparseVec;
    type Output = Dense;

    fn forward(&mut self, input: &SparseVec, output: &mut Dense) {
        output.fill(0.0);

        for o in 0..self.out_size {
            let mut sum = self.b[o];
            for &(i, v) in input {
                sum += self.w[o * self.in_size + i] * v;
            }
            output[o] = sum;
        }
    }

    fn backward(&mut self, grad_output: &Dense, input: &SparseVec, grad_input: &mut SparseVec) {
        grad_input.clear(); // 入力勾配は不要

        for o in 0..self.out_size {
            let go = grad_output[o];
            self.grad_b[o] += go;

            for &(i, v) in input {
                let idx = o * self.in_size + i;
                self.grad_w[idx] += go * v;
            }
        }
    }

    fn step(&mut self, lr: f32, batch_size: usize) {
        let scale = lr / batch_size as f32;

        for i in 0..self.w.len() {
            self.w[i] -= scale * self.grad_w[i];
            self.grad_w[i] = 0.0;
        }
        for i in 0..self.b.len() {
            self.b[i] -= scale * self.grad_b[i];
            self.grad_b[i] = 0.0;
        }
    }
}
