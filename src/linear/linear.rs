use crate::nn::{Dense, Module};

pub struct Linear {
    w: Dense,
    b: Dense,
    grad_w: Dense,
    grad_b: Dense,
    in_size: usize,
    out_size: usize,
}

impl Linear {
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

impl Module for Linear {
    type Input = Dense;
    type Output = Dense;

    fn new_output(&self, input: &Dense) -> Dense {
        debug_assert_eq!(input.len(), self.in_size);
        vec![0.0; self.out_size]
    }

    fn forward(&mut self, input: &Dense, output: &mut Dense) {
        for o in 0..self.out_size {
            let mut sum = self.b[o];
            for i in 0..self.in_size {
                sum += self.w[o * self.in_size + i] * input[i];
            }
            output[o] = sum;
        }
    }

    fn backward(&mut self, grad_output: &Dense, input: &Dense, grad_input: &mut Dense) {
        grad_input.fill(0.0);

        for o in 0..self.out_size {
            let go = grad_output[o];
            self.grad_b[o] += go;

            for i in 0..self.in_size {
                let idx = o * self.in_size + i;
                self.grad_w[idx] += go * input[i];
                grad_input[i] += self.w[idx] * go;
            }
        }
    }

    fn step(&mut self, lr: f32, batch: usize) {
        let s = lr / batch as f32;
        for i in 0..self.w.len() {
            self.w[i] -= s * self.grad_w[i];
            self.grad_w[i] = 0.0;
        }
        for i in 0..self.b.len() {
            self.b[i] -= s * self.grad_b[i];
            self.grad_b[i] = 0.0;
        }
    }
}
