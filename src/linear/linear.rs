use crate::nn::Module;

pub struct Linear {
    w: Vec<f32>,
    b: Vec<f32>,

    grad_w: Vec<f32>,
    grad_b: Vec<f32>,

    in_size: usize,
    out_size: usize,
}

impl Linear {
    pub fn new(in_size: usize, out_size: usize) -> Self {
        Self {
            w: (0..in_size * out_size).map(|_| super::init_weight()).collect(),
            b: vec![0.0; out_size],
            grad_w: vec![0.0; in_size * out_size],
            grad_b: vec![0.0; out_size],
            in_size,
            out_size,
        }
    }
}

impl Module for Linear {
    fn forward(&mut self, input: &[f32], output: &mut [f32]) {
        for o in 0..self.out_size {
            let mut sum = self.b[o];
            for i in 0..self.in_size {
                sum += self.w[o * self.in_size + i] * input[i];
            }
            output[o] = sum;
        }
    }

    fn backward(&mut self, grad_output: &[f32], input: &[f32], grad_input: &mut [f32]) {
        grad_input.fill(0.0);

        for o in 0..self.out_size {
            for i in 0..self.in_size {
                let idx = o * self.in_size + i;
                self.grad_w[idx] += grad_output[o] * input[i];
                grad_input[i] += self.w[idx] * grad_output[o];
            }
            self.grad_b[o] += grad_output[o];
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

    fn output_size(&self) -> usize {
        self.out_size
    }
}
