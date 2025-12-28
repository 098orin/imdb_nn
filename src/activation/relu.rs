use crate::nn::Module;

pub struct ReLU {
    size: usize,
}

impl ReLU {
    pub fn new(size: usize) -> Self {
        Self { size }
    }
}

impl Module for ReLU {
    fn forward(&mut self, input: &[f32], output: &mut [f32]) {
        for i in 0..self.size {
            output[i] = input[i].max(0.0);
        }
    }

    fn backward(&mut self, grad_output: &[f32], input: &[f32], grad_input: &mut [f32]) {
        for i in 0..self.size {
            grad_input[i] = if input[i] > 0.0 { grad_output[i] } else { 0.0 };
        }
    }

    fn step(&mut self, _lr: f32, _batch_size: usize) {}

    fn output_size(&self) -> usize {
        self.size
    }
}
