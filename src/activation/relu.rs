use crate::nn::{Dense, Module};

pub struct ReLU;

impl Module for ReLU {
    type Input = Dense;
    type Output = Dense;

    fn forward(&mut self, input: &Dense, output: &mut Dense) {
        for i in 0..input.len() {
            output[i] = input[i].max(0.0);
        }
    }

    fn backward(&mut self, grad_output: &Dense, input: &Dense, grad_input: &mut Dense) {
        for i in 0..input.len() {
            grad_input[i] = if input[i] > 0.0 { grad_output[i] } else { 0.0 };
        }
    }

    fn step(&mut self, _: f32, _: usize) {}
}
