use crate::nn::Module;

pub struct Softmax;

impl Module for Softmax {
    fn forward(&mut self, x: Vec<f32>) -> Vec<f32> {
        let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;

        let mut y = x;

        for v in y.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }

        for v in y.iter_mut() {
            *v /= sum;
        }
        y
    }

    fn backward(&mut self, grad: Vec<f32>) -> Vec<f32> {
        // Softmax + CrossEntropy 前提
        grad
    }
}
