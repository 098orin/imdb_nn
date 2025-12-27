use crate::nn::Module;

pub struct ReLU {
    last_output: Vec<f32>,
}

impl ReLU {
    pub fn new() -> Self {
        Self {
            last_output: Vec::new(),
        }
    }
}

impl Module for ReLU {
    fn forward(&mut self, x: Vec<f32>) -> Vec<f32> {
        let y: Vec<f32> = x.iter().map(|v| v.max(0.0)).collect();
        self.last_output = y.clone();
        y
    }

    fn backward(&mut self, mut grad: Vec<f32>) -> Vec<f32> {
        for (g, &out) in grad.iter_mut().zip(self.last_output.iter()) {
            if out <= 0.0 {
                *g = 0.0;
            }
        }
        grad
    }
}
