use crate::nn::Loss;

pub struct CrossEntropyLoss;

impl Loss for CrossEntropyLoss {
    fn forward(&mut self, logits: &[f32], target: usize, loss: &mut f32) {
        let mut max = logits[0];
        for &v in logits.iter().skip(1) {
            if v > max {
                max = v;
            }
        }

        let mut sum = 0.0;
        for &v in logits {
            sum += (v - max).exp();
        }

        *loss = sum.ln() + max - logits[target];
    }

    fn backward(&mut self, logits: &[f32], target: usize, grad: &mut [f32]) {
        let mut max = logits[0];
        for &v in logits.iter().skip(1) {
            if v > max {
                max = v;
            }
        }

        let mut sum = 0.0;
        for &v in logits {
            sum += (v - max).exp();
        }

        for i in 0..logits.len() {
            grad[i] = (logits[i] - max).exp() / sum;
        }
        grad[target] -= 1.0;
    }
}
