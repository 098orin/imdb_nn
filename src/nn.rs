pub trait Module {
    fn forward(&mut self, x: Vec<f32>) -> Vec<f32>;
    fn backward(&mut self, grad: Vec<f32>) -> Vec<f32>;
    fn step(&mut self, _lr: f32, _batch_size: usize) {}
}

pub struct Linear {
    w: Vec<f32>,
    b: Vec<f32>,

    grad_w: Vec<f32>,
    grad_b: Vec<f32>,

    in_size: usize,
    out_size: usize,

    last_input: Vec<f32>,
}

impl Linear {
    pub fn new(in_size: usize, out_size: usize) -> Self {
        Self {
            w: (0..in_size * out_size).map(|_| init_weight()).collect(),
            b: vec![0.0; out_size],
            grad_w: vec![0.0; in_size * out_size],
            grad_b: vec![0.0; out_size],
            in_size,
            out_size,
            last_input: Vec::new(),
        }
    }
}

impl Module for Linear {
    fn forward(&mut self, input: Vec<f32>) -> Vec<f32> {
        self.last_input = input.to_vec();

        let mut out = vec![0.0; self.out_size];
        for o in 0..self.out_size {
            let mut sum = self.b[o];
            for i in 0..self.in_size {
                sum += self.w[o * self.in_size + i] * input[i];
            }
            out[o] = sum;
        }
        out
    }

    fn backward(&mut self, grad: Vec<f32>) -> Vec<f32> {
        // 次に流す勾配（dL/dinput）
        let mut grad_input = vec![0.0; self.in_size];

        for o in 0..self.out_size {
            for i in 0..self.in_size {
                let idx = o * self.in_size + i;

                grad_input[i] += self.w[idx] * grad[o];
                self.grad_w[idx] += grad[o] * self.last_input[i];
            }
            self.grad_b[o] += grad[o];
        }

        grad_input
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

pub struct Model {
    layers: Vec<Box<dyn Module>>,
}

impl Model {
    pub fn new(layers: Vec<Box<dyn Module>>) -> Self {
        Self { layers }
    }

    pub fn forward(&mut self, input: &[f32]) -> Vec<f32> {
        let mut x = input.to_vec();
        for layer in &mut self.layers {
            x = layer.forward(x);
        }
        x
    }

    pub fn step(&mut self, lr: f32, batch_size: usize) {
        for layer in self.layers.iter_mut() {
            layer.step(lr, batch_size);
        }
    }

    pub fn train_batch(&mut self, batch_x: &[Vec<f32>], batch_y: &[usize], lr: f32) {
        let batch_size = batch_x.len();

        for (x, &y) in batch_x.iter().zip(batch_y.iter()) {
            let out = self.forward(x);
            let mut grad = out;
            grad[y] -= 1.0;

            for layer in self.layers.iter_mut().rev() {
                grad = layer.backward(grad);
            }
        }

        self.step(lr, batch_size);
    }
}

pub fn init_weight() -> f32 {
    // 簡易擬似乱数
    static mut SEED: u32 = 123456789;
    unsafe {
        SEED = SEED.wrapping_mul(1103515245).wrapping_add(12345);
        (SEED as f32 / u32::MAX as f32 - 0.5) * 0.1
    }
}
