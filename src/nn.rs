pub trait Module {
    fn forward(&mut self, input: &[f32], output: &mut [f32]);
    fn backward(&mut self, grad_output: &[f32], input: &[f32], grad_input: &mut [f32]);
    fn step(&mut self, lr: f32, batch_size: usize);
    fn output_size(&self) -> usize;
}

#[allow(dead_code)]
pub trait Loss {
    /// 損失値を計算（評価用）
    fn forward(&mut self, prediction: &[f32], target: usize, loss: &mut f32);

    /// ∂L/∂prediction を計算（逆伝播の起点）
    fn backward(&mut self, prediction: &[f32], target: usize, grad_prediction: &mut [f32]);
}

pub struct Model {
    layers: Vec<Box<dyn Module>>,
}

pub struct ModelBuffers {
    pub activations: Vec<Vec<f32>>,
    pub gradients: Vec<Vec<f32>>,
}

impl Model {
    pub fn new(layers: Vec<Box<dyn Module>>) -> Self {
        Self { layers }
    }

    pub fn forward(&mut self, input: &[f32], buffers: &mut [Vec<f32>]) {
        buffers[0].copy_from_slice(input);

        for i in 0..self.layers.len() {
            let (inp, out) = buffers.split_at_mut(i + 1);
            self.layers[i].forward(&inp[i], &mut out[0]);
        }
    }

    pub fn train_batch(
        &mut self,
        batch_x: &[Vec<f32>],
        batch_y: &[usize],
        loss: &mut dyn Loss,
        lr: f32,
    ) {
        let batch_size = batch_x.len();

        let mut buffers = self.create_buffers(batch_x[0].len());

        for (x, &y) in batch_x.iter().zip(batch_y.iter()) {
            // forward
            self.forward(x, &mut buffers.activations);

            // loss backward
            loss.backward(
                buffers.activations.last().unwrap(),
                y,
                buffers.gradients.last_mut().unwrap(),
            );

            // backward
            self.backward(&buffers.activations, &mut buffers.gradients);
        }

        self.step(lr, batch_size);
    }

    pub fn backward(&mut self, buffers: &[Vec<f32>], grad_buffers: &mut [Vec<f32>]) {
        for i in (0..self.layers.len()).rev() {
            let (left, right) = grad_buffers.split_at_mut(i + 1);
            let grad_input = &mut left[i];
            let grad_output = &right[0];

            self.layers[i].backward(grad_output, &buffers[i], grad_input);
        }
    }

    pub fn step(&mut self, lr: f32, batch_size: usize) {
        for layer in &mut self.layers {
            layer.step(lr, batch_size);
        }
    }

    pub fn create_buffers(&self, input_size: usize) -> ModelBuffers {
        let mut activations = Vec::with_capacity(self.layers.len() + 1);
        let mut gradients = Vec::with_capacity(self.layers.len() + 1);

        // input
        activations.push(vec![0.0; input_size]);
        gradients.push(vec![0.0; input_size]);

        // 各 layer の出力
        for layer in &self.layers {
            let size = layer.output_size();
            activations.push(vec![0.0; size]);
            gradients.push(vec![0.0; size]);
        }

        ModelBuffers {
            activations,
            gradients,
        }
    }
}
