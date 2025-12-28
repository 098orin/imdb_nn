#[macro_export]
macro_rules! chain {
    ($last:expr $(,)?) => {
        $crate::nn::Chain {
            head: $last,
            tail: $crate::nn::End::<crate::nn::Dense>::new(),
        }
    };

    ($head:expr, $($rest:expr),+ $(,)?) => {
        $crate::nn::Chain {
            head: $head,
            tail: chain!($($rest),+),
        }
    };
}

pub type Dense = Vec<f32>;
pub type SparseVec = Vec<(usize, f32)>; // (index, value)

pub trait Module {
    type Input: Buffer;
    type Output: Buffer;

    fn forward(&mut self, input: &Self::Input, output: &mut Self::Output);
    fn backward(
        &mut self,
        grad_output: &Self::Output,
        input: &Self::Input,
        grad_input: &mut Self::Input,
    );
    fn step(&mut self, lr: f32, batch_size: usize);

    fn new_output(&self, input: &Self::Input) -> Self::Output;
}

pub trait Buffer: Sized + Clone {
    fn zeros_like(&self) -> Self;
}

impl Buffer for Dense {
    fn zeros_like(&self) -> Self {
        vec![0.0; self.len()]
    }
}

impl Buffer for SparseVec {
    fn zeros_like(&self) -> Self {
        Vec::new()
    }
}

pub struct End<T>(std::marker::PhantomData<T>);

impl<T> End<T> {
    pub fn new() -> Self {
        End(std::marker::PhantomData)
    }
}

impl<T: Buffer> Module for End<T> {
    type Input = T;
    type Output = T;

    fn new_output(&self, input: &T) -> T {
        input.clone()
    }

    fn forward(&mut self, input: &T, output: &mut T) {
        *output = input.clone();
    }

    fn backward(&mut self, grad_output: &T, _input: &T, grad_input: &mut T) {
        *grad_input = grad_output.clone();
    }

    fn step(&mut self, _: f32, _: usize) {}
}

pub struct Chain<M, N> {
    pub head: M,
    pub tail: N,
}

impl<M, N> Module for Chain<M, N>
where
    M: Module,
    N: Module<Input = M::Output>,
{
    type Input = M::Input;
    type Output = N::Output;

    fn new_output(&self, input: &Self::Input) -> Self::Output {
        let mid = self.head.new_output(input);
        self.tail.new_output(&mid)
    }

    fn forward(&mut self, input: &Self::Input, output: &mut Self::Output) {
        let mut mid = self.head.new_output(input);
        self.head.forward(input, &mut mid);
        self.tail.forward(&mid, output);
    }

    fn backward(
        &mut self,
        grad_output: &Self::Output,
        input: &Self::Input,
        grad_input: &mut Self::Input,
    ) {
        let mut mid = self.head.new_output(input);
        let mut grad_mid = mid.zeros_like();

        self.head.forward(input, &mut mid);
        self.tail.backward(grad_output, &mid, &mut grad_mid);
        self.head.backward(&grad_mid, input, grad_input);
    }

    fn step(&mut self, lr: f32, batch: usize) {
        self.head.step(lr, batch);
        self.tail.step(lr, batch);
    }
}

impl<M, N> Chain<M, N>
where
    Self: Module<Output = Dense>,
{
    pub fn train_batch<L: Loss>(
        &mut self,
        xs: &[<Self as Module>::Input],
        ys: &[usize],
        loss: &mut L,
        lr: f32,
    ) {
        let batch_size = xs.len();

        for (x, &y) in xs.iter().zip(ys.iter()) {
            // forward
            let mut y_pred = self.new_output(x);
            self.forward(x, &mut y_pred);

            // loss backward
            let mut grad_y = y_pred.zeros_like();
            loss.backward(&y_pred, y, &mut grad_y);

            // backward
            let mut grad_x = x.zeros_like();
            self.backward(&grad_y, x, &mut grad_x);
        }

        self.step(lr, batch_size);
    }
}

#[allow(dead_code)]
pub trait Loss {
    /// 損失値を計算（評価用）
    fn forward(&mut self, prediction: &[f32], target: usize, loss: &mut f32);

    /// ∂L/∂prediction を計算（逆伝播の起点）
    fn backward(&mut self, prediction: &[f32], target: usize, grad_prediction: &mut [f32]);
}
