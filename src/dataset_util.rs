pub struct BatchIter<I> {
    iter: I,
    batch_size: usize,
    index: usize,
    max_size: Option<usize>,
}

impl<I> BatchIter<I>
where
    I: Iterator,
{
    pub fn new(iter: I, batch_size: usize) -> Self {
        Self {
            iter,
            batch_size,
            index: 0,
            max_size: None,
        }
    }

    pub fn set_max_size(mut self, max_size: usize) -> Self {
        self.max_size = Some(max_size);
        self
    }

    pub fn take_skipping(&mut self, u: usize) {
        let _ = self.iter.by_ref().take(u);
    }
}

impl<I> Iterator for BatchIter<I>
where
    I: Iterator,
{
    type Item = Vec<I::Item>;

    fn next(&mut self) -> Option<Self::Item> {
        self.index += self.batch_size;
        if let Some(max_size) = self.max_size
            && max_size + self.batch_size <= self.index
        {
            return None;
        }
        let batch: Vec<_> = self.iter.by_ref().take(self.batch_size).collect();
        if batch.is_empty() { None } else { Some(batch) }
    }
}
