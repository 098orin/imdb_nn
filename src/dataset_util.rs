pub struct BatchIter<I> {
    iter: I,
    batch_size: usize,
}

impl<I> BatchIter<I>
where
    I: Iterator,
{
    pub fn new(iter: I, batch_size: usize) -> Self {
        Self {
            iter,
            batch_size,
        }
    }
}

impl<I> Iterator for BatchIter<I>
where
    I: Iterator,
{
    type Item = Vec<I::Item>;

    fn next(&mut self) -> Option<Self::Item> {
        let batch: Vec<_> = self.iter.by_ref().take(self.batch_size).collect();
        if batch.is_empty() { None } else { Some(batch) }
    }
}
