use typeclass::hkt::HKT;

pub trait FlatMap<B>: HKT<B> {
    fn flat_map<F>(self, f: F) -> <Self as HKT<B>>::Target
        where
            F: Fn(<Self as HKT<B>>::Current) -> <Self as HKT<B>>::Target;
}

impl<A, B> FlatMap<B> for Option<A> {
    fn flat_map<F>(self, f: F) -> Self::Target
        where
            F: FnOnce(A) -> <Self as HKT<B>>::Target,
    {
        self.and_then(f)
    }
}

impl<A, B, E> FlatMap<B> for Result<A, E> {
    fn flat_map<F>(self, f: F) -> Self::Target
        where
            F: FnOnce(A) -> <Self as HKT<B>>::Target,
    {
        self.and_then(f)
    }
}