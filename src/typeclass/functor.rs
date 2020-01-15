use typeclass::hkt::HKT;

pub trait Functor<B>: HKT<B> {
    fn map<F>(self, f: F) -> Self::Target
        where
            F: Fn(Self::Current) -> B;
}

impl<A, B> Functor<B> for Option<A> {
    fn map<F>(self, f: F) -> Self::Target
        where
        // A is Self::Current
            F: FnOnce(A) -> B,
    {
        self.map(f)
    }
}

impl<A, B, E> Functor<B> for Result<A, E> {
    fn map<F>(self, f: F) -> Self::Target
        where
        // A is Self::Current
            F: FnOnce(A) -> B,
    {
        self.map(f)
    }
}