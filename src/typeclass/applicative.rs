use typeclass::pure::Pure;
use typeclass::apply::Apply;

pub trait Applicative<A, B>: Apply<B> + Pure<A> {}

impl<A, B> Applicative<A, B> for Option<A> {}

impl<A, B, E> Applicative<A, B> for Result<A, E> {}