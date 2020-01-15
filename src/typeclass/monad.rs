use typeclass::flat_map::FlatMap;
use typeclass::applicative::Applicative;

pub trait Monad<A, B>: FlatMap<B> + Applicative<A, B> {}

impl<A, B> Monad<A, B> for Option<A> {}

impl<A, B, E> Monad<A, B> for Result<A, E> {}
