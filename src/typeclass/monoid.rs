use typeclass::empty::Empty;
use typeclass::semigroup::Semigroup;

pub trait Monoid: Empty + Semigroup {}

impl Monoid for i32 {}
impl Monoid for i64 {}
impl<T: Clone> Monoid for Vec<T> {}
impl Monoid for String {}