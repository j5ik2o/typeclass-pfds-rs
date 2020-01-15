use std::sync::Arc;
use typeclass::hkt::HKT;
use typeclass::empty::Empty;
use typeclass::semigroup::Semigroup;
use typeclass::pure::Pure;
use typeclass::functor::Functor;
use typeclass::flat_map::FlatMap;

#[derive(Debug)]
pub enum StackError {
    NoSuchElementError,
    IndexOutOfRange,
}

pub trait Stack<A: Clone> {
    fn cons(&self, value: A) -> Self;
    fn head(&self) -> Result<A, StackError>;
    fn tail(&self) -> Arc<Self>;
    fn filter<F>(&self, f: F) -> Self where F: Fn(A) -> bool;
    fn size(&self) -> usize;
    fn update(&self, index: u32, new_value: A) -> Result<Self, StackError> where Self: Sized;
    fn get(&self, i: u32) -> Result<A, StackError>;
    fn fold_left<B, F>(&self, z: B, f: F) -> B where F: Fn(B, A) -> B;
    fn fold_right<B: Clone, F>(&self, z: B, f: F) -> B where F: Fn(A, B) -> B;
    fn to_list(&self) -> Self;
}

#[derive(Debug, Clone)]
pub enum List<T> {
    Nil,
    Cons {
        head: T,
        tail: Arc<List<T>>,
    },
    Filtered {
        l: Arc<List<T>>,
        p: fn(T) -> bool,
    },
}

impl<T, U> HKT<U> for List<T> {
    type Current = T;
    type Target = List<U>;
}

//impl<A: Clone, B: Clone> Apply<B> for List<A> {
//    fn ap(self, f: Applicator<B, Self>) -> <Self as HKT<B>>::Target {
//        self.flat_map(|v| {
//
//        })
//    }
//}


impl<A: Clone, B: Clone> FlatMap<B> for List<A> {
    fn flat_map<F>(self, f: F) -> Self::Target
        where
            Self: Stack<A>,
            F: Fn(A) -> <Self as HKT<B>>::Target,
    {
        if self.is_empty() {
            List::Nil
        } else {
            let mut result: List<B> = List::empty();
            let mut cur: &List<A> = &self;
            while let List::Cons { ref head, ref tail } = *cur {
                result = result.combine(f(head.clone()));
                cur = tail
            }
            result
        }
    }
}

impl<T: Clone> Pure<T> for List<T> {
    fn of(c: Self::Current) -> Self::Target {
        List::empty().cons(c)
    }
}

impl<T> Empty for List<T> {
    fn empty() -> List<T> {
        List::Nil
    }
    fn is_empty(&self) -> bool {
        match *self {
            List::Nil => true,
            _ => false
        }
    }
}

impl<T: Clone> Semigroup for List<T> {
    fn combine(&self, ys: Self) -> Self where Self: Stack<T> {
        match *self {
            List::Nil => ys,
            List::Cons { head: ref h, tail: ref t } =>
                List::Cons{ head: h.clone(), tail: Arc::new(t.combine(ys))},
            List::Filtered { ref l, ref p } =>
                List::Filtered { l: Arc::new(l.combine(ys)), p: *p }
        }
    }
}

impl<A: Clone, B: Clone> Functor<B> for List<A> {
    fn map<F>(self, f: F) -> Self::Target
        where
        // A is Self::Current
            Self: Stack<A>,
            F: Fn(A) -> B,
    {
        if self.is_empty() {
            List::Nil
        } else {
            let mut result: List<B> = List::empty();
            let mut cur: &List<A> = &self;
            while let List::Cons { ref head, ref tail } = *cur {
                result = result.cons(f(head.clone()));
                cur = tail
            }
            result
        }
    }
}

impl<A: Clone> Stack<A> for List<A> {
    fn cons(&self, value: A) -> List<A> {
        List::Cons {
            head: value,
            tail: Arc::new(self.clone()),
        }
    }

    fn head(&self) -> Result<A, StackError> {
        match *self {
            List::Nil => Err(StackError::NoSuchElementError),
            List::Cons { head: ref value, .. } => Ok(value.clone()),
            List::Filtered { ref l, ref p } => l.filter(p).head()
        }
    }

    fn tail(&self) -> Arc<List<A>> {
        match *self {
            List::Nil => Arc::new(List::Nil),
            List::Cons { ref tail, .. } => tail.clone(),
            List::Filtered { ref l, ref p } => l.filter(p).tail()
        }
    }

    fn filter<F>(&self, f: F) -> Self where F: Fn(A) -> bool {
        self.fold_right(List::Nil, |h, t| {
            if f(h.clone()) {
                List::Cons { head: h.clone(), tail: Arc::from(t.clone()) }
            } else {
                t
            }
        })
    }

    fn size(&self) -> usize {
        match *self {
            List::Nil => 0,
            List::Cons { ref tail, .. } => 1 + tail.size(),
            List::Filtered { ref l, ref p } => l.filter(p).size()
        }
    }

    fn update(&self, index: u32, new_value: A) -> Result<List<A>, StackError> where Self: Sized {
        match *self {
            List::Nil => Err(StackError::IndexOutOfRange),
            List::Cons { head: ref value, ref tail } => match index {
                0 => Ok(tail.clone().cons(new_value)),
                _ => {
                    let updated_tail = tail.update(index - 1, new_value)?;
                    Ok(updated_tail.cons(value.clone()))
                }
            },
            List::Filtered { ref l, ref p } =>
                l.filter(|x| p(x)).update(index, new_value)
        }
    }

    fn get(&self, i: u32) -> Result<A, StackError> {
        match *self {
            List::Nil => Err(StackError::NoSuchElementError),
            List::Cons { head: ref value, ref tail } => match i {
                0 => Ok(value.clone()),
                _ => tail.get(i - 1)
            },
            List::Filtered { ref l, ref p } =>
                l.filter(p).get(i)
        }
    }

    fn fold_left<B, F>(&self, z: B, f: F) -> B where F: Fn(B, A) -> B {
        match *self {
            List::Nil => z,
            List::Cons { ref head, ref tail } =>
                tail.fold_left(f(z, head.clone()), f),
            List::Filtered { ref l, ref p } =>
                l.fold_left(z, |r, x| if p(x.clone()) { f(r, x.clone()) } else { r })
        }
    }

    /**
    error[E0308]: mismatched types
   --> src/data/stack.rs:212:30
    |
212 |                     Box::new(|b: B| g(f(a.clone(), b)))
    |                              ^^^^^^^^^^^^^^^^^^^^^^^^^ expected closure, found a different closure
    |
    = note: expected type `[closure@src/data/stack.rs:211:50: 211:58]`
               found type `[closure@src/data/stack.rs:212:30: 212:55 g:_, f:_, a:_]`
    = note: no two closures, even if identical, have the same type
    = help: consider boxing your closure and/or using it as a trait object
    */
    fn fold_right<B: Clone, F>(&self, z: B, f: F) -> B where F: Fn(A, B) -> B {
        match *self {
            List::Nil => z,
            List::Cons { .. } => {
                //   def foldRight[B](z: B)(f: (A, B) => B): B =
                //    foldLeft((b: B) => b)((g, a) => b => g(f(a, b)))(z)
                let ffn= self.fold_left(Box::new(|b: B| b), |g, a: A| {
                    Box::new(|b: B| g(f(a.clone(), b)))
                });
                ffn(z)
            },
            List::Filtered { ref l, ref p } =>
                l.fold_right(z, |r, x| if p(r.clone()) { f(r.clone(), x) } else { x })
        }
    }

    fn to_list(&self) -> Self {
        match *self {
            List::Nil => List::Nil,
            List::Filtered { ref l, ref p } => l.filter(|x| p(x)),
            _ => self.clone()
        }
    }
}

#[cfg(test)]
fn suffixes<T: Clone>(stack: &Arc<List<T>>) -> List<Arc<List<T>>> {
    let tail_suffixes = match **stack {
        List::Nil => List::empty(),
        List::Cons { ref tail, .. } => suffixes(&tail),
        List::Filtered { ref l, ref p } => suffixes(&Arc::new(l.filter(|x| p(x))))
    };

    return tail_suffixes.cons(stack.clone());
}

#[cfg(test)]
mod tests {
    use data::stack::{List, Stack, StackError, suffixes};
    use typeclass::empty::Empty;
    use std::sync::Arc;
    use typeclass::semigroup::Semigroup;
    use typeclass::functor::Functor;
    use typeclass::flat_map::FlatMap;

    #[test]
    fn empty_is_empty() {
        let stack: List<()> = List::empty();

        assert!(stack.is_empty());
        assert!(stack.size() == 0);
    }

    #[test]
    fn cons_is_not_empty() {
        let stack: List<i32> = List::empty().cons(4);

        assert!(!stack.is_empty());
        assert!(stack.size() == 1);
    }

    #[test]
    fn head_empty_error() {
        let stack: List<()> = List::empty();

        assert!(stack.head().is_err());
    }

    #[test]
    fn head_last_item() {
        let stack: List<i32> = List::empty().cons(5).cons(6);
        let head: Result<i32, StackError> = stack.head();

        assert!(head.is_ok());
        assert!(head.unwrap() == 6);
    }

    #[test]
    fn tail_empty_is_error() {
        // let stack: List<()> = List::empty();

        // assert_eq!( Arc::try_unwrap(stack.tail()).unwrap(), List::Nil as List<()>);
    }


    #[test]
    fn head_after_tail() {
        let stack: List<i32> = List::empty().cons(1).cons(2).cons(3);
        let tailtail = stack.tail().tail();

        assert!(tailtail.head().unwrap() == 1);
    }

    #[test]
    fn size_multiple_items() {
        let stack: List<i32> = List::empty().cons(1).cons(2).cons(3);

        assert!(stack.size() == 3);
    }

    #[test]
    fn get_valid() {
        let stack: List<i32> = List::empty().cons(1).cons(2).cons(3);

        assert!(stack.get(1).unwrap() == 2);
    }

    #[test]
    fn get_out_of_range() {
        let stack: List<i32> = List::empty().cons(1).cons(2).cons(3);

        assert!(stack.get(3).is_err());
    }

    #[test]
    fn cloneable() {
        let stack: List<i32> = List::empty().cons(1).cons(2).cons(3);
        let stack2 = stack.clone();

        let tailtail = stack.tail().tail();
        let tail = stack2.tail();

        assert!(tailtail.head().unwrap() == 1);
        assert!(tail.head().unwrap() == 2);
    }

    #[test]
    fn update_valid() {
        let stack: List<i32> = List::empty().cons(1).cons(2).cons(3);
        let updated = stack.clone().update(1, 10).unwrap();

        assert!(updated.size() == 3);
        assert!(updated.get(0).unwrap() == 3);
        assert!(updated.get(1).unwrap() == 10);
        assert!(updated.get(2).unwrap() == 1);

        // And stack is unchanged (I think the typesystem ensures this?)
        assert!(stack.size() == 3);
        assert!(stack.get(0).unwrap() == 3);
        assert!(stack.get(1).unwrap() == 2);
        assert!(stack.get(2).unwrap() == 1);
    }

    #[test]
    fn update_invalid() {
        let stack: List<i32> = List::empty().cons(1).cons(2).cons(3);
        let updated = stack.clone().update(4, 10);

        assert!(updated.is_err());
    }

    #[test]
    fn suffixes_empty() {
        let stack: Arc<List<()>> = Arc::new(List::empty());
        let suffixes = suffixes(&stack);

        assert!(suffixes.size() == 1);
        assert!(suffixes.get(0).unwrap().is_empty());
    }

    #[test]
    fn suffixes_nonempty() {
        let stack: Arc<List<i32>> = Arc::new(List::empty().cons(1).cons(2));
        let suffixes = suffixes(&stack);

        assert!(suffixes.size() == 3);

        let suffix1 = suffixes.get(0).unwrap();
        assert!(suffix1.size() == 2);
        assert!(suffix1.get(0).unwrap() == 2);
        assert!(suffix1.get(1).unwrap() == 1);

        let suffix2 = suffixes.get(1).unwrap();
        assert!(suffix2.size() == 1);
        assert!(suffix2.get(0).unwrap() == 1);

        let suffix3 = suffixes.get(2).unwrap();
        assert!(suffix3.is_empty());
    }

    #[test]
    fn combine() {
        let l1: List<i64> = List::empty();
        assert_eq!(l1.size(), 0);
        let l2: List<i64> = l1.cons(1);
        assert_eq!(l2.size(), 1);
        let l3: List<i64> = l1.cons(3).cons(2);
        let l4: List<i64> = l2.combine(l3);
        let l5: List<i64> = l4.map(|x| x * 2);
        let l6: List<i64> = l5.flat_map(|_| List::Nil);
        println!("{:?}", l2);
        println!("{:?}", l6);
    }
}
