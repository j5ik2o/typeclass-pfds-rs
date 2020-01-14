use std::sync::Arc;
use typeclass::hkt::HKT;
use typeclass::empty::Empty;
use typeclass::semigroup::Semigroup;
use typeclass::pure::Pure;
use typeclass::functor::Functor;

#[derive(Debug)]
pub enum StackError {
    NoSuchElementError,
    IndexOutOfRange,
}

pub trait Stack<T: Clone> {
    fn cons(&self, value: T) -> Self;
    fn head(&self) -> Result<T, StackError>;
    fn tail(&self) -> Arc<Self>;
    fn size(&self) -> usize;
    fn update(&self, index: u32, new_value: T) -> Result<Self, StackError> where Self: Sized;
    fn get(&self, i: u32) -> Result<T, StackError>;
}

#[derive(Debug, Clone)]
pub enum List<T> {
    Nil,
    Cons {
        value: T,
        tail: Arc<List<T>>,
    },
}

impl<T, U> HKT<U> for List<T> {
    type Current = T;
    type Target = List<U>;
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
        if self.is_empty() {
            ys
        } else {
            let x = self.head().unwrap();
            self.tail().combine(ys).cons(x)
        }
    }
}

impl<A : Clone, B: Clone> Functor<B> for List<A> {
    fn fmap<F>(self, f: F) -> Self::Target
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
            while let List::Cons{ ref value, ref tail} = *cur {
                result = result.cons(f(value.clone()));
                cur = tail
            }
            result
        }
    }
}

impl<T: Clone> Stack<T> for List<T> {
    fn cons(&self, value: T) -> List<T> {
        List::Cons {
            value: value,
            tail: Arc::new(self.clone()),
        }
    }

    fn head(&self) -> Result<T, StackError> {
        match *self {
            List::Nil => Err(StackError::NoSuchElementError),
            List::Cons { ref value, .. } => Ok(value.clone())
        }
    }

    fn tail(&self) -> Arc<List<T>> {
        match *self {
            List::Nil => Arc::new(List::Nil),
            List::Cons { ref tail, .. } => tail.clone(),
        }
    }

    fn size(&self) -> usize {
        match *self {
            List::Nil => 0,
            List::Cons { ref tail, .. } => 1 + tail.size()
        }
    }


    fn update(&self, index: u32, new_value: T) -> Result<List<T>, StackError> where Self: Sized {
        match *self {
            List::Nil => Err(StackError::IndexOutOfRange),
            List::Cons { ref value, ref tail } => match index {
                0 => Ok(tail.clone().cons(new_value)),
                _ => {
                    let updated_tail = tail.update(index - 1, new_value)?;
                    Ok(updated_tail.cons(value.clone()))
                }
            }
        }
    }

    fn get(&self, i: u32) -> Result<T, StackError> {
        match *self {
            List::Nil => Err(StackError::NoSuchElementError),
            List::Cons { ref value, ref tail } => match i {
                0 => Ok(value.clone()),
                _ => tail.get(i - 1)
            }
        }
    }
}

#[cfg(test)]
fn suffixes<T: Clone>(stack: &Arc<List<T>>) -> List<Arc<List<T>>> {
    let tail_suffixes = match **stack {
        List::Nil => List::empty(),
        List::Cons { ref tail, .. } => suffixes(&tail),
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
        let l5: List<i64> = l4.fmap(|x| x * 2);
        println!("{:?}", l2);
        println!("{:?}", l5);
    }
}
