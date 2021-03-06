pub trait Empty {
    fn empty() -> Self;
    fn is_empty(&self) -> bool;
}

macro_rules! numeric_empty_impl {
    ($($t:ty)*) => ($(
        impl Empty for $t {
            fn empty() -> Self {
                0
            }
            fn is_empty(&self) -> bool {
                match *self {
                  0 => true,
                  _ => false
                }
            }
        }
    )*)
}

numeric_empty_impl! { usize u8 u16 u32 u64 u128 isize i8 i16 i32 i64 i128 }

macro_rules! floating_numeric_empty_impl {
    ($($t:ty)*) => ($(
        impl Empty for $t {
            fn empty() -> Self {
                0.0
            }
            #[allow(illegal_floating_point_literal_pattern)]
            fn is_empty(&self) -> bool {
                match *self {
                  0.0 => true,
                  _ => false
                }
            }
        }
    )*)
}

floating_numeric_empty_impl! { f32 f64 }

impl<T> Empty for Vec<T> {
    fn empty() -> Vec<T> {
        vec![]
    }
    fn is_empty(&self) -> bool {
        self.is_empty()
    }
}

impl Empty for String {
    fn empty() -> String {
        "".to_string()
    }

    fn is_empty(&self) -> bool {
        self.is_empty()
    }
}