//! softfloat-wrapper is a safe wrapper of [Berkeley SoftFloat](https://github.com/ucb-bar/berkeley-softfloat-3) based on [softfloat-sys](https://crates.io/crates/softfloat-sys).
//!
//! ## Examples
//!
//! ```
//! use softfloat_wrapper::{Float, F16, RoundingMode};
//!
//! fn main() {
//!     let a = 0x1234;
//!     let b = 0x1479;
//!
//!     let a = F16::from_bits(a);
//!     let b = F16::from_bits(b);
//!     let d = a.add(b, RoundingMode::TiesToEven);
//!
//!     let a = f32::from_bits(a.to_f32(RoundingMode::TiesToEven).bits());
//!     let b = f32::from_bits(b.to_f32(RoundingMode::TiesToEven).bits());
//!     let d = f32::from_bits(d.to_f32(RoundingMode::TiesToEven).bits());
//!
//!     println!("{} + {} = {}", a, b, d);
//! }
//! ```

mod f128;
mod f16;
mod f32;
mod f64;
pub use crate::f128::F128;
pub use crate::f16::F16;
pub use crate::f32::F32;
pub use crate::f64::F64;

use num_traits::{
    identities::{One, Zero},
    PrimInt,
};
use std::borrow::Borrow;
use std::cmp::Ordering;
use std::fmt::{LowerHex, UpperHex};

/// floating-point rounding mode defined by standard
#[derive(Copy, Clone, Debug)]
pub enum RoundingMode {
    /// to nearest, ties to even
    TiesToEven,
    /// toward 0
    TowardZero,
    /// toward −∞
    TowardNegative,
    /// toward +∞
    TowardPositive,
    /// to nearest, ties away from zero
    TiesToAway,
}

impl RoundingMode {
    fn set(&self) {
        unsafe {
            softfloat_sys::softfloat_roundingMode_write_helper(self.to_softfloat());
        }
    }

    fn to_softfloat(&self) -> u8 {
        match self {
            RoundingMode::TiesToEven => softfloat_sys::softfloat_round_near_even,
            RoundingMode::TowardZero => softfloat_sys::softfloat_round_minMag,
            RoundingMode::TowardNegative => softfloat_sys::softfloat_round_min,
            RoundingMode::TowardPositive => softfloat_sys::softfloat_round_max,
            RoundingMode::TiesToAway => softfloat_sys::softfloat_round_near_maxMag,
        }
    }
}

/// exception flags defined by standard
///
/// ## Examples
///
/// ```
/// use softfloat_wrapper::{ExceptionFlags, Float, RoundingMode, F16};
///
/// let a = 0x0;
/// let b = 0x0;
/// let a = F16::from_bits(a);
/// let b = F16::from_bits(b);
/// let mut flag = ExceptionFlags::default();
/// flag.set();
/// let _d = a.div(b, RoundingMode::TiesToEven);
/// flag.get();
/// assert!(flag.is_invalid());
/// ```
#[derive(Copy, Clone, Debug, Default)]
pub struct ExceptionFlags(u8);

impl ExceptionFlags {
    const FLAG_INEXACT: u8 = softfloat_sys::softfloat_flag_inexact;
    const FLAG_INFINITE: u8 = softfloat_sys::softfloat_flag_infinite;
    const FLAG_INVALID: u8 = softfloat_sys::softfloat_flag_invalid;
    const FLAG_OVERFLOW: u8 = softfloat_sys::softfloat_flag_overflow;
    const FLAG_UNDERFLOW: u8 = softfloat_sys::softfloat_flag_underflow;

    pub fn from_bits(x: u8) -> Self {
        Self(x)
    }

    pub fn bits(&self) -> u8 {
        self.0
    }

    pub fn is_inexact(&self) -> bool {
        self.0 & Self::FLAG_INEXACT != 0
    }

    pub fn is_infinite(&self) -> bool {
        self.0 & Self::FLAG_INFINITE != 0
    }

    pub fn is_invalid(&self) -> bool {
        self.0 & Self::FLAG_INVALID != 0
    }

    pub fn is_overflow(&self) -> bool {
        self.0 & Self::FLAG_OVERFLOW != 0
    }

    pub fn is_underflow(&self) -> bool {
        self.0 & Self::FLAG_UNDERFLOW != 0
    }

    pub fn set(&self) {
        unsafe {
            softfloat_sys::softfloat_exceptionFlags_write_helper(self.bits());
        }
    }

    pub fn get(&mut self) {
        let x = unsafe { softfloat_sys::softfloat_exceptionFlags_read_helper() };
        self.0 = x;
    }
}

/// arbitrary floting-point type
///
/// ## Examples
///
/// `Float` can be used for generic functions.
///
/// ```
/// use softfloat_wrapper::{Float, RoundingMode, F16, F32};
///
/// fn rsqrt<T: Float>(x: T) -> T {
///     let ret = x.sqrt(RoundingMode::TiesToEven);
///     let one = T::from_u8(1, RoundingMode::TiesToEven);
///     one.div(ret, RoundingMode::TiesToEven)
/// }
///
/// let a = F16::from_bits(0x1234);
/// let a = rsqrt(a);
/// let a = F32::from_bits(0x12345678);
/// let a = rsqrt(a);
/// ```
pub trait Float {
    type Payload: PrimInt + UpperHex + LowerHex;

    const EXPONENT_BIT: Self::Payload;
    const FRACTION_BIT: Self::Payload;
    const SIGN_POS: usize;
    const EXPONENT_POS: usize;

    fn set_payload(&mut self, x: Self::Payload);

    fn from_bits(v: Self::Payload) -> Self;

    fn bits(&self) -> Self::Payload;

    fn add<T: Borrow<Self>>(&self, x: T, rnd: RoundingMode) -> Self;

    fn sub<T: Borrow<Self>>(&self, x: T, rnd: RoundingMode) -> Self;

    fn mul<T: Borrow<Self>>(&self, x: T, rnd: RoundingMode) -> Self;

    fn fused_mul_add<T: Borrow<Self>>(&self, x: T, y: T, rnd: RoundingMode) -> Self;

    fn div<T: Borrow<Self>>(&self, x: T, rnd: RoundingMode) -> Self;

    fn rem<T: Borrow<Self>>(&self, x: T, rnd: RoundingMode) -> Self;

    fn sqrt(&self, rnd: RoundingMode) -> Self;

    fn compare<T: Borrow<Self>>(&self, x: T) -> Option<Ordering>;

    fn from_u32(x: u32, rnd: RoundingMode) -> Self;

    fn from_u64(x: u64, rnd: RoundingMode) -> Self;

    fn from_i32(x: i32, rnd: RoundingMode) -> Self;

    fn from_i64(x: i64, rnd: RoundingMode) -> Self;

    fn to_u32(&self, rnd: RoundingMode) -> u32;

    fn to_u64(&self, rnd: RoundingMode) -> u64;

    fn to_i32(&self, rnd: RoundingMode) -> i32;

    fn to_i64(&self, rnd: RoundingMode) -> i64;

    fn to_f16(&self, rnd: RoundingMode) -> F16;

    fn to_f32(&self, rnd: RoundingMode) -> F32;

    fn to_f64(&self, rnd: RoundingMode) -> F64;

    fn to_f128(&self, rnd: RoundingMode) -> F128;

    fn round_to_integral(&self, rnd: RoundingMode) -> Self;

    #[inline]
    fn from_u8(x: u8, rnd: RoundingMode) -> Self
    where
        Self: Sized,
    {
        Self::from_u32(x as u32, rnd)
    }

    #[inline]
    fn from_u16(x: u16, rnd: RoundingMode) -> Self
    where
        Self: Sized,
    {
        Self::from_u32(x as u32, rnd)
    }

    #[inline]
    fn from_i8(x: i8, rnd: RoundingMode) -> Self
    where
        Self: Sized,
    {
        Self::from_i32(x as i32, rnd)
    }

    #[inline]
    fn from_i16(x: i16, rnd: RoundingMode) -> Self
    where
        Self: Sized,
    {
        Self::from_i32(x as i32, rnd)
    }

    #[inline]
    fn neg(&self) -> Self
    where
        Self: Sized,
    {
        let mut ret = Self::from_bits(self.bits());
        ret.set_sign(!self.sign());
        ret
    }

    #[inline]
    fn abs(&self) -> Self
    where
        Self: Sized,
    {
        let mut ret = Self::from_bits(self.bits());
        ret.set_sign(Self::Payload::zero());
        ret
    }

    #[inline]
    fn sign(&self) -> Self::Payload {
        (self.bits() >> Self::SIGN_POS) & Self::Payload::one()
    }

    #[inline]
    fn exponent(&self) -> Self::Payload {
        (self.bits() >> Self::EXPONENT_POS) & Self::EXPONENT_BIT
    }

    #[inline]
    fn fraction(&self) -> Self::Payload {
        self.bits() & Self::FRACTION_BIT
    }

    #[inline]
    fn is_positive(&self) -> bool {
        self.sign() == Self::Payload::zero()
    }

    #[inline]
    fn is_positive_zero(&self) -> bool {
        self.is_positive()
            && self.exponent() == Self::Payload::zero()
            && self.fraction() == Self::Payload::zero()
    }

    #[inline]
    fn is_positive_subnormal(&self) -> bool {
        self.is_positive()
            && self.exponent() == Self::Payload::zero()
            && self.fraction() != Self::Payload::zero()
    }

    #[inline]
    fn is_positive_normal(&self) -> bool {
        self.is_positive()
            && self.exponent() != Self::Payload::zero()
            && self.exponent() != Self::EXPONENT_BIT
    }

    #[inline]
    fn is_positive_infinity(&self) -> bool {
        self.is_positive()
            && self.exponent() == Self::EXPONENT_BIT
            && self.fraction() == Self::Payload::zero()
    }

    #[inline]
    fn is_negative(&self) -> bool {
        self.sign() == Self::Payload::one()
    }

    #[inline]
    fn is_negative_zero(&self) -> bool {
        self.is_negative()
            && self.exponent() == Self::Payload::zero()
            && self.fraction() == Self::Payload::zero()
    }

    #[inline]
    fn is_negative_subnormal(&self) -> bool {
        self.is_negative()
            && self.exponent() == Self::Payload::zero()
            && self.fraction() != Self::Payload::zero()
    }

    #[inline]
    fn is_negative_normal(&self) -> bool {
        self.is_negative()
            && self.exponent() != Self::Payload::zero()
            && self.exponent() != Self::EXPONENT_BIT
    }

    #[inline]
    fn is_negative_infinity(&self) -> bool {
        self.is_negative()
            && self.exponent() == Self::EXPONENT_BIT
            && self.fraction() == Self::Payload::zero()
    }

    #[inline]
    fn is_nan(&self) -> bool {
        self.exponent() == Self::EXPONENT_BIT && self.fraction() != Self::Payload::zero()
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.is_positive_zero() || self.is_negative_zero()
    }

    #[inline]
    fn is_subnormal(&self) -> bool {
        self.exponent() == Self::Payload::zero()
    }

    #[inline]
    fn set_sign(&mut self, x: Self::Payload) {
        self.set_payload(
            (self.bits() & !(Self::Payload::one() << Self::SIGN_POS))
                | ((x & Self::Payload::one()) << Self::SIGN_POS),
        );
    }

    #[inline]
    fn set_exponent(&mut self, x: Self::Payload) {
        self.set_payload(
            (self.bits() & !(Self::EXPONENT_BIT << Self::EXPONENT_POS))
                | ((x & Self::EXPONENT_BIT) << Self::EXPONENT_POS),
        );
    }

    #[inline]
    fn set_fraction(&mut self, x: Self::Payload) {
        self.set_payload((self.bits() & !Self::FRACTION_BIT) | (x & Self::FRACTION_BIT));
    }

    #[inline]
    fn positive_infinity() -> Self
    where
        Self: Sized,
    {
        let mut x = Self::from_bits(Self::Payload::zero());
        x.set_exponent(Self::EXPONENT_BIT);
        x
    }

    #[inline]
    fn positive_zero() -> Self
    where
        Self: Sized,
    {
        let x = Self::from_bits(Self::Payload::zero());
        x
    }

    #[inline]
    fn negative_infinity() -> Self
    where
        Self: Sized,
    {
        let mut x = Self::from_bits(Self::Payload::zero());
        x.set_sign(Self::Payload::one());
        x.set_exponent(Self::EXPONENT_BIT);
        x
    }

    #[inline]
    fn negative_zero() -> Self
    where
        Self: Sized,
    {
        let mut x = Self::from_bits(Self::Payload::zero());
        x.set_sign(Self::Payload::one());
        x
    }

    #[inline]
    fn quiet_nan() -> Self
    where
        Self: Sized,
    {
        let mut x = Self::from_bits(Self::Payload::zero());
        x.set_exponent(Self::EXPONENT_BIT);
        x.set_fraction(Self::Payload::one() << (Self::EXPONENT_POS - 1));
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flag_inexact() {
        let a = 0x1234;
        let b = 0x7654;
        let a = F16::from_bits(a);
        let b = F16::from_bits(b);
        let mut flag = ExceptionFlags::default();
        flag.set();
        let _d = a.add(b, RoundingMode::TiesToEven);
        flag.get();
        assert!(flag.is_inexact());
        assert!(!flag.is_infinite());
        assert!(!flag.is_invalid());
        assert!(!flag.is_overflow());
        assert!(!flag.is_underflow());
    }

    #[test]
    fn flag_infinite() {
        let a = 0x1234;
        let b = 0x0;
        let a = F16::from_bits(a);
        let b = F16::from_bits(b);
        let mut flag = ExceptionFlags::default();
        flag.set();
        let _d = a.div(b, RoundingMode::TiesToEven);
        flag.get();
        assert!(!flag.is_inexact());
        assert!(flag.is_infinite());
        assert!(!flag.is_invalid());
        assert!(!flag.is_overflow());
        assert!(!flag.is_underflow());
    }

    #[test]
    fn flag_invalid() {
        let a = 0x0;
        let b = 0x0;
        let a = F16::from_bits(a);
        let b = F16::from_bits(b);
        let mut flag = ExceptionFlags::default();
        flag.set();
        let _d = a.div(b, RoundingMode::TiesToEven);
        flag.get();
        assert!(!flag.is_inexact());
        assert!(!flag.is_infinite());
        assert!(flag.is_invalid());
        assert!(!flag.is_overflow());
        assert!(!flag.is_underflow());
    }

    #[test]
    fn flag_overflow() {
        let a = 0x7bff;
        let b = 0x7bff;
        let a = F16::from_bits(a);
        let b = F16::from_bits(b);
        let mut flag = ExceptionFlags::default();
        flag.set();
        let _d = a.add(b, RoundingMode::TiesToEven);
        flag.get();
        assert!(flag.is_inexact());
        assert!(!flag.is_infinite());
        assert!(!flag.is_invalid());
        assert!(flag.is_overflow());
        assert!(!flag.is_underflow());
    }

    #[test]
    fn flag_underflow() {
        let a = 0x0001;
        let b = 0x0001;
        let a = F16::from_bits(a);
        let b = F16::from_bits(b);
        let mut flag = ExceptionFlags::default();
        flag.set();
        let _d = a.mul(b, RoundingMode::TiesToEven);
        flag.get();
        assert!(flag.is_inexact());
        assert!(!flag.is_infinite());
        assert!(!flag.is_invalid());
        assert!(!flag.is_overflow());
        assert!(flag.is_underflow());
    }
}
