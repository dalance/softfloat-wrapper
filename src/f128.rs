use crate::{Float, RoundingMode, F16, F32, F64};
use softfloat_sys::float128_t;
use std::borrow::Borrow;

/// standard 128-bit float
#[derive(Copy, Clone, Debug)]
pub struct F128(float128_t);

impl F128 {
    /// Converts primitive `f32` to `F128`
    pub fn from_f32(v: f32) -> Self {
        F32::from_bits(v.to_bits()).to_f128(RoundingMode::TiesToEven)
    }

    /// Converts primitive `f64` to `F128`
    pub fn from_f64(v: f64) -> Self {
        F64::from_bits(v.to_bits()).to_f128(RoundingMode::TiesToEven)
    }
}

impl Float for F128 {
    type Payload = u128;

    const EXPONENT_BIT: Self::Payload = 0x7fff;
    const FRACTION_BIT: Self::Payload = 0xffff_ffff_ffff_ffff_ffff_ffff_ffff;
    const SIGN_POS: usize = 127;
    const EXPONENT_POS: usize = 112;

    #[inline]
    fn set_payload(&mut self, x: Self::Payload) {
        let x = [x as u64, (x >> 64) as u64];
        self.0.v = x;
    }

    #[inline]
    fn from_bits(v: Self::Payload) -> Self {
        let v = [v as u64, (v >> 64) as u64];
        Self(float128_t { v })
    }

    #[inline]
    fn to_bits(&self) -> Self::Payload {
        let mut ret = 0u128;
        ret |= self.0.v[0] as u128;
        ret |= (self.0.v[1] as u128) << 64;
        ret
    }

    #[inline]
    fn bits(&self) -> Self::Payload {
        self.to_bits()
    }

    fn add<T: Borrow<Self>>(&self, x: T, rnd: RoundingMode) -> Self {
        rnd.set();
        let ret = unsafe { softfloat_sys::f128_add(self.0, x.borrow().0) };
        Self(ret)
    }

    fn sub<T: Borrow<Self>>(&self, x: T, rnd: RoundingMode) -> Self {
        rnd.set();
        let ret = unsafe { softfloat_sys::f128_sub(self.0, x.borrow().0) };
        Self(ret)
    }

    fn mul<T: Borrow<Self>>(&self, x: T, rnd: RoundingMode) -> Self {
        rnd.set();
        let ret = unsafe { softfloat_sys::f128_mul(self.0, x.borrow().0) };
        Self(ret)
    }

    fn fused_mul_add<T: Borrow<Self>>(&self, x: T, y: T, rnd: RoundingMode) -> Self {
        rnd.set();
        let ret = unsafe { softfloat_sys::f128_mulAdd(self.0, x.borrow().0, y.borrow().0) };
        Self(ret)
    }

    fn div<T: Borrow<Self>>(&self, x: T, rnd: RoundingMode) -> Self {
        rnd.set();
        let ret = unsafe { softfloat_sys::f128_div(self.0, x.borrow().0) };
        Self(ret)
    }

    fn rem<T: Borrow<Self>>(&self, x: T, rnd: RoundingMode) -> Self {
        rnd.set();
        let ret = unsafe { softfloat_sys::f128_rem(self.0, x.borrow().0) };
        Self(ret)
    }

    fn sqrt(&self, rnd: RoundingMode) -> Self {
        rnd.set();
        let ret = unsafe { softfloat_sys::f128_sqrt(self.0) };
        Self(ret)
    }

    fn eq<T: Borrow<Self>>(&self, x: T) -> bool {
        unsafe { softfloat_sys::f128_eq(self.0, x.borrow().0) }
    }

    fn lt<T: Borrow<Self>>(&self, x: T) -> bool {
        unsafe { softfloat_sys::f128_lt(self.0, x.borrow().0) }
    }

    fn le<T: Borrow<Self>>(&self, x: T) -> bool {
        unsafe { softfloat_sys::f128_le(self.0, x.borrow().0) }
    }

    fn lt_quiet<T: Borrow<Self>>(&self, x: T) -> bool {
        unsafe { softfloat_sys::f128_lt_quiet(self.0, x.borrow().0) }
    }

    fn le_quiet<T: Borrow<Self>>(&self, x: T) -> bool {
        unsafe { softfloat_sys::f128_le_quiet(self.0, x.borrow().0) }
    }

    fn eq_signaling<T: Borrow<Self>>(&self, x: T) -> bool {
        unsafe { softfloat_sys::f128_eq_signaling(self.0, x.borrow().0) }
    }

    fn is_signaling_nan(&self) -> bool {
        unsafe { softfloat_sys::f128_isSignalingNaN(self.0) }
    }

    fn from_u32(x: u32, rnd: RoundingMode) -> Self {
        rnd.set();
        let ret = unsafe { softfloat_sys::ui32_to_f128(x) };
        Self(ret)
    }

    fn from_u64(x: u64, rnd: RoundingMode) -> Self {
        rnd.set();
        let ret = unsafe { softfloat_sys::ui64_to_f128(x) };
        Self(ret)
    }

    fn from_i32(x: i32, rnd: RoundingMode) -> Self {
        rnd.set();
        let ret = unsafe { softfloat_sys::i32_to_f128(x) };
        Self(ret)
    }

    fn from_i64(x: i64, rnd: RoundingMode) -> Self {
        rnd.set();
        let ret = unsafe { softfloat_sys::i64_to_f128(x) };
        Self(ret)
    }

    fn to_u32(&self, rnd: RoundingMode, exact: bool) -> u32 {
        let ret = unsafe { softfloat_sys::f128_to_ui32(self.0, rnd.to_softfloat(), exact) };
        ret as u32
    }

    fn to_u64(&self, rnd: RoundingMode, exact: bool) -> u64 {
        let ret = unsafe { softfloat_sys::f128_to_ui64(self.0, rnd.to_softfloat(), exact) };
        ret
    }

    fn to_i32(&self, rnd: RoundingMode, exact: bool) -> i32 {
        let ret = unsafe { softfloat_sys::f128_to_i32(self.0, rnd.to_softfloat(), exact) };
        ret as i32
    }

    fn to_i64(&self, rnd: RoundingMode, exact: bool) -> i64 {
        let ret = unsafe { softfloat_sys::f128_to_i64(self.0, rnd.to_softfloat(), exact) };
        ret
    }

    fn to_f16(&self, rnd: RoundingMode) -> F16 {
        rnd.set();
        let ret = unsafe { softfloat_sys::f128_to_f16(self.0) };
        F16::from_bits(ret.v)
    }

    fn to_f32(&self, rnd: RoundingMode) -> F32 {
        rnd.set();
        let ret = unsafe { softfloat_sys::f128_to_f32(self.0) };
        F32::from_bits(ret.v)
    }

    fn to_f64(&self, rnd: RoundingMode) -> F64 {
        rnd.set();
        let ret = unsafe { softfloat_sys::f128_to_f64(self.0) };
        F64::from_bits(ret.v)
    }

    fn to_f128(&self, _rnd: RoundingMode) -> F128 {
        Self::from_bits(self.to_bits())
    }

    fn round_to_integral(&self, rnd: RoundingMode) -> Self {
        let ret = unsafe { softfloat_sys::f128_roundToInt(self.0, rnd.to_softfloat(), false) };
        Self(ret)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ExceptionFlags;
    use std::cmp::Ordering;

    #[test]
    fn f128_add() {
        let a = 0x12345678ffffffffffffffffffffffff;
        let b = 0x76546410aaaaaaaaffffffffffffffff;
        let a0 = F128::from_bits(a);
        let b0 = F128::from_bits(b);
        let d0 = a0.add(b0, RoundingMode::TiesToEven);
        let a1 = simple_soft_float::F128::from_bits(a);
        let b1 = simple_soft_float::F128::from_bits(b);
        let d1 = a1.add(&b1, Some(simple_soft_float::RoundingMode::TiesToEven), None);
        assert_eq!(d0.to_bits(), *d1.bits());
    }

    #[test]
    fn f128_sub() {
        let a = 0x12345678ffffffffffffffffffffffff;
        let b = 0x76546410aaaaaaaaffffffffffffffff;
        let a0 = F128::from_bits(a);
        let b0 = F128::from_bits(b);
        let d0 = a0.sub(b0, RoundingMode::TiesToEven);
        let a1 = simple_soft_float::F128::from_bits(a);
        let b1 = simple_soft_float::F128::from_bits(b);
        let d1 = a1.sub(&b1, Some(simple_soft_float::RoundingMode::TiesToEven), None);
        assert_eq!(d0.to_bits(), *d1.bits());
    }

    #[test]
    fn f128_mul() {
        let a = 0x12345678ffffffffffffffffffffffff;
        let b = 0x76546410aaaaaaaaffffffffffffffff;
        let a0 = F128::from_bits(a);
        let b0 = F128::from_bits(b);
        let d0 = a0.mul(b0, RoundingMode::TiesToEven);
        let a1 = simple_soft_float::F128::from_bits(a);
        let b1 = simple_soft_float::F128::from_bits(b);
        let d1 = a1.mul(&b1, Some(simple_soft_float::RoundingMode::TiesToEven), None);
        assert_eq!(d0.to_bits(), *d1.bits());
    }

    #[test]
    #[ignore]
    fn f128_fused_mul_add() {
        let a = 0x12345678ffffffffffffffffffffffff;
        let b = 0x12345678aaaaaaaaffffffffffffffff;
        let c = 0x12345678aaaaaaaaffffffffffffffff;
        let a0 = F128::from_bits(a);
        let b0 = F128::from_bits(b);
        let c0 = F128::from_bits(c);
        let d0 = a0.fused_mul_add(b0, c0, RoundingMode::TiesToEven);
        let a1 = simple_soft_float::F128::from_bits(a);
        let b1 = simple_soft_float::F128::from_bits(b);
        let c1 = simple_soft_float::F128::from_bits(c);
        let d1 = a1.fused_mul_add(
            &b1,
            &c1,
            Some(simple_soft_float::RoundingMode::TiesToEven),
            None,
        );
        assert_eq!(d0.to_bits(), *d1.bits());
    }

    #[test]
    fn f128_div() {
        let a = 0x76545678ffffffffffffffffffffffff;
        let b = 0x12346410aaaaaaaaffffffffffffffff;
        let a0 = F128::from_bits(a);
        let b0 = F128::from_bits(b);
        let d0 = a0.div(b0, RoundingMode::TiesToEven);
        let a1 = simple_soft_float::F128::from_bits(a);
        let b1 = simple_soft_float::F128::from_bits(b);
        let d1 = a1.div(&b1, Some(simple_soft_float::RoundingMode::TiesToEven), None);
        assert_eq!(d0.to_bits(), *d1.bits());
    }

    #[test]
    #[ignore]
    fn f128_rem() {
        let a = 0x76545678ffffffffffffffffffffffff;
        let b = 0x12346410aaaaaaaaffffffffffffffff;
        let a0 = F128::from_bits(a);
        let b0 = F128::from_bits(b);
        let d0 = a0.rem(b0, RoundingMode::TiesToEven);
        let a1 = simple_soft_float::F128::from_bits(a);
        let b1 = simple_soft_float::F128::from_bits(b);
        let d1 = a1.ieee754_remainder(&b1, Some(simple_soft_float::RoundingMode::TiesToEven), None);
        assert_eq!(d0.to_bits(), *d1.bits());
    }

    #[test]
    #[ignore]
    fn f128_sqrt() {
        let a = 0x76546410aaaaaaaaffffffffffffffff;
        let a0 = F128::from_bits(a);
        let d0 = a0.sqrt(RoundingMode::TiesToEven);
        let a1 = simple_soft_float::F128::from_bits(a);
        let d1 = a1.sqrt(Some(simple_soft_float::RoundingMode::TiesToEven), None);
        assert_eq!(d0.to_bits(), *d1.bits());
    }

    #[test]
    fn f128_compare() {
        let a = F128::from_bits(0x76546410ffffffffffffffffffffffff);
        let b = F128::from_bits(0x12345678aaaaaaaaffffffffffffffff);
        let d = a.compare(b);
        assert_eq!(d, Some(Ordering::Greater));

        let a = F128::from_bits(0x12345678ffffffffffffffffffffffff);
        let b = F128::from_bits(0x76546410aaaaaaaaffffffffffffffff);
        let d = a.compare(b);
        assert_eq!(d, Some(Ordering::Less));

        let a = F128::from_bits(0x12345678aaaaaaaaffffffffffffffff);
        let b = F128::from_bits(0x12345678aaaaaaaaffffffffffffffff);
        let d = a.compare(b);
        assert_eq!(d, Some(Ordering::Equal));
    }

    #[test]
    fn f128_signaling() {
        let a = F128::from_bits(0x7fff0000000000000000000000000001);
        let b = F128::from_bits(0x7fff8000000000000000000000000001);
        assert_eq!(a.is_signaling_nan(), true);
        assert_eq!(b.is_signaling_nan(), false);

        let mut flag = ExceptionFlags::default();
        flag.set();
        assert_eq!(a.eq(a), false);
        flag.get();
        assert_eq!(flag.is_invalid(), true);

        let mut flag = ExceptionFlags::default();
        flag.set();
        assert_eq!(b.eq(b), false);
        flag.get();
        assert_eq!(flag.is_invalid(), false);

        let mut flag = ExceptionFlags::default();
        flag.set();
        assert_eq!(a.eq_signaling(a), false);
        flag.get();
        assert_eq!(flag.is_invalid(), true);

        let mut flag = ExceptionFlags::default();
        flag.set();
        assert_eq!(b.eq_signaling(b), false);
        flag.get();
        assert_eq!(flag.is_invalid(), true);
    }

    #[test]
    fn from_f32() {
        let a = F128::from_f32(0.1);
        assert_eq!(a.to_bits(), 0x3ffb99999a0000000000000000000000);
    }

    #[test]
    fn from_f64() {
        let a = F128::from_f64(0.1);
        assert_eq!(a.to_bits(), 0x3ffb999999999999a000000000000000);
    }
}
