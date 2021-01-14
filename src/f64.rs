use crate::{Float, RoundingMode, F128, F16, F32};
use softfloat_sys::float64_t;
use std::borrow::Borrow;

/// standard 64-bit float
#[derive(Copy, Clone, Debug)]
pub struct F64(float64_t);

impl Float for F64 {
    type Payload = u64;

    const EXPONENT_BIT: Self::Payload = 0x7ff;
    const FRACTION_BIT: Self::Payload = 0xf_ffff_ffff_ffff;
    const SIGN_POS: usize = 63;
    const EXPONENT_POS: usize = 52;

    #[inline]
    fn set_payload(&mut self, x: Self::Payload) {
        self.0.v = x;
    }

    #[inline]
    fn from_bits(v: Self::Payload) -> Self {
        Self(float64_t { v })
    }

    #[inline]
    fn to_bits(&self) -> Self::Payload {
        self.0.v
    }

    #[inline]
    fn bits(&self) -> Self::Payload {
        self.to_bits()
    }

    fn add<T: Borrow<Self>>(&self, x: T, rnd: RoundingMode) -> Self {
        rnd.set();
        let ret = unsafe { softfloat_sys::f64_add(self.0, x.borrow().0) };
        Self(ret)
    }

    fn sub<T: Borrow<Self>>(&self, x: T, rnd: RoundingMode) -> Self {
        rnd.set();
        let ret = unsafe { softfloat_sys::f64_sub(self.0, x.borrow().0) };
        Self(ret)
    }

    fn mul<T: Borrow<Self>>(&self, x: T, rnd: RoundingMode) -> Self {
        rnd.set();
        let ret = unsafe { softfloat_sys::f64_mul(self.0, x.borrow().0) };
        Self(ret)
    }

    fn fused_mul_add<T: Borrow<Self>>(&self, x: T, y: T, rnd: RoundingMode) -> Self {
        rnd.set();
        let ret = unsafe { softfloat_sys::f64_mulAdd(self.0, x.borrow().0, y.borrow().0) };
        Self(ret)
    }

    fn div<T: Borrow<Self>>(&self, x: T, rnd: RoundingMode) -> Self {
        rnd.set();
        let ret = unsafe { softfloat_sys::f64_div(self.0, x.borrow().0) };
        Self(ret)
    }

    fn rem<T: Borrow<Self>>(&self, x: T, rnd: RoundingMode) -> Self {
        rnd.set();
        let ret = unsafe { softfloat_sys::f64_rem(self.0, x.borrow().0) };
        Self(ret)
    }

    fn sqrt(&self, rnd: RoundingMode) -> Self {
        rnd.set();
        let ret = unsafe { softfloat_sys::f64_sqrt(self.0) };
        Self(ret)
    }

    fn eq<T: Borrow<Self>>(&self, x: T) -> bool {
        unsafe { softfloat_sys::f64_eq(self.0, x.borrow().0) }
    }

    fn lt<T: Borrow<Self>>(&self, x: T) -> bool {
        unsafe { softfloat_sys::f64_lt(self.0, x.borrow().0) }
    }

    fn le<T: Borrow<Self>>(&self, x: T) -> bool {
        unsafe { softfloat_sys::f64_le(self.0, x.borrow().0) }
    }

    fn lt_quiet<T: Borrow<Self>>(&self, x: T) -> bool {
        unsafe { softfloat_sys::f64_lt_quiet(self.0, x.borrow().0) }
    }

    fn le_quiet<T: Borrow<Self>>(&self, x: T) -> bool {
        unsafe { softfloat_sys::f64_le_quiet(self.0, x.borrow().0) }
    }

    fn from_u32(x: u32, rnd: RoundingMode) -> Self {
        rnd.set();
        let ret = unsafe { softfloat_sys::ui32_to_f64(x) };
        Self(ret)
    }

    fn from_u64(x: u64, rnd: RoundingMode) -> Self {
        rnd.set();
        let ret = unsafe { softfloat_sys::ui64_to_f64(x) };
        Self(ret)
    }

    fn from_i32(x: i32, rnd: RoundingMode) -> Self {
        rnd.set();
        let ret = unsafe { softfloat_sys::i32_to_f64(x) };
        Self(ret)
    }

    fn from_i64(x: i64, rnd: RoundingMode) -> Self {
        rnd.set();
        let ret = unsafe { softfloat_sys::i64_to_f64(x) };
        Self(ret)
    }

    fn to_u32(&self, rnd: RoundingMode, exact: bool) -> u32 {
        let ret = unsafe { softfloat_sys::f64_to_ui32(self.0, rnd.to_softfloat(), exact) };
        ret as u32
    }

    fn to_u64(&self, rnd: RoundingMode, exact: bool) -> u64 {
        let ret = unsafe { softfloat_sys::f64_to_ui64(self.0, rnd.to_softfloat(), exact) };
        ret
    }

    fn to_i32(&self, rnd: RoundingMode, exact: bool) -> i32 {
        let ret = unsafe { softfloat_sys::f64_to_i32(self.0, rnd.to_softfloat(), exact) };
        ret as i32
    }

    fn to_i64(&self, rnd: RoundingMode, exact: bool) -> i64 {
        let ret = unsafe { softfloat_sys::f64_to_i64(self.0, rnd.to_softfloat(), exact) };
        ret
    }

    fn to_f16(&self, rnd: RoundingMode) -> F16 {
        rnd.set();
        let ret = unsafe { softfloat_sys::f64_to_f16(self.0) };
        F16::from_bits(ret.v)
    }

    fn to_f32(&self, rnd: RoundingMode) -> F32 {
        rnd.set();
        let ret = unsafe { softfloat_sys::f64_to_f32(self.0) };
        F32::from_bits(ret.v)
    }

    fn to_f64(&self, _rnd: RoundingMode) -> F64 {
        Self::from_bits(self.to_bits())
    }

    fn to_f128(&self, rnd: RoundingMode) -> F128 {
        rnd.set();
        let ret = unsafe { softfloat_sys::f64_to_f128(self.0) };
        let mut v = 0u128;
        v |= ret.v[0] as u128;
        v |= (ret.v[1] as u128) << 64;
        F128::from_bits(v)
    }

    fn round_to_integral(&self, rnd: RoundingMode) -> Self {
        let ret = unsafe { softfloat_sys::f64_roundToInt(self.0, rnd.to_softfloat(), false) };
        Self(ret)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cmp::Ordering;

    #[test]
    fn f64_add() {
        let a = 0x12345678ffffffff;
        let b = 0x76546410aaaaaaaa;
        let a0 = F64::from_bits(a);
        let b0 = F64::from_bits(b);
        let d0 = a0.add(b0, RoundingMode::TiesToEven);
        let a1 = simple_soft_float::F64::from_bits(a);
        let b1 = simple_soft_float::F64::from_bits(b);
        let d1 = a1.add(&b1, Some(simple_soft_float::RoundingMode::TiesToEven), None);
        assert_eq!(d0.to_bits(), *d1.bits());
    }

    #[test]
    fn f64_sub() {
        let a = 0x12345678ffffffff;
        let b = 0x76546410aaaaaaaa;
        let a0 = F64::from_bits(a);
        let b0 = F64::from_bits(b);
        let d0 = a0.sub(b0, RoundingMode::TiesToEven);
        let a1 = simple_soft_float::F64::from_bits(a);
        let b1 = simple_soft_float::F64::from_bits(b);
        let d1 = a1.sub(&b1, Some(simple_soft_float::RoundingMode::TiesToEven), None);
        assert_eq!(d0.to_bits(), *d1.bits());
    }

    #[test]
    fn f64_mul() {
        let a = 0x12345678ffffffff;
        let b = 0x76546410aaaaaaaa;
        let a0 = F64::from_bits(a);
        let b0 = F64::from_bits(b);
        let d0 = a0.mul(b0, RoundingMode::TiesToEven);
        let a1 = simple_soft_float::F64::from_bits(a);
        let b1 = simple_soft_float::F64::from_bits(b);
        let d1 = a1.mul(&b1, Some(simple_soft_float::RoundingMode::TiesToEven), None);
        assert_eq!(d0.to_bits(), *d1.bits());
    }

    #[test]
    fn f64_fused_mul_add() {
        let a = 0x12345678ffffffff;
        let b = 0x12345678aaaaaaaa;
        let c = 0x12345678aaaaaaaa;
        let a0 = F64::from_bits(a);
        let b0 = F64::from_bits(b);
        let c0 = F64::from_bits(c);
        let d0 = a0.fused_mul_add(b0, c0, RoundingMode::TiesToEven);
        let a1 = simple_soft_float::F64::from_bits(a);
        let b1 = simple_soft_float::F64::from_bits(b);
        let c1 = simple_soft_float::F64::from_bits(c);
        let d1 = a1.fused_mul_add(
            &b1,
            &c1,
            Some(simple_soft_float::RoundingMode::TiesToEven),
            None,
        );
        assert_eq!(d0.to_bits(), *d1.bits());
    }

    #[test]
    fn f64_div() {
        let a = 0x76545678ffffffff;
        let b = 0x12346410aaaaaaaa;
        let a0 = F64::from_bits(a);
        let b0 = F64::from_bits(b);
        let d0 = a0.div(b0, RoundingMode::TiesToEven);
        let a1 = simple_soft_float::F64::from_bits(a);
        let b1 = simple_soft_float::F64::from_bits(b);
        let d1 = a1.div(&b1, Some(simple_soft_float::RoundingMode::TiesToEven), None);
        assert_eq!(d0.to_bits(), *d1.bits());
    }

    #[test]
    fn f64_rem() {
        let a = 0x76545678ffffffff;
        let b = 0x12346410aaaaaaaa;
        let a0 = F64::from_bits(a);
        let b0 = F64::from_bits(b);
        let d0 = a0.rem(b0, RoundingMode::TiesToEven);
        let a1 = simple_soft_float::F64::from_bits(a);
        let b1 = simple_soft_float::F64::from_bits(b);
        let d1 = a1.ieee754_remainder(&b1, Some(simple_soft_float::RoundingMode::TiesToEven), None);
        assert_eq!(d0.to_bits(), *d1.bits());
    }

    #[test]
    fn f64_sqrt() {
        let a = 0x76546410aaaaaaaa;
        let a0 = F64::from_bits(a);
        let d0 = a0.sqrt(RoundingMode::TiesToEven);
        let a1 = simple_soft_float::F64::from_bits(a);
        let d1 = a1.sqrt(Some(simple_soft_float::RoundingMode::TiesToEven), None);
        assert_eq!(d0.to_bits(), *d1.bits());
    }

    #[test]
    fn f64_compare() {
        let a = F64::from_bits(0x76546410ffffffff);
        let b = F64::from_bits(0x12345678aaaaaaaa);
        let d = a.compare(b);
        assert_eq!(d, Some(Ordering::Greater));

        let a = F64::from_bits(0x12345678ffffffff);
        let b = F64::from_bits(0x76546410aaaaaaaa);
        let d = a.compare(b);
        assert_eq!(d, Some(Ordering::Less));

        let a = F64::from_bits(0x12345678aaaaaaaa);
        let b = F64::from_bits(0x12345678aaaaaaaa);
        let d = a.compare(b);
        assert_eq!(d, Some(Ordering::Equal));
    }
}
