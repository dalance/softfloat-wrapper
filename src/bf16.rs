use crate::{Float, RoundingMode, F128, F16, F32, F64};
use softfloat_sys::{float16_t, float32_t};
use std::borrow::Borrow;

/// standard 16-bit float
#[derive(Copy, Clone, Debug)]
pub struct BF16(float16_t);

impl BF16 {
    /// Converts primitive `f32` to `F16`
    pub fn from_f32(v: f32) -> Self {
        F32::from_bits(v.to_bits()).to_bf16(RoundingMode::TiesToEven)
    }

    /// Converts primitive `f64` to `F16`
    pub fn from_f64(v: f64) -> Self {
        F64::from_bits(v.to_bits()).to_bf16(RoundingMode::TiesToEven)
    }
}

fn to_f32(x: float16_t) -> float32_t {
    float32_t {
        v: (x.v as u32) << 16,
    }
}

fn from_f32(x: float32_t) -> float16_t {
    float16_t {
        v: (x.v >> 16) as u16,
    }
}

impl Float for BF16 {
    type Payload = u16;

    const EXPONENT_BIT: Self::Payload = 0xff;
    const FRACTION_BIT: Self::Payload = 0x7f;
    const SIGN_POS: usize = 15;
    const EXPONENT_POS: usize = 7;

    #[inline]
    fn set_payload(&mut self, x: Self::Payload) {
        self.0.v = x;
    }

    #[inline]
    fn from_bits(v: Self::Payload) -> Self {
        Self(float16_t { v })
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
        let ret = unsafe { softfloat_sys::f32_add(to_f32(self.0), to_f32(x.borrow().0)) };
        Self(from_f32(ret))
    }

    fn sub<T: Borrow<Self>>(&self, x: T, rnd: RoundingMode) -> Self {
        rnd.set();
        let ret = unsafe { softfloat_sys::f32_sub(to_f32(self.0), to_f32(x.borrow().0)) };
        Self(from_f32(ret))
    }

    fn mul<T: Borrow<Self>>(&self, x: T, rnd: RoundingMode) -> Self {
        rnd.set();
        let ret = unsafe { softfloat_sys::f32_mul(to_f32(self.0), to_f32(x.borrow().0)) };
        Self(from_f32(ret))
    }

    fn fused_mul_add<T: Borrow<Self>>(&self, x: T, y: T, rnd: RoundingMode) -> Self {
        rnd.set();
        let ret = unsafe {
            softfloat_sys::f32_mulAdd(to_f32(self.0), to_f32(x.borrow().0), to_f32(y.borrow().0))
        };
        Self(from_f32(ret))
    }

    fn div<T: Borrow<Self>>(&self, x: T, rnd: RoundingMode) -> Self {
        rnd.set();
        let ret = unsafe { softfloat_sys::f32_div(to_f32(self.0), to_f32(x.borrow().0)) };
        Self(from_f32(ret))
    }

    fn rem<T: Borrow<Self>>(&self, x: T, rnd: RoundingMode) -> Self {
        rnd.set();
        let ret = unsafe { softfloat_sys::f32_rem(to_f32(self.0), to_f32(x.borrow().0)) };
        Self(from_f32(ret))
    }

    fn sqrt(&self, rnd: RoundingMode) -> Self {
        rnd.set();
        let ret = unsafe { softfloat_sys::f32_sqrt(to_f32(self.0)) };
        Self(from_f32(ret))
    }

    fn eq<T: Borrow<Self>>(&self, x: T) -> bool {
        unsafe { softfloat_sys::f32_eq(to_f32(self.0), to_f32(x.borrow().0)) }
    }

    fn lt<T: Borrow<Self>>(&self, x: T) -> bool {
        unsafe { softfloat_sys::f32_lt(to_f32(self.0), to_f32(x.borrow().0)) }
    }

    fn le<T: Borrow<Self>>(&self, x: T) -> bool {
        unsafe { softfloat_sys::f32_le(to_f32(self.0), to_f32(x.borrow().0)) }
    }

    fn lt_quiet<T: Borrow<Self>>(&self, x: T) -> bool {
        unsafe { softfloat_sys::f32_lt_quiet(to_f32(self.0), to_f32(x.borrow().0)) }
    }

    fn le_quiet<T: Borrow<Self>>(&self, x: T) -> bool {
        unsafe { softfloat_sys::f32_le_quiet(to_f32(self.0), to_f32(x.borrow().0)) }
    }

    fn eq_signaling<T: Borrow<Self>>(&self, x: T) -> bool {
        unsafe { softfloat_sys::f32_eq_signaling(to_f32(self.0), to_f32(x.borrow().0)) }
    }

    fn is_signaling_nan(&self) -> bool {
        unsafe { softfloat_sys::f32_isSignalingNaN(to_f32(self.0)) }
    }

    fn from_u32(x: u32, rnd: RoundingMode) -> Self {
        rnd.set();
        let ret = unsafe { softfloat_sys::ui32_to_f32(x) };
        Self(from_f32(ret))
    }

    fn from_u64(x: u64, rnd: RoundingMode) -> Self {
        rnd.set();
        let ret = unsafe { softfloat_sys::ui64_to_f32(x) };
        Self(from_f32(ret))
    }

    fn from_i32(x: i32, rnd: RoundingMode) -> Self {
        rnd.set();
        let ret = unsafe { softfloat_sys::i32_to_f32(x) };
        Self(from_f32(ret))
    }

    fn from_i64(x: i64, rnd: RoundingMode) -> Self {
        rnd.set();
        let ret = unsafe { softfloat_sys::i64_to_f32(x) };
        Self(from_f32(ret))
    }

    fn to_u32(&self, rnd: RoundingMode, exact: bool) -> u32 {
        let ret = unsafe { softfloat_sys::f32_to_ui32(to_f32(self.0), rnd.to_softfloat(), exact) };
        ret as u32
    }

    fn to_u64(&self, rnd: RoundingMode, exact: bool) -> u64 {
        let ret = unsafe { softfloat_sys::f32_to_ui64(to_f32(self.0), rnd.to_softfloat(), exact) };
        ret
    }

    fn to_i32(&self, rnd: RoundingMode, exact: bool) -> i32 {
        let ret = unsafe { softfloat_sys::f32_to_i32(to_f32(self.0), rnd.to_softfloat(), exact) };
        ret as i32
    }

    fn to_i64(&self, rnd: RoundingMode, exact: bool) -> i64 {
        let ret = unsafe { softfloat_sys::f32_to_i64(to_f32(self.0), rnd.to_softfloat(), exact) };
        ret
    }

    fn to_f16(&self, rnd: RoundingMode) -> F16 {
        rnd.set();
        let ret = unsafe { softfloat_sys::f32_to_f16(to_f32(self.0)) };
        F16::from_bits(ret.v)
    }

    fn to_bf16(&self, _rnd: RoundingMode) -> BF16 {
        BF16::from_bits(self.to_bits())
    }

    fn to_f32(&self, _rnd: RoundingMode) -> F32 {
        F32::from_bits(to_f32(self.0).v)
    }

    fn to_f64(&self, rnd: RoundingMode) -> F64 {
        rnd.set();
        let ret = unsafe { softfloat_sys::f32_to_f64(to_f32(self.0)) };
        F64::from_bits(ret.v)
    }

    fn to_f128(&self, rnd: RoundingMode) -> F128 {
        rnd.set();
        let ret = unsafe { softfloat_sys::f32_to_f128(to_f32(self.0)) };
        let mut v = 0u128;
        v |= ret.v[0] as u128;
        v |= (ret.v[1] as u128) << 64;
        F128::from_bits(v)
    }

    fn round_to_integral(&self, rnd: RoundingMode) -> Self {
        let ret =
            unsafe { softfloat_sys::f32_roundToInt(to_f32(self.0), rnd.to_softfloat(), false) };
        Self(from_f32(ret))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ExceptionFlags;
    use std::cmp::Ordering;

    #[test]
    fn bf16_add() {
        let a = 0x1234;
        let b = 0x7654;
        let a0 = BF16::from_bits(a);
        let b0 = BF16::from_bits(b);
        let d0 = a0.add(b0, RoundingMode::TiesToEven);
        let a1 = simple_soft_float::F32::from_bits((a as u32) << 16);
        let b1 = simple_soft_float::F32::from_bits((b as u32) << 16);
        let d1 = a1.add(&b1, Some(simple_soft_float::RoundingMode::TiesToEven), None);
        assert_eq!(d0.to_bits(), (*d1.bits() >> 16) as u16);
    }

    #[test]
    fn bf16_sub() {
        let a = 0x1234;
        let b = 0x7654;
        let a0 = BF16::from_bits(a);
        let b0 = BF16::from_bits(b);
        let d0 = a0.sub(b0, RoundingMode::TiesToEven);
        let a1 = simple_soft_float::F32::from_bits((a as u32) << 16);
        let b1 = simple_soft_float::F32::from_bits((b as u32) << 16);
        let d1 = a1.sub(&b1, Some(simple_soft_float::RoundingMode::TiesToEven), None);
        assert_eq!(d0.to_bits(), (*d1.bits() >> 16) as u16);
    }

    #[test]
    fn bf16_mul() {
        let a = 0x1234;
        let b = 0x7654;
        let a0 = BF16::from_bits(a);
        let b0 = BF16::from_bits(b);
        let d0 = a0.mul(b0, RoundingMode::TiesToEven);
        let a1 = simple_soft_float::F32::from_bits((a as u32) << 16);
        let b1 = simple_soft_float::F32::from_bits((b as u32) << 16);
        let d1 = a1.mul(&b1, Some(simple_soft_float::RoundingMode::TiesToEven), None);
        assert_eq!(d0.to_bits(), (*d1.bits() >> 16) as u16);
    }

    #[test]
    fn bf16_fused_mul_add() {
        let a = 0x1234;
        let b = 0x1234;
        let c = 0x1234;
        let a0 = BF16::from_bits(a);
        let b0 = BF16::from_bits(b);
        let c0 = BF16::from_bits(c);
        let d0 = a0.fused_mul_add(b0, c0, RoundingMode::TiesToEven);
        let a1 = simple_soft_float::F32::from_bits((a as u32) << 16);
        let b1 = simple_soft_float::F32::from_bits((b as u32) << 16);
        let c1 = simple_soft_float::F32::from_bits((c as u32) << 16);
        let d1 = a1.fused_mul_add(
            &b1,
            &c1,
            Some(simple_soft_float::RoundingMode::TiesToEven),
            None,
        );
        assert_eq!(d0.to_bits(), (*d1.bits() >> 16) as u16);
    }

    #[test]
    fn bf16_div() {
        let a = 0x7654;
        let b = 0x1234;
        let a0 = BF16::from_bits(a);
        let b0 = BF16::from_bits(b);
        let d0 = a0.div(b0, RoundingMode::TiesToEven);
        let a1 = simple_soft_float::F32::from_bits((a as u32) << 16);
        let b1 = simple_soft_float::F32::from_bits((b as u32) << 16);
        let d1 = a1.div(&b1, Some(simple_soft_float::RoundingMode::TiesToEven), None);
        assert_eq!(d0.to_bits(), (*d1.bits() >> 16) as u16);
    }

    #[test]
    fn bf16_rem() {
        let a = 0x7654;
        let b = 0x1234;
        let a0 = BF16::from_bits(a);
        let b0 = BF16::from_bits(b);
        let d0 = a0.rem(b0, RoundingMode::TiesToEven);
        let a1 = simple_soft_float::F32::from_bits((a as u32) << 16);
        let b1 = simple_soft_float::F32::from_bits((b as u32) << 16);
        let d1 = a1.ieee754_remainder(&b1, Some(simple_soft_float::RoundingMode::TiesToEven), None);
        assert_eq!(d0.to_bits(), (*d1.bits() >> 16) as u16);
    }

    #[test]
    fn bf16_sqrt() {
        let a = 0x7654;
        let a0 = BF16::from_bits(a);
        let d0 = a0.sqrt(RoundingMode::TiesToEven);
        let a1 = simple_soft_float::F32::from_bits((a as u32) << 16);
        let d1 = a1.sqrt(Some(simple_soft_float::RoundingMode::TiesToEven), None);
        assert_eq!(d0.to_bits(), (*d1.bits() >> 16) as u16);
    }

    #[test]
    fn bf16_compare() {
        let a = BF16::from_bits(0x7654);
        let b = BF16::from_bits(0x1234);
        let d = a.compare(b);
        assert_eq!(d, Some(Ordering::Greater));

        let a = BF16::from_bits(0x1234);
        let b = BF16::from_bits(0x7654);
        let d = a.compare(b);
        assert_eq!(d, Some(Ordering::Less));

        let a = BF16::from_bits(0x1234);
        let b = BF16::from_bits(0x1234);
        let d = a.compare(b);
        assert_eq!(d, Some(Ordering::Equal));
    }

    #[test]
    fn bf16_signaling() {
        let a = BF16::from_bits(0x7f81);
        let b = BF16::from_bits(0x7fc1);
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
        let a = BF16::from_f32(0.1);
        assert_eq!(a.to_bits(), 0x3dcc);
    }

    #[test]
    fn from_f64() {
        let a = BF16::from_f64(0.1);
        assert_eq!(a.to_bits(), 0x3dcc);
    }
}
