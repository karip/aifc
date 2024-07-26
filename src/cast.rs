
/// Casts i8 to u8, mapping negative values to 128..255.
#[allow(clippy::cast_sign_loss)] // mapping to positive values is expected
#[inline(always)]
pub const fn i8_to_u8(value: i8) -> u8 {
    value as u8
}

/// Casts u8 to i8, mapping 128..255 to negative values.
#[allow(clippy::cast_possible_wrap)] // mapping to negative values is expected
#[inline(always)]
pub const fn u8_to_i8(value: u8) -> i8 {
    value as i8
}

/// Casts u16 to i16, mapping 32768..65535 to negative values.
#[allow(clippy::cast_possible_wrap)] // mapping to negative values is expected
#[inline(always)]
pub const fn u16_to_i16(value: u16) -> i16 {
    value as i16
}

/// Casts u64 to u8, keeping only the lowest 8 bits.
#[allow(clippy::cast_possible_truncation)] // truncation is expected
#[inline(always)]
pub const fn u64_to_u8(value: u64) -> u8 {
    value as u8
}

/// Casts usize to u64. Returns an error if usize doesn't fit in u64.
#[inline(always)]
pub fn usize_to_u64(value: usize, err: crate::AifcError) -> crate::AifcResult<u64> {
    // this should always succeed, unless usize is extended to be 128 bits long
    u64::try_from(value).map_err(|_| err)
}

/// Casts u64 to f64.
#[inline(always)]
pub const fn u64_to_f64(value: u64) -> f64 {
    // this always succeeds
    value as f64
}

/// Casts i16 to u64 clamping negative values to 0.
#[allow(clippy::cast_sign_loss)] // value has been checked to be positive before casting
#[inline(always)]
pub const fn clamp_i16_to_u64(value: i16) -> u64 {
    if value >= 0 {
        value as u64
    } else {
        0
    }
}

/// Casts i16 to u64 clamping negative values to 0.
#[allow(clippy::cast_sign_loss)] // value has been checked to be positive before casting
#[inline(always)]
pub const fn clamp_i16_to_usize(value: i16) -> usize {
    if value >= 0 {
        value as usize
    } else {
        0
    }
}
