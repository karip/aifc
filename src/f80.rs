use crate::cast;

#[cfg(feature = "internal-no-panic")]
use no_panic::no_panic;

/// Converts a 80-bit extended floating point value to a 64-bit floating point value.
#[cfg_attr(feature = "internal-no-panic", no_panic)]
pub fn f80_to_f64(bytes: &[u8]) -> f64 {
    assert_eq!(bytes.len(), 10);

    const POW_2_NEG16382: f64 = 0.0; // 2.powi(-16382) -> rounds to 0.0f64
    const POW_2_63: f64 = cast::u64_to_f64(1u64 << 63); // 2.powi(63)

    // see https://en.wikipedia.org/wiki/Extended_precision#x86_extended_precision_format

    let sign: f64 = if bytes[0] & 0x80 == 0 { 1.0 } else { -1.0 };
    let exponent = u16::from_be_bytes([ bytes[0], bytes[1] ]) & 0x7fff;
    let mut significand_buf = [0u8; 8];
    significand_buf.copy_from_slice(&bytes[2..10]);
    let significand = u64::from_be_bytes(significand_buf);

    let bit62 = significand >> 62 & 0x01;
    let bit63 = significand >> 63;

    let bits62 = significand & 0x3fffffff_ffffffff;
    let bits61 = significand & 0x1fffffff_ffffffff;

    // note: this may underflow, possibly losing accuracy
    let m = cast::u64_to_f64(significand) / POW_2_63;

    match (exponent, bit63, bit62) {
        (0x0000, 0, _) => {
            if bits62 == 0 { sign * 0.0 } // zero
            else { sign * m * POW_2_NEG16382 } // denormal
        },
        (0x0000, 1, _) => { // pseudo denormal
            sign * m * POW_2_NEG16382
        },

        (0x7fff, 0, 0) => {
            if bits61 == 0 { sign * f64::INFINITY } // pseudo-infinity
            else { f64::NAN } // pseudo Not a Number
        },
        (0x7fff, 0, 1) => { // pseudo Not a Number
            f64::NAN
        },
        (0x7fff, 1, 0) => {
            if bits61 == 0 { sign * f64::INFINITY } // infinity
            else { f64::NAN } // signalling Not a Number
        },
        (0x7fff, 1, 1) => { // floating-point indefinite or Quiet Not a Number
            f64::NAN
        },

        _ => { // unnormal or normalized value
            // powi() isn't available in no_std, so it isn't used:
            //sign * m * 2.0f64.powi(i32::from(exponent)-16383)

            if exponent > 16383+1024 { // overflows f64
                sign * f64::INFINITY
            } else if exponent < 16383-1022 { // underflows f64
                sign * 0.0
            } else if exponent >= 16383 { // positive exponent including zero exponent
                let f64exp = u64::from(exponent - 16383 + 1023);
                let pow = f64::from_bits(f64exp << 52);
                sign * m * pow
            } else { // negative exponent
                let f64exp = u64::from(16383 - exponent + 1023);
                let pow = f64::from_bits(f64exp << 52);
                sign * m / pow
            }
        }
    }
}

/// Converts a 64-bit floating point value to a 80-bit extended floating point value.
#[cfg_attr(feature = "internal-no-panic", no_panic)]
pub fn f64_to_f80(value: f64) -> [u8; 10] {
    if !value.is_finite() {
        if !value.is_nan() {
            if value.is_sign_positive() { // +inf
                return [127, 255, 0, 0, 0, 0, 0, 0, 0, 0];
            } else { // -inf
                return [255, 255, 0, 0, 0, 0, 0, 0, 0, 0];
            }
        } else { // nan
            return [127, 255, 255, 255, 255, 255, 255, 255, 255, 255];
        }
    }

    if value == 0.0 {
        if value.is_sign_positive() { // positive zero
            return [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ];
        } else { // negative zero
            return [ 128, 0, 0, 0, 0, 0, 0, 0, 0, 0 ];
        }
    }
    // shift f80 bits to match f64
    let bits = value.to_bits();
    let sign = bits & 0x80000000_00000000;
    let exponent = ((bits & 0x7ff00000_00000000) >> 52) + 16383 - 1023;
    let fraction = bits & 0x000fffff_ffffffff;
    [
        cast::u64_to_u8(sign >> 56 | (exponent >> 8)),
        cast::u64_to_u8(exponent & 0xff),
        // bit 63 is always 1 to match f64
        cast::u64_to_u8(0x80 | ((fraction & 0x000fe000_00000000) >> 45)),
        cast::u64_to_u8((fraction & 0x00001fe0_00000000) >> 37),
        cast::u64_to_u8((fraction & 0x0000001f_e0000000) >> 29),
        cast::u64_to_u8((fraction & 0x00000000_1fe00000) >> 21),

        cast::u64_to_u8((fraction & 0x00000000_001fe000) >> 13),
        cast::u64_to_u8((fraction & 0x00000000_00001fe0) >> 5),
        cast::u64_to_u8((fraction & 0x00000000_0000001f) << 3),
        0
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compares two 64-bit floating points for equality, checking the signs of signed zeros
    /// and making two NANs match.
    macro_rules! assert_eq_f64 {
        ($left:expr, $right:expr $(,)?) => {
            let l: f64 = $left;
            let r: f64 = $right;
            if (r.is_nan()) { assert!(l.is_nan()); }
            else if (l == 0.0 && r == 0.0) { assert_eq!(l.signum(), r.signum()); }
            else { assert_eq!($left, $right); }
        }
    }

    /// Casts u16 to u8, keeping only the lowest 8 bits.
    const fn u16_to_u8(value: u16) -> u8 {
        value as u8
    }

    /// Creates f80 [u8; 10] from the given parameters.
    fn f80(out: &mut [u8; 10], sign: i8, exponent: u16, bit63: u8, bit62: u8,
            significand: u64) -> &[u8; 10] {
        out[0] = (cast::i8_to_u8(sign) & 0x80) | u16_to_u8(exponent >> 8 & 0x7f);
        out[1] = u16_to_u8(exponent & 0xff);
        let sbytes = significand.to_be_bytes();
        out[2] = sbytes[0] | (bit63 << 7) | (bit62 << 6);
        out[3] = sbytes[1];
        out[4] = sbytes[2];
        out[5] = sbytes[3];
        out[6] = sbytes[4];
        out[7] = sbytes[5];
        out[8] = sbytes[6];
        out[9] = sbytes[7];
        out
    }

    #[test]
    fn test_f80_to_f64() {
        let mut f80buf: [u8; 10] = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ];

        // exponent all zeros

        assert_eq_f64!(f80_to_f64(f80(&mut f80buf, 1, 0x0000,  0, 0,  0)), 0.0);
        assert_eq_f64!(f80_to_f64(f80(&mut f80buf, -1, 0x0000,  0, 0,  0)), -0.0);
        assert_eq_f64!(f80_to_f64(f80(&mut f80buf,  1, 0x0000,  0, 1,  1000000)), 0.0);
        assert_eq_f64!(f80_to_f64(f80(&mut f80buf,  1, 0x0000,  0, 1,  0x7fffffff_ffffffff)), 0.0);
        assert_eq_f64!(f80_to_f64(f80(&mut f80buf, -1, 0x0000,  1, 0,  1234)), -0.0);
        assert_eq_f64!(f80_to_f64(f80(&mut f80buf, -1, 0x0000,  1, 1,  0xffffffff_ffffffff)), -0.0);

        // exponent all ones: infinities and NANs

        assert_eq_f64!(f80_to_f64(f80(&mut f80buf,  1, 0x7fff,  0, 0,  0)), f64::INFINITY);
        assert_eq_f64!(f80_to_f64(f80(&mut f80buf, -1, 0x7fff,  0, 0,  0)), -f64::INFINITY);
        assert_eq_f64!(f80_to_f64(f80(&mut f80buf,  1, 0x7fff,  0, 0,  2)), f64::NAN);
        assert_eq_f64!(f80_to_f64(f80(&mut f80buf, -1, 0x7fff,  0, 0,  3)), f64::NAN);

        assert_eq_f64!(f80_to_f64(f80(&mut f80buf,  1, 0x7fff,  0, 1,  0)), f64::NAN);
        assert_eq_f64!(f80_to_f64(f80(&mut f80buf, -1, 0x7fff,  0, 1,  6)), f64::NAN);

        assert_eq_f64!(f80_to_f64(f80(&mut f80buf,  1, 0x7fff,  1, 0,  0)), f64::INFINITY);
        assert_eq_f64!(f80_to_f64(f80(&mut f80buf, -1, 0x7fff,  1, 0,  0)), -f64::INFINITY);
        assert_eq_f64!(f80_to_f64(f80(&mut f80buf,  1, 0x7fff,  1, 0,  2)), f64::NAN);
        assert_eq_f64!(f80_to_f64(f80(&mut f80buf, -1, 0x7fff,  1, 0,  6)), f64::NAN);

        assert_eq_f64!(f80_to_f64(f80(&mut f80buf,  1, 0x7fff,  1, 1,  0)), f64::NAN);
        assert_eq_f64!(f80_to_f64(f80(&mut f80buf, -1, 0x7fff,  1, 1,  6)), f64::NAN);

        // all other values

        assert_eq_f64!(f80_to_f64(f80(&mut f80buf,  1, 16383,  0, 0,  0x00000000_00000000)), 0.0);
        assert_eq_f64!(f80_to_f64(f80(&mut f80buf,  1, 16383,  0, 0,  0x80000000_00000000)), 1.0);
        assert_eq_f64!(f80_to_f64(f80(&mut f80buf, -1, 16383,  0, 0,  0x80000000_00000000)), -1.0);

        assert_eq_f64!(f80_to_f64(f80(&mut f80buf,  1, 16384,  0, 0,  0x00000000_00000000)), 0.0);
        assert_eq_f64!(f80_to_f64(f80(&mut f80buf,  1, 16384,  0, 0,  0x80000000_00000000)), 2.0);
        assert_eq_f64!(f80_to_f64(f80(&mut f80buf, -1, 16384,  0, 0,  0x80000000_00000000)), -2.0);

        assert_eq_f64!(f80_to_f64(f80(&mut f80buf,  1, 16382,  0, 0,  0x00000000_00000000)), 0.0);
        assert_eq_f64!(f80_to_f64(f80(&mut f80buf,  1, 16382,  0, 0,  0x80000000_00000000)), 0.5);
        assert_eq_f64!(f80_to_f64(f80(&mut f80buf, -1, 16382,  0, 0,  0x80000000_00000000)), -0.5);

        assert_eq_f64!(f80_to_f64(f80(&mut f80buf,  1, 16385,  0, 0,  0x40000000_00000000)),
            2.0);
        assert_eq_f64!(f80_to_f64(f80(&mut f80buf, -1, 16385,  0, 0,  0x40000000_00000000)),
            -2.0);
        assert_eq_f64!(f80_to_f64(f80(&mut f80buf,  1, 16381,  0, 0,  0x40000000_00000000)),
            0.125);
        assert_eq_f64!(f80_to_f64(f80(&mut f80buf, -1, 16381,  0, 0,  0x40000000_00000000)),
            -0.125);
        assert_eq_f64!(f80_to_f64(f80(&mut f80buf,  1, 16383+1023,  0, 0,  0x80000000_00000000)),
            8.98846567431158e307);
        assert_eq_f64!(f80_to_f64(f80(&mut f80buf,  1, 16383-1021,  0, 0,  0x80000000_00000000)),
            4.450147717014403e-308);
        assert_eq_f64!(f80_to_f64(&[0x40, 0x0e, 0xac, 0x44, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]),
            44100.0);

        // large f80 value overflows f64
        assert_eq_f64!(f80_to_f64(f80(&mut f80buf,  1, 32766,  0, 0,  0xffffffff_ffffffff)),
            f64::INFINITY);
        assert_eq_f64!(f80_to_f64(f80(&mut f80buf, -1, 32766,  0, 0,  0xffffffff_ffffffff)),
            -f64::INFINITY);

        // small f80 value underflows f64
        assert_eq_f64!(f80_to_f64(f80(&mut f80buf,  1, 1,  0, 0,  0x00000000_00000000)), 0.0);
        assert_eq_f64!(f80_to_f64(f80(&mut f80buf, -1, 1,  0, 0,  0x00000000_00000000)), -0.0);
    }

    #[test]
    fn test_f64_to_f80() {
        assert_eq!(f64_to_f80(0.0), [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]);
        assert_eq!(f64_to_f80(-0.0), [ 128, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]);

        assert_eq!(f64_to_f80(1.0), [63, 255, 128, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(f64_to_f80(2.0), [64, 0, 128, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(f64_to_f80(0.5), [63, 254, 128, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(f64_to_f80(0.125), [63, 252, 128, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(f64_to_f80(20000.0002), [64, 13, 156, 64, 0, 26, 54, 226, 232, 0]);
        assert_eq!(f64_to_f80(44100.0), [64, 14, 172, 68, 0, 0, 0, 0, 0, 0]);
        assert_eq!(f64_to_f80(1.341e152), [65, 248, 163, 221, 229, 184, 249, 213, 80, 0]);
        assert_eq!(f64_to_f80(1.341e-152), [62, 6, 179, 204, 119, 227, 61, 192, 40, 0]);
        assert_eq!(f64_to_f80(8.98846567431158e307), [67, 254, 128, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(f64_to_f80(4.450147717014403e-308), [60, 2, 128, 0, 0, 0, 0, 0, 0, 0]);

        assert_eq!(f64_to_f80(-1.0), [191, 255, 128, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(f64_to_f80(-2.0), [192, 0, 128, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(f64_to_f80(-0.5), [191, 254, 128, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(f64_to_f80(-0.125), [191, 252, 128, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(f64_to_f80(-20000.0002), [192, 13, 156, 64, 0, 26, 54, 226, 232, 0]);
        assert_eq!(f64_to_f80(-44100.0), [192, 14, 172, 68, 0, 0, 0, 0, 0, 0]);
        assert_eq!(f64_to_f80(-1.341e152), [193, 248, 163, 221, 229, 184, 249, 213, 80, 0]);
        assert_eq!(f64_to_f80(-1.341e-152), [190, 6, 179, 204, 119, 227, 61, 192, 40, 0]);

        assert_eq!(f64_to_f80(f64::MAX), [67, 254, 255, 255, 255, 255, 255, 255, 248, 0]);
        assert_eq!(f64_to_f80(f64::MIN), [195, 254, 255, 255, 255, 255, 255, 255, 248, 0]);
        assert_eq!(f64_to_f80(f64::MIN_POSITIVE), [60, 1, 128, 0, 0, 0, 0, 0, 0, 0]);

        assert_eq!(f64_to_f80(f64::INFINITY), [127, 255, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(f64_to_f80(-f64::INFINITY), [255, 255, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(f64_to_f80(f64::NAN), [127, 255, 255, 255, 255, 255, 255, 255, 255, 255]);
    }
}
