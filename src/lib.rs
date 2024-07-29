//! # AIFF and AIFF-C Audio Format Reader and Writer
//!
//! This crate contains [`AifcReader`] and [`AifcWriter`] for Audio Interchange File Formats
//! AIFF and AIFF-C. This crate supports uncompressed sample data,
//! μ-law, A-law and IMA ADPCM ("ima4") compressed sample data.
//!
//! These audio formats are made of chunks, which contain header data (the COMM chunk),
//! audio sample data (the SSND chunk) and other data, such as marker data (the MARK chunk).
//!
//! # Examples
//!
//! Reading samples:
//!
//! ```no_run
//! # fn example() -> aifc::AifcResult<()> {
//! let mut stream = std::io::BufReader::new(std::fs::File::open("test.aiff")?);
//! let mut reader = aifc::AifcReader::new(&mut stream)?;
//! let info = reader.read_info()?;
//! for sample in reader.samples()? {
//!     println!("Got sample {:?}", sample.expect("Sample read error"));
//! }
//! # Ok(())
//! # }
//! ```
//!
//! Writing AIFF-C with the default 2 channels, sample rate 44100 and signed 16-bit integer samples:
//!
//! ```no_run
//! # fn example() -> aifc::AifcResult<()> {
//! let mut stream = std::io::BufWriter::new(std::fs::File::create("test.aiff")?);
//! let info = aifc::AifcWriteInfo::default();
//! let mut writer = aifc::AifcWriter::new(&mut stream, &info)?;
//! writer.write_samples_i16(&[ 0, 10, -10, 0 ])?;
//! writer.finalize()?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Text decoding
//!
//! AIFF and AIFF-C originally support ASCII only text in their metadata chunks (NAME, ANNO, etc.).
//! In addition to ASCII, older apps may have used ISO-8859-1 and newer apps may have used UTF-8.
//!
//! A text decoder could try to decode UTF-8 first with
//! [`String::from_utf8()`](String::from_utf8) and if it fails, try to decode ISO-8859-1,
//! and if it fails, decode ASCII. Or it could just assume that everything is UTF-8 and
//! call [`String::from_utf8_lossy()`](String::from_utf8_lossy).
//!
//! When writing new files, the ID3 chunk has proper support for UTF-8 text and can be used
//! as a replacement for most metadata chunks.

#![forbid(
    unsafe_code,
    clippy::panic,
    clippy::exit,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::unimplemented,
    clippy::todo,
    clippy::unreachable,
)]
#![deny(
    clippy::cast_ptr_alignment,
    clippy::char_lit_as_u8,
    clippy::unnecessary_cast,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::checked_conversions,
)]

// silly way to test rust code blocks in README.md
// https://doc.rust-lang.org/rustdoc/write-documentation/documentation-tests.html
#[doc = include_str!("../README.md")]
#[cfg(doctest)]
pub struct ReadmeDoctests;

use std::io::{Read, Write, Seek, SeekFrom};

mod aifcreader;
mod aifcwriter;
mod chunks;
mod aifcresult;
mod cast;
mod f80;

pub use aifcreader::{AifcReader, AifcReadInfo, Samples, Sample, Chunks};
pub use aifcwriter::{AifcWriter, AifcWriteInfo};
pub use chunks::{Markers, Marker, Comments, Comment, Instrument, Loop};
pub use aifcresult::{AifcResult, AifcError};

/// A chunk id is a four byte identifier.
///
/// Valid chunk ids are made of ASCII characters in the range 0x20-0x7e.
/// Chunk ids should not start with a space character (0x20).
/// Chunk ids are case-sensitive.
pub type ChunkId = [u8; 4];

/// Marker id, which should be a positive number.
pub type MarkerId = i16;

fn buffer_size_error() -> std::io::Error { std::io::Error::from(std::io::ErrorKind::InvalidInput) }
fn read_error() -> std::io::Error { std::io::Error::from(std::io::ErrorKind::Other) }
fn unexpectedeof() -> std::io::Error { std::io::Error::from(std::io::ErrorKind::UnexpectedEof) }

/// Offset between comment timestamp and Unix timestamp.
const UNIX_TIMESTAMP_OFFSET: i64 = 2082844800;

const CHUNKID_FORM: [u8; 4] = *b"FORM";
const CHUNKID_AIFF: [u8; 4] = *b"AIFF";
const CHUNKID_AIFC: [u8; 4] = *b"AIFC";

/// The common (header) "COMM" chunk id.
pub const CHUNKID_COMM: [u8; 4] = *b"COMM";
/// The format version "FVER" chunk id.
pub const CHUNKID_FVER: [u8; 4] = *b"FVER";
/// The sound data "SSND" chunk id.
pub const CHUNKID_SSND: [u8; 4] = *b"SSND";
/// The marker "MARK" chunk id.
pub const CHUNKID_MARK: [u8; 4] = *b"MARK";
/// The comments "COMT" chunk id.
pub const CHUNKID_COMT: [u8; 4] = *b"COMT";
/// The instrument "INST" chunk id.
pub const CHUNKID_INST: [u8; 4] = *b"INST";
/// The MIDI "MIDI" data chunk id. A stream may contain multiple MIDI chunks.
pub const CHUNKID_MIDI: [u8; 4] = *b"MIDI";
/// The audio recording "AESD" chunk id.
pub const CHUNKID_AESD: [u8; 4] = *b"AESD";
/// The application specific "APPL" chunk id. A stream may contain multiple APPL chunks.
pub const CHUNKID_APPL: [u8; 4] = *b"APPL";
/// The name "NAME" chunk id.
pub const CHUNKID_NAME: [u8; 4] = *b"NAME";
/// The author "AUTH" chunk id.
pub const CHUNKID_AUTH: [u8; 4] = *b"AUTH";
/// The copyright "(c) " chunk id.
pub const CHUNKID_COPY: [u8; 4] = *b"(c) ";
/// The annotation "ANNO" chunk id. A stream may contain multiple ANNO chunks.
///
/// It is recommended to write comments "COMT" chunks instead of annotation "ANNO" chunks in AIFF-C.
pub const CHUNKID_ANNO: [u8; 4] = *b"ANNO";
/// The ID3 "ID3 " data chunk id.
pub const CHUNKID_ID3: [u8; 4] = *b"ID3 ";

const COMPRESSIONTYPE_NONE: [u8; 4] = *b"NONE";
const COMPRESSIONTYPE_TWOS: [u8; 4] = *b"twos";
const COMPRESSIONTYPE_IN24: [u8; 4] = *b"in24";
const COMPRESSIONTYPE_IN32: [u8; 4] = *b"in32";
const COMPRESSIONTYPE_RAW: [u8; 4] = *b"raw ";
const COMPRESSIONTYPE_SOWT: [u8; 4] = *b"sowt";
const COMPRESSIONTYPE_23NI: [u8; 4] = *b"23ni";
const COMPRESSIONTYPE_FL32_UPPER: [u8; 4] = *b"FL32";
const COMPRESSIONTYPE_FL32: [u8; 4] = *b"fl32";
const COMPRESSIONTYPE_FL64_UPPER: [u8; 4] = *b"FL64";
const COMPRESSIONTYPE_FL64: [u8; 4] = *b"fl64";
const COMPRESSIONTYPE_ULAW: [u8; 4] = *b"ulaw";
const COMPRESSIONTYPE_ULAW_UPPER: [u8; 4] = *b"ULAW";
const COMPRESSIONTYPE_ALAW: [u8; 4] = *b"alaw";
const COMPRESSIONTYPE_ALAW_UPPER: [u8; 4] = *b"ALAW";
const COMPRESSIONTYPE_IMA4: [u8; 4] = *b"ima4";

/// ChunkRef contains a chunk id, chunk start position and its size.
#[derive(Debug, Clone, PartialEq)]
pub struct ChunkRef {
    /// Chunk id.
    pub id: ChunkId,
    /// Chunk start position (the start of the chunk id) in the stream relative
    /// to the start of the FORM chunk.
    pub pos: u64,
    /// Chunk byte size without the chunk id and size fields.
    pub size: u32
}

/// File format: AIFF or AIFF-C.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FileFormat {
    /// AIFF.
    Aiff,
    /// AIFF-C.
    Aifc
}

/// Sample format.
///
/// Unsupported sample formats are represented as Custom values.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SampleFormat {
    /// Unsigned 8-bit integer sample format.
    U8,
    /// Signed 8-bit integer sample format.
    I8,
    /// Signed big-endian 16-bit integer sample format.
    I16,
    /// Signed little-endian 16-bit integer sample format.
    I16LE,
    /// Signed big-endian 24-bit integer sample format.
    I24,
    /// Signed big-endian 32-bit integer sample format.
    I32,
    /// Signed little-endian 32-bit integer sample format.
    I32LE,
    /// Signed 32-bit floating point sample format.
    F32,
    /// Signed 64-bit floating point sample format.
    F64,
    /// Compressed μ-law sample format. Reading and writing samples should happen
    /// as signed 16-bit integers.
    CompressedUlaw,
    /// Compressed A-law sample format. Reading and writing samples should happen
    /// as signed 16-bit integers.
    CompressedAlaw,
    /// Compressed IMA ADPCM "ima4" sample format. Reading and writing samples should happen
    /// as signed 16-bit integers.
    CompressedIma4,
    /// Custom: unsupported compression type. The inner value is the the four byte name of
    /// the compression type. Samples can be read and written only as raw data.
    Custom([u8; 4])
}

impl SampleFormat {
    /// Returns the size of the decoded sample in bytes.
    /// Custom formats always return 0.
    pub fn decoded_size(&self) -> usize {
        match &self {
            SampleFormat::U8 => 1,
            SampleFormat::I8 => 1,
            SampleFormat::I16 | SampleFormat::I16LE => 2,
            SampleFormat::I24 => 3,
            SampleFormat::I32 | SampleFormat::I32LE => 4,
            SampleFormat::F32 => 4,
            SampleFormat::F64 => 8,
            SampleFormat::CompressedUlaw => 2,
            SampleFormat::CompressedAlaw => 2,
            SampleFormat::CompressedIma4 => 2,
            SampleFormat::Custom(_) => 0,
        }
    }

    /// Returns the size of the sample in the stream in bytes.
    /// CompressedIma4 returns 0. Custom formats return 1 (they are assumed to write single bytes).
    #[inline(always)]
    fn encoded_size(&self) -> u64 {
        match &self {
            SampleFormat::U8 => 1,
            SampleFormat::I8 => 1,
            SampleFormat::I16 | SampleFormat::I16LE => 2,
            SampleFormat::I24 => 3,
            SampleFormat::I32 | SampleFormat::I32LE => 4,
            SampleFormat::F32 => 4,
            SampleFormat::F64 => 8,
            SampleFormat::CompressedUlaw => 1,
            SampleFormat::CompressedAlaw => 1,
            SampleFormat::CompressedIma4 => 0,
            SampleFormat::Custom(_) => 1,
        }
    }

    /// Calculates the sample count based on byte length and sample format.
    fn calculate_sample_len(&self, sample_byte_len: u32) -> Option<u64> {
        match self {
            SampleFormat::CompressedIma4 => {
                // floor down to match macOS Audio Toolbox behavior
                Some(u64::from(sample_byte_len / 34) * 64)
            },
            SampleFormat::Custom(_) => None,
            _ => {
                // floor down to match macOS Audio Toolbox behavior
                Some(u64::from(sample_byte_len) / self.encoded_size())
            }
        }
    }

    /// The maximum channel count for the sample format.
    fn maximum_channel_count(&self) -> i16 {
        match self {
            SampleFormat::CompressedIma4 => 2,
            _ => i16::MAX
        }
    }

    /// Returns the COMM chunk's bits per sample value for the writer.
    /// Compressed formats and custom formats return 0.
    fn bits_per_sample(&self) -> u8 {
        match &self {
            SampleFormat::U8 => 8,
            SampleFormat::I8 => 8,
            SampleFormat::I16 => 16,
            SampleFormat::I16LE => 16,
            SampleFormat::I24 => 24,
            SampleFormat::I32 => 32,
            SampleFormat::I32LE => 32,
            SampleFormat::F32 => 32,
            SampleFormat::F64 => 64,
            SampleFormat::CompressedUlaw => 0,
            SampleFormat::CompressedAlaw => 0,
            SampleFormat::CompressedIma4 => 0,
            SampleFormat::Custom(_) => 0,
        }
    }
}

/// Checks if the given data is the start of AIFF or AIFF-C.
///
/// Only the first 12 bytes are checked. If the data length is less than 12 bytes,
/// then the result is always None.
///
/// # Examples
///
/// ```
/// match aifc::recognize(b"This is not an AIFF or AIFF-C") {
///     Some(aifc::FileFormat::Aiff) => { println!("It's AIFF"); },
///     Some(aifc::FileFormat::Aifc) => { println!("It's AIFF-C"); },
///     None => { println!("Not AIFF or AIFF-C"); },
/// }
/// ```
pub fn recognize(data: &[u8]) -> Option<FileFormat> {
    if data.len() < 12 ||
        data[0] != b'F' || data[1] != b'O' || data[2] != b'R' || data[3] != b'M' ||
        data[8] != b'A' || data[9] != b'I' || data[10] != b'F' {
        return None;
    }
    match data[11] {
        b'F' => Some(FileFormat::Aiff),
        b'C' => Some(FileFormat::Aifc),
        _ => None
    }
}

/// CountingWrite counts the bytes written to the underlying Write object.
struct CountingWrite<W> where W: Write {
    pub handle: W,
    pub bytes_written: u64
}

impl<W: Write> CountingWrite<W> {
    pub fn new(handle: W) -> CountingWrite<W> {
        CountingWrite {
            handle,
            bytes_written: 0
        }
    }

    /// Writes buf to the stream without counting them in bytes_written.
    fn write_not_counted(&mut self, buf: &[u8]) -> Result<usize, crate::aifcresult::AifcError> {
        self.handle.write_all(buf)?;
        Ok(buf.len())
    }
}

impl<W: Write> Write for CountingWrite<W> {
    fn write(&mut self, buf: &[u8]) -> Result<usize, std::io::Error> {
        self.handle.write_all(buf)?;
        self.bytes_written += u64::try_from(buf.len()).map_err(|_| crate::buffer_size_error())?;
        Ok(buf.len())
    }
    fn flush(&mut self) -> Result<(), std::io::Error> {
        self.handle.flush()
    }
}

fn is_even_u32(value: u32) -> bool {
    value & 1 == 0
}

fn is_even_u64(value: u64) -> bool {
    value & 1 == 0
}

fn is_even_usize(value: usize) -> bool {
    value & 1 == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recognize() {
        assert_eq!(recognize(&[]), None);
        assert_eq!(recognize(b"FORM"), None);
        assert_eq!(recognize(b"FORM....AIFX"), None);
        assert_eq!(recognize(b"form....AIFF"), None);
        assert_eq!(recognize(b"FORM....aiff"), None);
        assert_eq!(recognize(b"FORM....AIFF"), Some(FileFormat::Aiff));
        assert_eq!(recognize(b"FORM....AIFC"), Some(FileFormat::Aifc));
        assert_eq!(recognize(b"FORM....AIFFCOMM....blahblah.."), Some(FileFormat::Aiff));
    }
}
