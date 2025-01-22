use audio_codec_algorithms::AdpcmImaState;
use crate::{cast, AifcError, AifcResult, ChunkRef, ChunkId, FileFormat, Read,
    SampleFormat, Seek, SeekFrom};

fn to_i16(b0: u8, b1: u8) -> i16 {
    i16::from(b0) << 8 | i16::from(b1)
}
fn to_u32(b0: u8, b1: u8, b2: u8, b3: u8) -> u32 {
    u32::from(b0) << 24 | u32::from(b1) << 16 | u32::from(b2) << 8 | u32::from(b3)
}

/// Checked add for reading.
#[inline(always)]
fn rchecked_add(lhs: u64, rhs: u64) -> AifcResult<u64> {
    lhs.checked_add(rhs).ok_or(AifcError::ReadError)
}

/// Sample data.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Sample {
    /// Unsigned 8-bit integer sample.
    U8(u8),
    /// Signed 8-bit integer sample.
    I8(i8),
    /// Signed 16-bit integer sample.
    I16(i16),
    /// Signed 24-bit integer sample. The `i32` value is always in the range [-8388608, 8388607].
    I24(i32),
    /// Signed 32-bit integer sample.
    I32(i32),
    /// Signed 32-bit floating point sample.
    F32(f32),
    /// Signed 64-bit floating point sample.
    F64(f64)
}

/// Audio info returned by `AifcReader`.
#[derive(Debug, Clone, PartialEq)]
pub struct AifcReadInfo {

    /// File format: AIFF or AIFF-C.
    pub file_format: FileFormat,

    /// Number of channels. This is always greater than zero.
    pub channels: i16,

    /// Sample rate, samples per second.
    /// Note: this may be zero, negative, infinity or NaN.
    pub sample_rate: f64,

    /// Format of the samples. This is derived from `comm_sample_size` and `comm_compression_type`.
    pub sample_format: SampleFormat,

    /// Sample count. To get the frame count, divide by `channels`.
    /// The value is `None` for custom sample formats, because it isn't possible to calculate
    /// the sample count for them.
    pub sample_len: Option<u64>,

    /// Byte length of sample data (the SSND audio data size).
    pub sample_byte_len: u32,

    /// Number of audio frames. This is a value directly read from the COMM chunk.
    ///
    /// This value may differ from the actual audio frame count to be read, because
    /// the actual audio frame count is based on the SSND chunk size. So, it's usually better
    /// to use `sample_len` instead of this field.
    pub comm_num_sample_frames: u32,

    /// Sample size in bits. This is a value directly read from the COMM chunk.
    ///
    /// This value contains a valid sample size value (1-32) if `compression_type` is `"NONE"`.
    /// If `compression_type` is not `"NONE"`, then this field is usually 0.
    ///
    /// Note: The decoded sample size can be reliably determined for all supported sample formats
    /// by calling [`SampleFormat::decoded_size()`].
    pub comm_sample_size: i16,

    /// Compression type. This is a value directly read from the COMM chunk in AIFF-C files.
    /// AIFF doesn't have this field, so this is always `"NONE"` for [`FileFormat::Aiff`].
    pub comm_compression_type: [u8; 4],
}

/// The SampleRead trait allows for reading samples from a source.
trait SampleRead {
    /// Reads the next sample.
    fn read_sample_for_iter(&mut self) -> Option<AifcResult<Sample>>;
    /// Returns a tuple where the first and second elements are the remaining sample count.
    fn size_hint(&self) -> (usize, Option<usize>);
}

/// Iterator to read samples one by one.
///
/// Use [`AifcReader::samples()`] to create this iterator.
pub struct Samples<'a, R> {
    samples_left: u64,
    reader: &'a mut R
}

impl<'a, R> Samples<'a, R> {
    /// Creates a new sample iterator.
    fn new(samples_left: u64, reader: &'a mut R) -> Samples<'a, R> {
        Samples {
            samples_left,
            reader
        }
    }
}

/// Iterator implementation for samples.
impl<R: SampleRead> Iterator for Samples<'_, R> {
    type Item = AifcResult<Sample>;

    /// Reads the next sample.
    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        // manually count samples left so that this can't return infinite number of errors
        // (don't just rely that read_sample() will return None)
        if self.samples_left == 0 {
            return None;
        }
        self.samples_left -= 1;
        self.reader.read_sample_for_iter()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.reader.size_hint()
    }
}

/// The ChunkRead trait allows for reading chunk data from a source.
pub trait ChunkRead {
    fn read_exact(&mut self, buf: &mut [u8]) -> AifcResult<()>;
    fn seek(&mut self, pos: u64) -> AifcResult<()>;
}

/// Iterator to read chunks one by one.
///
/// The iterator returns [`ChunkRef`]s and the actual chunk data can be read
/// with [`Chunks::read_data()`].
///
/// See [`AifcReader::chunks()`] for an example.
pub struct Chunks<'a, R> {
    reader: &'a mut R,
    chunks_left: u32,
    next_chunk_pos: u64,
}

impl<'a, R: ChunkRead> Chunks<'a, R> {
    /// Creates a new chunk iterator.
    fn new(total_chunk_count: u32, reader: &'a mut R) -> Chunks<'a, R> {
        Chunks {
            reader,
            chunks_left: total_chunk_count,
            next_chunk_pos: 12,
        }
    }

    /// Reads the given chunk to `buf`. The `buf` length must match the chunk size.
    pub fn read_data(&mut self, chunkref: &ChunkRef, buf: &mut [u8]) -> AifcResult<()> {
        let buf_len_u32 = u32::try_from(buf.len()).map_err(|_| AifcError::InvalidParameter)?;
        if buf_len_u32 != chunkref.size {
            return Err(AifcError::InvalidParameter);
        }
        let data_pos = rchecked_add(chunkref.pos, 8)?;
        self.reader.seek(data_pos)?;
        self.reader.read_exact(buf)
    }
}

/// Iterator implementation for chunks.
impl<R: ChunkRead> Iterator for Chunks<'_, R> {
    type Item = AifcResult<ChunkRef>;

    /// Reads the next chunk.
    fn next(&mut self) -> Option<Self::Item> {
        // count chunks manually so that next() can't return infinite number of errors
        if self.chunks_left == 0 {
            return None;
        }
        self.chunks_left -= 1;
        let curpos = self.next_chunk_pos;
        match self.reader.seek(curpos) {
            Ok(()) => {},
            Err(e) => { return Some(Err(e)); }
        }
        let mut cid = [ 0u8; 4 ];
        match self.reader.read_exact(&mut cid) {
            Ok(()) => {},
            Err(e) => { return Some(Err(e)); }
        }
        let mut csize = [ 0u8; 4 ];
        match self.reader.read_exact(&mut csize) {
            Ok(()) => {},
            Err(e) => { return Some(Err(e)); }
        }
        let chunksize = u32::from_be_bytes(csize);
        let mut chunksize_u64 = u64::from(u32::from_be_bytes(csize)) + 8;
        if !crate::is_even_u64(chunksize_u64) {
            chunksize_u64 += 1;
        }
        self.next_chunk_pos = match curpos.checked_add(chunksize_u64) {
            Some(val) => val,
            None => { return Some(Err(AifcError::ReadError)); }
        };
        Some(Ok(ChunkRef { id: cid, pos: curpos, size: chunksize }))
    }
}

/// `SeekableRead` can be seeked forward for a stream implementing `Read` and
/// forward and backward for a stream implementing `Read+Seek`.
struct SeekableRead<R> {
    /// The underlying reader.
    stream: R,
    /// The current position from the absolute position 0.
    current_pos: u64,
    /// Initial position from the absolute position 0.
    initial_pos: u64,

    seek_fn: fn(&mut SeekableRead<R>, pos: SeekFrom) -> AifcResult<()>,
}

impl<R: Read> Read for SeekableRead<R> {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, std::io::Error> {
        match self.read_exact(buf) {
            Err(AifcError::StdIoError(e)) => { Err(e) },
            Err(_) => { Err(crate::read_error()) },
            Ok(()) => Ok(buf.len())
        }
    }
}

impl<R: Read> SeekableRead<R> {
    #[inline(always)]
    fn read_exact(&mut self, buf: &mut [u8]) -> AifcResult<()> {
        self.stream.read_exact(buf)?;
        self.current_pos =
            rchecked_add(self.current_pos, cast::usize_to_u64(buf.len(), AifcError::ReadError)?)?;
        Ok(())
    }
    fn seek(&mut self, pos: SeekFrom) -> AifcResult<()> {
        (self.seek_fn)(self, pos)
    }
}

/// AIFF / AIFF-C reader.
///
/// `AifcReader` takes a stream which implements `Read` or `Read+Seek`.
/// It can read audio sample data and any other chunk data.
///
/// When reading samples, channel data is interleaved. The stream can also be seeked to
/// a specific sample position.
///
/// Data reading is done on demand. The reader doesn't perform much buffering, so it's
/// recommended to use a buffered reader with it.
///
/// # Errors
///
/// If any of the methods returns an error, then the reader is in an undefined state and
/// shouldn't be used anymore.
///
/// # Examples
///
/// Reading audio info and samples:
///
/// ```no_run
/// # fn example() -> aifc::AifcResult<()> {
/// let mut rd = std::io::BufReader::new(std::fs::File::open("test.aiff")?);
/// let mut reader = aifc::AifcReader::new(&mut rd)?;
/// let info = reader.info();
/// for sample in reader.samples()? {
///     println!("Got sample {:?}", sample.expect("sample read error"));
/// }
/// # Ok(())
/// # }
/// ```
/// Reading sample data as raw bytes (useful for reading unsupported compression types).
/// The stream position is guaranteed to be at the first sample position after `AifcReader`
/// has been created.
///
/// ```no_run
/// # fn example() -> aifc::AifcResult<()> {
/// use std::io::Read;
/// let mut rd = std::io::BufReader::new(std::fs::File::open("test.aiff")?);
/// let mut reader = aifc::AifcReader::new(&mut rd)?;
/// let info = reader.info();
/// let mut ireader = reader.into_inner();
/// let size = usize::try_from(info.sample_byte_len).expect("size too large");
/// let mut raw_sample_data = vec![0u8; size];
/// ireader.read_exact(&mut raw_sample_data)?;
/// # Ok(())
/// # }
/// ```
pub struct AifcReader<R> {
    /// True if this is a single pass reader (created with new_single_pass()).
    is_single_pass: bool,
    /// The underlying reader.
    stream: SeekableRead<R>,
    /// Info read from the stream.
    info: AifcReadInfo,
    /// Sample start position in bytes.
    sample_byte_start_pos: u64,
    /// Sample read position in bytes.
    sample_byte_read_pos: u64,
    /// Sample read position.
    sample_read_pos: u64,

    // for multi-pass

    /// Flag to indicate that read_sample() must seek to sample_byte_read_pos,
    /// because chunk reading has seeked the stream.
    needs_to_seek_to_read_pos: bool,

    /// Total number of chunks in the stream. Zero for single-pass reader.
    total_chunk_count: u32,

    /// Marker chunk data.
    marker_chunkref: Option<ChunkRef>,
    /// Comments chunk data.
    comments_chunkref: Option<ChunkRef>,
    /// ID3 chunk data.
    id3_chunkref: Option<ChunkRef>,
    /// Compression name start position in bytes.
    comm_compression_name_start: u64,
    /// Compression name size in bytes.
    comm_compression_name_len: u8,

    // for ima4 decompression

    /// IMA ADPCM "ima4" stream byte position, which has been uncompressed to `sample_buffer`.
    /// `None` means that nothing has been uncompressed to `sample_buffer`.
    /// For mono, this points to 34 byte chunks in the audio data.
    /// For stereo, this points to 68 byte chunks in the audio data.
    ima4_sample_buffer_byte_pos: Option<u64>,
    /// Sample buffer stores samples for uncompressed ima4 data.
    /// It stores 64 samples for mono audio and 128 samples for stereo audio.
    ima4_sample_buffer: [i16; 128],
    /// IMA ADPCM states for up to 2 channels.
    ima4_state: [AdpcmImaState; 2],
}

impl<R: Read> AifcReader<R> {
    /// Creates a new single pass `AifcReader` for a stream implementing the `Read` trait.
    ///
    /// Header data is read immediately from the inner reader. After that, the stream position
    /// is at the start of the sample data.
    pub fn new_single_pass(stream: R) -> AifcResult<AifcReader<R>> {
        // SeekableRead which can only be seeked forward
        let mut sread = SeekableRead {
            stream,
            current_pos: 0,
            initial_pos: 0,
            seek_fn: |sr: &mut SeekableRead<R>, pos: SeekFrom| {
                let seek_size = match pos {
                    SeekFrom::Start(p) => {
                        if p < sr.current_pos {
                            // can't seek backwards, so return an error:
                            // single_pass reader should never seek the stream backwards
                            return Err(AifcError::SeekError);
                        }
                        p - sr.current_pos
                    },
                    // Current and End are not used
                    _ => 0
                };
                let mut skipbuf = [0u8; 1];
                for _ in 0..seek_size {
                    sr.stream.read_exact(&mut skipbuf)?;
                    sr.current_pos = sr.current_pos
                        .checked_add(1)
                        .ok_or(AifcError::SeekError)?;
                }
                Ok(())
            },
        };
        let hdr = read_header(&mut sread, true)?;
        Ok(AifcReader {
            is_single_pass: true,
            stream: sread,
            info: hdr.info,
            sample_byte_start_pos: hdr.sample_byte_start_pos,
            sample_byte_read_pos: hdr.sample_byte_start_pos,
            sample_read_pos: 0,
            needs_to_seek_to_read_pos: false,
            total_chunk_count: 0,
            marker_chunkref: hdr.marker_chunkref,
            comments_chunkref: hdr.comments_chunkref,
            id3_chunkref: hdr.id3_chunkref,
            comm_compression_name_start: hdr.comm_compression_name_start,
            comm_compression_name_len: hdr.comm_compression_name_len,
            ima4_sample_buffer_byte_pos: None,
            ima4_sample_buffer: [0i16; 128],
            ima4_state: [ AdpcmImaState::new(), AdpcmImaState::new() ]
        })
    }

    /// Returns a struct containing audio info, such as number of channels and sample rate.
    pub fn info(&self) -> AifcReadInfo {
        self.info.clone()
    }

    /// Reads one sample.
    ///
    /// Returns `None` when all samples have been read. Returns an error for unsupported encodings
    /// (`SampleFormat::Custom`) or if reading from the underlying reader fails.
    #[inline(always)]
    pub fn read_sample(&mut self) -> AifcResult<Option<Sample>> {
        if self.sample_read_pos >= self.info.sample_len.unwrap_or(0) {
            return Ok(None);
        }
        if self.needs_to_seek_to_read_pos {
            self.stream.seek(SeekFrom::Start(self.sample_byte_read_pos))?;
        }
        let sample;
        match self.info.sample_format {
            SampleFormat::U8 => {
                let mut buf = [0u8; 1];
                self.stream.read_exact(&mut buf)?;
                sample = Ok(Some(Sample::U8(buf[0])));
                self.sample_byte_read_pos = rchecked_add(self.sample_byte_read_pos,
                    self.info.sample_format.encoded_size())?;
            },
            SampleFormat::I8 => {
                let mut buf = [0u8; 1];
                self.stream.read_exact(&mut buf)?;
                sample = Ok(Some(Sample::I8(cast::u8_to_i8(buf[0]))));
                self.sample_byte_read_pos = rchecked_add(self.sample_byte_read_pos,
                    self.info.sample_format.encoded_size())?;
            },
            SampleFormat::I16 => {
                let mut buf = [0u8; 2];
                self.stream.read_exact(&mut buf)?;
                sample = Ok(Some(Sample::I16(i16::from_be_bytes(buf))));
                self.sample_byte_read_pos = rchecked_add(self.sample_byte_read_pos,
                    self.info.sample_format.encoded_size())?;
            },
            SampleFormat::I16LE => {
                let mut buf = [0u8; 2];
                self.stream.read_exact(&mut buf)?;
                sample = Ok(Some(Sample::I16(i16::from_le_bytes(buf))));
                self.sample_byte_read_pos = rchecked_add(self.sample_byte_read_pos,
                    self.info.sample_format.encoded_size())?;
            },
            SampleFormat::I24 => {
                let mut buf = [0u8; 3];
                self.stream.read_exact(&mut buf)?;
                let mut res = i32::from(buf[0]) << 16 | i32::from(buf[1]) << 8 | i32::from(buf[2]);
                if res >= 8388608 {
                    res -= 16777216;
                }
                sample = Ok(Some(Sample::I24(res)));
                self.sample_byte_read_pos = rchecked_add(self.sample_byte_read_pos,
                    self.info.sample_format.encoded_size())?;
            },
            SampleFormat::I32 => {
                let mut buf = [0u8; 4];
                self.stream.read_exact(&mut buf)?;
                sample = Ok(Some(Sample::I32(i32::from_be_bytes(buf))));
                self.sample_byte_read_pos = rchecked_add(self.sample_byte_read_pos,
                    self.info.sample_format.encoded_size())?;
            },
            SampleFormat::I32LE => {
                let mut buf = [0u8; 4];
                self.stream.read_exact(&mut buf)?;
                sample = Ok(Some(Sample::I32(i32::from_le_bytes(buf))));
                self.sample_byte_read_pos = rchecked_add(self.sample_byte_read_pos,
                    self.info.sample_format.encoded_size())?;
            },
            SampleFormat::F32 => {
                let mut buf = [0u8; 4];
                self.stream.read_exact(&mut buf)?;
                sample = Ok(Some(Sample::F32(f32::from_be_bytes(buf))));
                self.sample_byte_read_pos = rchecked_add(self.sample_byte_read_pos,
                    self.info.sample_format.encoded_size())?;
            },
            SampleFormat::F64 => {
                let mut buf = [0u8; 8];
                self.stream.read_exact(&mut buf)?;
                sample = Ok(Some(Sample::F64(f64::from_be_bytes(buf))));
                self.sample_byte_read_pos = rchecked_add(self.sample_byte_read_pos,
                    self.info.sample_format.encoded_size())?;
            },
            SampleFormat::CompressedUlaw => {
                let mut buf = [0u8; 1];
                self.stream.read_exact(&mut buf)?;
                let val = buf[0];
                sample = Ok(Some(Sample::I16(audio_codec_algorithms::decode_ulaw(val))));
                self.sample_byte_read_pos = rchecked_add(self.sample_byte_read_pos,
                    self.info.sample_format.encoded_size())?;
            },
            SampleFormat::CompressedAlaw => {
                let mut buf = [0u8; 1];
                self.stream.read_exact(&mut buf)?;
                let val = buf[0];
                sample = Ok(Some(Sample::I16(audio_codec_algorithms::decode_alaw(val))));
                self.sample_byte_read_pos = rchecked_add(self.sample_byte_read_pos,
                    self.info.sample_format.encoded_size())?;
            },
            SampleFormat::CompressedIma4 => {
                let channels_u64 = cast::clamp_i16_to_u64(self.info.channels);
                let sample_buffer_len = 64 * channels_u64;
                let byte_pos = (self.sample_read_pos / sample_buffer_len) * (34*channels_u64);
                let byte_pos = rchecked_add(byte_pos, self.sample_byte_start_pos)?;
                // uncompress packets if the current sample buffer's stream position doesn't
                // match the new byte_pos
                if self.ima4_sample_buffer_byte_pos != Some(byte_pos) {
                    // uncompress samples for all channels
                    let channels_usize = cast::clamp_i16_to_usize(self.info.channels);
                    for ch in 0..channels_usize {
                        let mut data = [0u8; 34];
                        self.stream.read_exact(&mut data)?;
                        let mut sample_buffer = [0i16; 64];
                        audio_codec_algorithms::decode_adpcm_ima_ima4(&data,
                            &mut self.ima4_state[ch], &mut sample_buffer);
                        for (i, d) in sample_buffer.iter().enumerate() {
                            self.ima4_sample_buffer[i*channels_usize + ch] = *d;
                        }
                    }
                    self.ima4_sample_buffer_byte_pos = Some(byte_pos);
                    self.sample_byte_read_pos = rchecked_add(byte_pos, 34 * channels_u64)?;
                }
                // read sample from the sample buffer
                let sample_buffer_pos = usize::try_from(self.sample_read_pos % sample_buffer_len)
                    .unwrap_or(0);
                sample = Ok(Some(Sample::I16(self.ima4_sample_buffer[sample_buffer_pos])));
            },
            SampleFormat::Custom(_) => {
                return Err(AifcError::Unsupported);
            }
        }
        self.sample_read_pos += 1;
        sample
    }

    /// Returns an iterator for samples. Returns an error for unsupported sample formats
    /// ([`SampleFormat::Custom`]).
    pub fn samples(&mut self) -> AifcResult<Samples<'_, AifcReader<R>>> {
        if let SampleFormat::Custom(_) = self.info.sample_format {
            return Err(AifcError::Unsupported);
        }
        let Some(slen) = self.info.sample_len else {
            return Err(AifcError::Unsupported);
        };
        // defensive check, slen should always be greater than or equal to sample_read_pos
        if slen < self.sample_read_pos {
            return Err(AifcError::ReadError);
        }
        let samples_left = slen - self.sample_read_pos;
        Ok(Samples::new(samples_left, self))
    }

    /// Consumes this `AifcReader` and returns the underlying reader.
    pub fn into_inner(self) -> R {
        self.stream.stream
    }

    /// Gets a reference to the underlying reader.
    ///
    /// It is not recommended to directly read from the underlying reader
    /// as it may corrupt this reader's state.
    pub const fn get_ref(&self) -> &R {
        &self.stream.stream
    }

    /// Gets a mutable reference to the underlying reader.
    ///
    /// Care should be taken to avoid modifying the internal I/O state of the
    /// underlying reader as it may corrupt this reader's state.
    pub fn get_mut(&mut self) -> &mut R {
        &mut self.stream.stream
    }
}

impl<R: Read> SampleRead for AifcReader<R> {
    /// Reads the next sample.
    #[inline(always)]
    fn read_sample_for_iter(&mut self) -> Option<AifcResult<Sample>> {
        match self.read_sample() {
            Ok(Some(s)) => Some(Ok(s)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }

    /// Returns a tuple where the first and second elements are the remaining sample count.
    fn size_hint(&self) -> (usize, Option<usize>) {
        // get the lower and upper bound of the sample count
        let Some(slen) = self.info.sample_len else {
            return (0, None);
        };
        // defensive check, slen should always be greater than or equal to sample_read_pos
        if slen < self.sample_read_pos {
            return (0, None);
        }
        let size64 = slen - self.sample_read_pos;
        if let Ok(ss) = usize::try_from(size64) {
            (ss, Some(ss))
        } else { // sample count is greater than usize::MAX
            (usize::MAX, None)
        }
    }
}

impl<R: Read + Seek> ChunkRead for AifcReader<R> {
    fn read_exact(&mut self, buf: &mut [u8]) -> AifcResult<()> {
        self.stream.read_exact(buf)
    }

    fn seek(&mut self, pos: u64) -> AifcResult<()> {
        self.stream.seek(SeekFrom::Start(pos))
    }
}

impl<R: Read + Seek> AifcReader<R> {
    /// Creates a new `AifcReader` for a stream implementing the `Read+Seek` traits.
    /// The stream will be seeked forwards and backwards to read data from it.
    ///
    /// Header data is read immediately from the inner reader. After that, the stream position
    /// is at the start of the sample data.
    pub fn new(mut stream: R) -> AifcResult<AifcReader<R>> {
        let ipos = stream.stream_position()?;
        // SeekableRead which can be seeked forwards and backwards
        let mut sread = SeekableRead {
            stream,
            current_pos: ipos,
            initial_pos: ipos,
            seek_fn: |sr: &mut SeekableRead<R>, pos: SeekFrom| {
                match pos {
                    SeekFrom::Start(rel_pos) => {
                        let abs_pos = rchecked_add(rel_pos, sr.initial_pos)?;
                        // optimization to not seek if the stream is already at the correct position
                        if abs_pos == sr.current_pos {
                            return Ok(());
                        }
                        sr.current_pos = sr.stream.seek(SeekFrom::Start(abs_pos))?;
                    },
                    SeekFrom::Current(rel_pos) => {
                        let abs_pos = rel_pos.checked_add_unsigned(sr.initial_pos)
                            .ok_or(AifcError::ReadError)?;
                        sr.current_pos = sr.stream.seek(SeekFrom::Current(abs_pos))?;
                    },
                    SeekFrom::End(rel_pos) => {
                        let abs_pos = rel_pos.checked_add_unsigned(sr.initial_pos)
                            .ok_or(AifcError::ReadError)?;
                        sr.current_pos = sr.stream.seek(SeekFrom::End(abs_pos))?;
                    }
                }
                Ok(())
            },
        };
        let hdr = read_header(&mut sread, false)?;
        Ok(AifcReader {
            is_single_pass: false,
            stream: sread,
            info: hdr.info,
            sample_byte_start_pos: hdr.sample_byte_start_pos,
            sample_byte_read_pos: hdr.sample_byte_start_pos,
            sample_read_pos: 0,
            needs_to_seek_to_read_pos: false,
            total_chunk_count: hdr.total_chunk_count,
            marker_chunkref: hdr.marker_chunkref,
            comments_chunkref: hdr.comments_chunkref,
            id3_chunkref: hdr.id3_chunkref,
            comm_compression_name_start: hdr.comm_compression_name_start,
            comm_compression_name_len: hdr.comm_compression_name_len,
            ima4_sample_buffer_byte_pos: None,
            ima4_sample_buffer: [0i16; 128],
            ima4_state: [ AdpcmImaState::new(), AdpcmImaState::new() ]
        })
    }

    /// Seeks the stream to the given sample position.
    /// Returns an error if trying to seek past the maximum sample position or
    /// if trying to seek a custom sample format.
    pub fn seek(&mut self, sample_position: u64) -> AifcResult<()> {
        if self.is_single_pass {
            return Err(AifcError::SeekError);
        }
        let Some(sample_len) = self.info.sample_len else {
            return Err(AifcError::Unsupported);
        };
        if sample_position > sample_len {
            return Err(AifcError::SeekError);
        }
        match self.info.sample_format {
            SampleFormat::CompressedIma4 => {
                // calculate new sample buffer byte pos and if it has changed, set
                // ima4_sample_buffer_byte_pos to None so that sample reading fills sample buffers
                let channels_u64 = cast::clamp_i16_to_u64(self.info.channels);
                let new_byte_pos = (sample_position / (64*channels_u64)) * (34*channels_u64);
                let Some(new_byte_pos) = new_byte_pos
                    .checked_add(self.sample_byte_start_pos) else {
                    return Err(AifcError::SeekError);
                };
                if self.ima4_sample_buffer_byte_pos != Some(new_byte_pos) {
                    self.sample_byte_read_pos = new_byte_pos;
                    self.stream.seek(SeekFrom::Start(new_byte_pos))?;
                    self.ima4_sample_buffer_byte_pos = None;
                    self.ima4_state[0] = AdpcmImaState::new();
                    self.ima4_state[1] = AdpcmImaState::new();
                }
                self.sample_read_pos = sample_position;
            },
            SampleFormat::Custom(_) => {
                return Err(AifcError::Unsupported);
            },
            _ => {
                let sample_size = self.info.sample_format.encoded_size();
                let Some(new_byte_pos) = sample_position
                    .checked_mul(sample_size)
                    .and_then(|val| val.checked_add(self.sample_byte_start_pos)) else {
                    return Err(AifcError::SeekError);
                };
                self.stream.seek(SeekFrom::Start(new_byte_pos))?;
                self.sample_byte_read_pos = new_byte_pos;
                self.sample_read_pos = sample_position;
            }
        }
        Ok(())
    }

    /// Reads the compression name to the given `buf`.
    /// Returns the compression name length, which is always less than 256.
    ///
    /// If an empty `buf` is passed in, then nothing is read and only the compression name length
    /// is returned.
    /// If a non-empty `buf` is passed in, then the length of `buf` must exactly match
    /// the compression name length.
    ///
    /// AIFF files always return 0, because they don't have the compression name field.
    /// AIFF-C files may also return 0 if the compression name field is empty.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # fn example() -> aifc::AifcResult<()> {
    /// # let mut rd = std::io::BufReader::new(std::fs::File::open("test.aiff")?);
    /// # let mut aifcreader = aifc::AifcReader::new(&mut rd)?;
    /// let clen = aifcreader.read_compression_name(&mut [])?;
    /// let mut cname = vec![0u8; clen];
    /// aifcreader.read_compression_name(&mut cname)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn read_compression_name(&mut self, buf: &mut [u8]) -> AifcResult<usize> {
        if self.is_single_pass {
            return Err(AifcError::SeekError);
        }
        let clen = usize::from(self.comm_compression_name_len);
        if buf.is_empty() || clen == 0 {
            return Ok(clen);
        }
        self.needs_to_seek_to_read_pos = true;
        self.stream.seek(SeekFrom::Start(self.comm_compression_name_start))?;
        self.stream.read_exact(&mut buf[0..clen])?;
        Ok(clen)
    }

    /// Returns an iterator for reading chunks. All chunks under the FORM chunk are read
    /// from the start to the end.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # fn example() -> aifc::AifcResult<()> {
    /// # let mut rd = std::io::BufReader::new(std::fs::File::open("test.aiff").expect("open failed"));
    /// # let mut aifcreader = aifc::AifcReader::new(&mut rd)?;
    /// let mut chunks = aifcreader.chunks()?;
    /// while let Some(chdata) = chunks.next() {
    ///     let chdata = chdata?;
    ///     match &chdata.id {
    ///         &aifc::CHUNKID_NAME => {
    ///             let mut buf = vec![0u8; usize::try_from(chdata.size).expect("Chunk too large")];
    ///             chunks.read_data(&chdata, &mut buf)?;
    ///         },
    ///         _ => {}
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn chunks(&mut self) -> AifcResult<Chunks<'_, AifcReader<R>>> {
        if self.is_single_pass {
            return Err(AifcError::SeekError);
        }
        self.needs_to_seek_to_read_pos = true;
        Ok(Chunks::new(self.total_chunk_count, self))
    }

    /// Reads the MARK chunk data to the given `buf` and returns the MARK chunk size if
    /// the MARK chunk exists in the stream.
    /// Returns `None` if the MARK chunk doesn't exist.
    /// Returns an error if a reading error happens.
    ///
    /// If an empty `buf` is passed in, then nothing is read and only the MARK chunk size
    /// is returned.
    /// If a non-empty `buf` is passed in, then the size of `buf` must exactly match
    /// the MARK chunk size.
    ///
    /// This method reads the chunk by seeking to it and by reading the required bytes.
    /// The entire stream won't be scanned for the chunk, which makes this more efficient than
    /// reading the chunk with `chunks()`.
    ///
    /// The bytes read into `buf` can be parsed with [`Markers`](crate::Markers).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # fn example() -> aifc::AifcResult<()> {
    /// # let mut rd = std::io::BufReader::new(std::fs::File::open("test.aiff")?);
    /// # let mut aifcreader = aifc::AifcReader::new(&mut rd)?;
    /// if let Some(size) = aifcreader.read_chunk_markers(&mut [])? {
    ///     let mut markers_buf = vec![0; size];
    ///     aifcreader.read_chunk_markers(&mut markers_buf)?;
    ///     // parse markers from the byte buffer
    ///     let markers: Vec<aifc::AifcResult<aifc::Marker>> = aifc::Markers::new(&markers_buf)?
    ///         .collect();
    ///     println!("Markers: {:?}", markers);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn read_chunk_markers(&mut self, buf: &mut [u8]) -> AifcResult<Option<usize>> {
        if self.is_single_pass {
            return Err(AifcError::SeekError);
        }
        let Some(chunkref) = self.marker_chunkref.clone() else {
            return Ok(None);
        };
        self.read_chunk_data(&chunkref, buf)
    }

    /// Reads the COMT chunk data to the given `buf` and returns the COMT chunk size if
    /// the COMT chunk exists in the stream.
    /// Returns `None` if the COMT chunk doesn't exist.
    /// Returns an error if a reading error happens.
    ///
    /// If an empty `buf` is passed in, then nothing is read and only the COMT chunk size
    /// is returned.
    /// If a non-empty `buf` is passed in, then the size of `buf` must exactly match
    /// the COMT chunk size.
    ///
    /// This method reads the chunk by seeking to it and by reading the required bytes.
    /// The entire stream won't be scanned for the chunk, which makes this more efficient than
    /// reading the chunk with `chunks()`.
    ///
    /// The bytes read into `buf` can be parsed with [`Comments`](crate::Comments).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # fn example() -> aifc::AifcResult<()> {
    /// # let mut rd = std::io::BufReader::new(std::fs::File::open("test.aiff")?);
    /// # let mut aifcreader = aifc::AifcReader::new(&mut rd)?;
    /// if let Some(size) = aifcreader.read_chunk_comments(&mut [])? {
    ///     let mut comments_buf = vec![0; size];
    ///     aifcreader.read_chunk_comments(&mut comments_buf)?;
    ///     // parse comments from the byte buffer
    ///     let comments: Vec<aifc::AifcResult<aifc::Comment>> = aifc::Comments::new(&comments_buf)?
    ///         .collect();
    ///     println!("Comments: {:?}", comments);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn read_chunk_comments(&mut self, buf: &mut [u8]) -> AifcResult<Option<usize>> {
        if self.is_single_pass {
            return Err(AifcError::SeekError);
        }
        let Some(chunkref) = self.comments_chunkref.clone() else {
            return Ok(None);
        };
        self.read_chunk_data(&chunkref, buf)
    }

    /// Reads the ID3 chunk data to the given `buf` and returns the ID3 chunk size if
    /// the ID3 chunk exists in the stream.
    /// Returns `None` if the ID3 chunk doesn't exist.
    /// Returns an error if a reading error happens.
    ///
    /// If an empty `buf` is passed in, then nothing is read and only the ID3 chunk size
    /// is returned.
    /// If a non-empty `buf` is passed in, then the size of `buf` must exactly match
    /// the ID3 chunk size.
    ///
    /// This method reads the chunk by seeking to it and by reading the required bytes.
    /// The entire stream won't be scanned for the chunk, which makes this more efficient than
    /// reading the chunk with `chunks()`.
    ///
    /// The bytes read into `buf` can be parsed using external ID3 parsers.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # fn example() -> aifc::AifcResult<()> {
    /// # let mut rd = std::io::BufReader::new(std::fs::File::open("test.aiff")?);
    /// # let mut aifcreader = aifc::AifcReader::new(&mut rd)?;
    /// if let Some(size) = aifcreader.read_chunk_id3(&mut [])? {
    ///     let mut id3_buf = vec![0; size];
    ///     aifcreader.read_chunk_id3(&mut id3_buf)?;
    ///     // parse id3 tags from id3_buf using a third-party id3 parser..
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn read_chunk_id3(&mut self, buf: &mut [u8]) -> AifcResult<Option<usize>> {
        if self.is_single_pass {
            return Err(AifcError::SeekError);
        }
        let Some(chunkref) = self.id3_chunkref.clone() else {
            return Ok(None);
        };
        self.read_chunk_data(&chunkref, buf)
    }

    /// Reads byte data for ChunkRef to the given `buf`.
    fn read_chunk_data(&mut self, chunkref: &ChunkRef, buf: &mut [u8])
        -> AifcResult<Option<usize>> {
        let size = usize::try_from(chunkref.size).map_err(|_| AifcError::SizeTooLarge)?;
        if buf.is_empty() {
            return Ok(Some(size));
        }
        if buf.len() != size {
            return Err(AifcError::InvalidParameter);
        }
        self.needs_to_seek_to_read_pos = true;
        let data_pos = rchecked_add(chunkref.pos, 8)?;
        self.stream.seek(SeekFrom::Start(data_pos))?;
        self.stream.read_exact(buf)?;
        Ok(Some(size))
    }
}

struct HeaderData {
    info: AifcReadInfo,
    sample_byte_start_pos: u64,
    marker_chunkref: Option<ChunkRef>,
    comments_chunkref: Option<ChunkRef>,
    id3_chunkref: Option<ChunkRef>,
    total_chunk_count: u32,
    comm_compression_name_start: u64,
    comm_compression_name_len: u8
}

/// Reads info from the data stream. The data is validated and an error is
/// returned if the stream contains invalid data.
/// For instance, zero or negative number of channels returns an error.
fn read_header<R: Read>(stream: &mut SeekableRead<R>, is_single_pass: bool)
    -> AifcResult<HeaderData> {
    let (formid, formsize) = read_chunkid_and_size(stream)?;
    if formid != crate::CHUNKID_FORM {
        return Err(AifcError::UnrecognizedFormat);
    }
    let aiffid = read_chunkid(stream)?;
    let file_format = match aiffid {
        crate::CHUNKID_AIFF => FileFormat::Aiff,
        crate::CHUNKID_AIFC => FileFormat::Aifc,
        _ => { return Err(AifcError::UnrecognizedFormat); }
    };
    let mut pos: u64 = 12;
    let mut sample_byte_len: u32 = 0;
    let mut comm_compression_name_start = 0;
    let mut comm_compression_name_len = 0;
    let mut sample_byte_start_pos = 0;
    let mut total_chunk_count = 0;
    let mut marker_chunkref = None;
    let mut comments_chunkref = None;
    let mut id3_chunkref = None;
    let mut info: Option<AifcReadInfo> = None;
    let max_pos = rchecked_add(stream.initial_pos, u64::from(formsize))?;
    while stream.current_pos < max_pos {
        let (chid, chsize) = read_chunkid_and_size(stream)?;
        if !is_single_pass {
            total_chunk_count += 1;
            let cd = ChunkRef {
                id: chid,
                pos,
                size: chsize
            };
            match chid {
                crate::CHUNKID_MARK => {
                    // store a reference to the first MARK chunk
                    if marker_chunkref.is_none() {
                        marker_chunkref = Some(cd.clone());
                    }
                },
                crate::CHUNKID_COMT => {
                    // store a reference to the first COMT chunk
                    if comments_chunkref.is_none() {
                        comments_chunkref = Some(cd.clone());
                    }
                },
                crate::CHUNKID_ID3 => {
                    // store a reference to the last ID3 chunk
                    id3_chunkref = Some(cd.clone());
                },
                _ => {}
            }
        }

        if chid == crate::CHUNKID_COMM {
            let (arinfo, cstart, clen) = read_comm_chunk(stream, chsize, file_format, pos)?;
            info = Some(arinfo);
            comm_compression_name_start = cstart;
            comm_compression_name_len = clen;

        } else if chid == crate::CHUNKID_SSND {
            if chsize < 8 {
                return Err(AifcError::StdIoError(crate::unexpectedeof()));
            }
            let mut offset = [0u8; 4];
            stream.read_exact(&mut offset)?;
            let mut start_offset = u32::from_be_bytes(offset);
            // start_offset can't be greater than chunk size
            start_offset = start_offset.min(chsize - 8);
            // note: the SSND blocksize value is not relevant for reading samples
            let mut blocksize_ignored = [0u8; 4];
            stream.read_exact(&mut blocksize_ignored)?;
            let sample_start_pos = rchecked_add(pos, u64::from(start_offset)+8+8)?;
            sample_byte_len = chsize - 8 - start_offset;
            sample_byte_start_pos = sample_start_pos;
            if is_single_pass {
                // check that COMM has been read - otherwise can't read samples
                if info.is_none() {
                    return Err(AifcError::CommChunkNotFoundBeforeSsndChunk);
                }
                break; // break out of loop, ready to read samples
            }
        }
        pos = rchecked_add(pos, u64::from(chsize) + 8)?;
        if !crate::is_even_u32(chsize) { // odd chunk size has one pad byte
            pos = rchecked_add(pos, 1)?;
        }
        stream.seek(SeekFrom::Start(pos))?;
    }
    if let Some(i) = &mut info {
        i.sample_byte_len = sample_byte_len;
        i.sample_len = i.sample_format.calculate_sample_len(sample_byte_len);
    }
    if sample_byte_start_pos > 0 {
        stream.seek(SeekFrom::Start(sample_byte_start_pos))?;
    }
    match &info {
        Some(i) => {
            Ok(HeaderData {
                info: i.clone(),
                sample_byte_start_pos,
                marker_chunkref,
                comments_chunkref,
                id3_chunkref,
                total_chunk_count,
                comm_compression_name_start,
                comm_compression_name_len
            })
        },
        None => { Err(AifcError::InvalidCommChunk) }
    }
}

// Reads the COMM chunk and returns a tuple containing
// aiff info, compression name start position and compression name length in bytes.
fn read_comm_chunk(handle: &mut dyn Read, comm_chsize: u32, file_format: FileFormat, pos: u64)
    -> AifcResult<(AifcReadInfo, u64, u8)> {
    if comm_chsize < 18 {
        return Err(AifcError::InvalidCommChunk);
    }
    let mut commdata = [0u8; 18];
    handle.read_exact(&mut commdata)?;
    let num_channels = to_i16(commdata[0], commdata[1]);
    if num_channels <= 0 {
        return Err(AifcError::InvalidNumberOfChannels);
    }
    let num_sample_frames = to_u32(commdata[2], commdata[3], commdata[4], commdata[5]);
    let comm_sample_size = to_i16(commdata[6], commdata[7]);
    let sample_rate: f64 = crate::f80::f80_to_f64(&commdata[8..18]);
    if file_format == FileFormat::Aiff {
        let sample_format = match comm_sample_size {
            1..=8 => SampleFormat::I8,
            9..=16 => SampleFormat::I16,
            17..=24 => SampleFormat::I24,
            25..=32 => SampleFormat::I32,
            _ => { return Err(AifcError::InvalidSampleSize); }
        };
        let info = AifcReadInfo {
            file_format,
            channels: num_channels,
            sample_rate,
            sample_format,
            sample_byte_len: 0,
            sample_len: None,
            comm_num_sample_frames: num_sample_frames,
            comm_sample_size,
            comm_compression_type: crate::COMPRESSIONTYPE_NONE,
        };
        Ok((info, 0, 0))

    } else { // FileFormat::Aifc
        if comm_chsize < 23 {
            return Err(AifcError::InvalidCommChunk);
        }
        let mut compression_type = [0u8; 4];
        handle.read_exact(&mut compression_type)?;
        let comm_compression_name_start = rchecked_add(pos, 8 + 23)?;
        let mut comm_compr_name_len_buf = [0u8; 1];
        handle.read_exact(&mut comm_compr_name_len_buf)?;
        let mut comm_compression_name_len = comm_compr_name_len_buf[0];
        // clamp compression name len to be at most the chunk size
        let chunksize = u8::try_from(comm_chsize - 23).unwrap_or(u8::MAX);
        comm_compression_name_len = comm_compression_name_len.min(chunksize);
        skip_bytes(handle, usize::from(comm_compression_name_len))?;
        // derive sample format for known compression type ids
        let sample_format = match compression_type {
            crate::COMPRESSIONTYPE_NONE => match comm_sample_size {
                1..=8 => SampleFormat::I8,
                9..=16 => SampleFormat::I16,
                17..=24 => SampleFormat::I24,
                25..=32 => SampleFormat::I32,
                _ => { return Err(AifcError::InvalidSampleSize); }
            },
            crate::COMPRESSIONTYPE_RAW => SampleFormat::U8,
            crate::COMPRESSIONTYPE_TWOS => SampleFormat::I16,
            crate::COMPRESSIONTYPE_SOWT => SampleFormat::I16LE,
            crate::COMPRESSIONTYPE_IN24 => SampleFormat::I24,
            crate::COMPRESSIONTYPE_IN32 => SampleFormat::I32,
            crate::COMPRESSIONTYPE_23NI => SampleFormat::I32LE,
            crate::COMPRESSIONTYPE_FL32_UPPER => SampleFormat::F32,
            crate::COMPRESSIONTYPE_FL32 => SampleFormat::F32,
            crate::COMPRESSIONTYPE_FL64_UPPER => SampleFormat::F64,
            crate::COMPRESSIONTYPE_FL64 => SampleFormat::F64,
            crate::COMPRESSIONTYPE_ULAW => SampleFormat::CompressedUlaw,
            crate::COMPRESSIONTYPE_ULAW_UPPER => SampleFormat::CompressedUlaw,
            crate::COMPRESSIONTYPE_ALAW => SampleFormat::CompressedAlaw,
            crate::COMPRESSIONTYPE_ALAW_UPPER => SampleFormat::CompressedAlaw,
            crate::COMPRESSIONTYPE_IMA4 => SampleFormat::CompressedIma4,
            _ => SampleFormat::Custom(compression_type)
        };
        if num_channels > sample_format.maximum_channel_count() {
            return Err(AifcError::InvalidNumberOfChannels);
        }
        let info = AifcReadInfo {
            file_format,
            channels: num_channels,
            sample_rate,
            sample_format,
            sample_byte_len: 0,
            sample_len: None,
            comm_num_sample_frames: num_sample_frames,
            comm_sample_size,
            comm_compression_type: compression_type,
        };
        Ok((info, comm_compression_name_start, comm_compression_name_len))
    }
}

// Skips the given size number of bytes by reading them.
fn skip_bytes(handle: &mut dyn Read, size: usize) -> AifcResult<()> {
    let mut skipbuf = [0u8; 1];
    for _ in 0..size {
        handle.read_exact(&mut skipbuf)?;
    }
    Ok(())
}

fn read_chunkid(handle: &mut dyn Read) -> AifcResult<ChunkId> {
    let mut chdata = [0u8; 4];
    handle.read_exact(&mut chdata)?;
    Ok(chdata)
}

fn read_chunkid_and_size(handle: &mut dyn Read) -> AifcResult<(ChunkId, u32)> {
    let mut chdata = [0u8; 8];
    handle.read_exact(&mut chdata)?;
    let chid = [ chdata[0], chdata[1], chdata[2], chdata[3] ];
    let chlen = to_u32(chdata[4], chdata[5], chdata[6], chdata[7]);
    Ok((chid, chlen))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// Creates aifc from the given parameters.
    fn create_aifc(bits_per_sample: u8, channels: u8, compression_type: &[u8; 4],
        sample_bytes: &[u8]) -> Vec<u8> {
        let sample_len = u8::try_from(sample_bytes.len()).expect("too many samples");
        let mut aifc = vec![
            b'F', b'O', b'R', b'M', 0, 0, 0, 64+sample_len, b'A', b'I', b'F', b'C',
            b'F', b'V', b'E', b'R', 0, 0, 0, 4, 0xA2, 0x80, 0x51, 0x40,
            b'C', b'O', b'M', b'M', 0, 0, 0, 24,   0, channels,    // num_channels
            0, 0, 0, sample_len/(bits_per_sample/8),        // num_sample_frames
            0, bits_per_sample,                             // sample_size
            0x40, 0x0E, 0xAC, 0x44, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // sample_rate
            compression_type[0], compression_type[1], compression_type[2], compression_type[3],
            0, 0,
            b'S', b'S', b'N', b'D', 0, 0, 0, 8+sample_len,  0, 0, 0, 0,  0, 0, 0, 0,
        ];
        aifc.extend_from_slice(sample_bytes);
        aifc
    }

    /// Reads len number of bytes from the stream.
    fn read_dummy_data(stream: &mut impl Read, len: usize) -> AifcResult<()> {
        let mut init_buf = vec![0u8; len];
        Ok(stream.read_exact(&mut init_buf)?)
    }

    #[test]
    fn test_to_i16() {
        assert_eq!(to_i16(0x41, 0x65), 0x4165);
        assert_eq!(to_i16(0x81, 0x65), -32411);
    }

    #[test]
    fn test_to_u32() {
        assert_eq!(to_u32(0x41, 0x65, 0x73, 0x21), 0x41657321);
    }

    #[test]
    fn test_new_transferring_ownership() -> AifcResult<()> {
        let aifc = create_aifc(8, 1, b"NONE", &[ 11, 12, 13, 14 ]);
        let cursor = Cursor::new(&aifc);
        let reader = AifcReader::new(cursor)?;
        let info = reader.info();
        assert_eq!(info.sample_format, SampleFormat::I8);
        Ok(())
    }

    #[test]
    fn test_new_with_mut_ref() -> AifcResult<()> {
        let aifc = create_aifc(8, 1, b"NONE", &[ 11, 12, 13, 14 ]);
        let mut cursor = Cursor::new(&aifc);
        let reader = AifcReader::new(&mut cursor)?;
        let info = reader.info();
        assert_eq!(info.sample_format, SampleFormat::I8);
        Ok(())
    }

    #[test]
    fn test_new_single_pass_with_ref_slice() -> AifcResult<()> {
        let aifc = create_aifc(8, 1, b"NONE", &[ 11, 12, 13, 14 ]);
        let mut slice: &[u8] = aifc.as_ref();
        let mut reader = AifcReader::new_single_pass(&mut slice)?;
        let info = reader.info();
        assert_eq!(info.file_format, FileFormat::Aifc);
        assert_eq!(info.sample_format, SampleFormat::I8);
        assert_eq!(reader.read_sample()?, Some(Sample::I8(11)));
        assert_eq!(reader.read_sample()?, Some(Sample::I8(12)));
        Ok(())
    }

    #[test]
    fn test_single_pass_with_ssnd_before_comm_is_error() -> AifcResult<()> {
        let aifc = vec![
            b'F', b'O', b'R', b'M', 0, 0, 0, 72, b'A', b'I', b'F', b'C',
            b'S', b'S', b'N', b'D', 0, 0, 0, 16,  0, 0, 0, 0,  0, 0, 0, 0,
            1, 2, 3, 4, 5, 6, 7, 8,
            b'F', b'V', b'E', b'R', 0, 0, 0, 4, 0xA2, 0x80, 0x51, 0x40,
            b'C', b'O', b'M', b'M', 0, 0, 0, 24,   0, 1,        // num_channels
            0, 0, 0, 8,                                         // num_sample_frames
            0, 8,                                               // sample_size
            0x40, 0x0E, 0xAC, 0x44, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // sample_rate
            b'N', b'O', b'N', b'E',
            0, 0,    // compression name
        ];
        let mut slice: &[u8] = aifc.as_ref();
        assert!(matches!(AifcReader::new_single_pass(&mut slice),
            Err(AifcError::CommChunkNotFoundBeforeSsndChunk)));
        Ok(())
    }

    #[test]
    fn test_into_inner() -> AifcResult<()> {
        let aifc = create_aifc(8, 1, b"NONE", &[ 11, 12, 13, 14 ]);
        let cursor = Cursor::new(&aifc);
        let mut reader = AifcReader::new(cursor)?;
        let info = reader.info();
        assert_eq!(info.sample_format, SampleFormat::I8);
        assert_eq!(reader.read_sample()?, Some(Sample::I8(11)));
        let mut inner_cursor = reader.into_inner();
        let mut buf = [0u8; 1];
        assert_eq!(inner_cursor.read(&mut buf)?, 1);
        assert_eq!(buf[0], 12);
        Ok(())
    }

    #[test]
    fn test_get_ref() -> AifcResult<()> {
        let aifc = create_aifc(8, 1, b"NONE", &[ 11, 12, 13, 14 ]);
        let cursor = Cursor::new(&aifc);
        let mut reader = AifcReader::new(cursor)?;
        let info = reader.info();
        assert_eq!(info.sample_format, SampleFormat::I8);
        assert_eq!(reader.read_sample()?, Some(Sample::I8(11)));
        let ref_cursor = reader.get_ref();
        assert_eq!(ref_cursor.position(), 73);
        assert_eq!(reader.read_sample()?, Some(Sample::I8(12)));
        Ok(())
    }

    #[test]
    fn test_get_mut() -> AifcResult<()> {
        let aifc = create_aifc(8, 1, b"NONE", &[ 11, 12, 13, 14 ]);
        let cursor = Cursor::new(&aifc);
        let mut reader = AifcReader::new(cursor)?;
        let info = reader.info();
        assert_eq!(info.sample_format, SampleFormat::I8);
        assert_eq!(reader.read_sample()?, Some(Sample::I8(11)));
        let inner_cursor = reader.get_mut();
        let mut buf = [0u8; 1];
        assert_eq!(inner_cursor.read(&mut buf)?, 1);
        assert_eq!(buf[0], 12);
        Ok(())
    }

    #[test]
    fn test_read_info() -> AifcResult<()> {
        // stream position is at the first sample after read_info() has been called
        let aifc = create_aifc(8, 1, b"NONE", &[ 11, 12, 13, 14 ]);
        let cursor = Cursor::new(&aifc);
        let mut reader = AifcReader::new(cursor)?;
        let info = reader.info();
        assert_eq!(info.sample_format, SampleFormat::I8);
        let cr = reader.get_mut();
        let mut buf = [0u8; 1];
        assert_eq!(cr.read(&mut buf)?, 1);
        assert_eq!(buf[0], 11);

        let aifc = create_aifc(8, 1, b"NONE", &[ 11, 12, 13, 14 ]);
        let cursor = Cursor::new(&aifc);
        let mut reader = AifcReader::new_single_pass(cursor)?;
        let info = reader.info();
        assert_eq!(info.sample_format, SampleFormat::I8);
        let cr = reader.get_mut();
        let mut buf = [0u8; 1];
        assert_eq!(cr.read(&mut buf)?, 1);
        assert_eq!(buf[0], 11);

        Ok(())
    }

    #[test]
    fn test_read_info_minimal_comm_chunk() -> AifcResult<()> {
        // minimal AIFF, not enough data
        let aifc = vec![
            b'F', b'O', b'R', b'M', 0, 0, 0, 29, b'A', b'I', b'F', b'F',
            b'C', b'O', b'M', b'M', 0, 0, 0, 17,   0, 1,        // num_channels
            0, 0, 0, 8,                                         // num_sample_frames
            0, 8,                                               // sample_size
            0x40, 0x0E, 0xAC, 0x44, 0x00, 0x00, 0x00, 0x00, 0x00, // sample_rate missing byte
        ];
        let mut cursor = std::io::Cursor::new(&aifc);
        assert!(matches!(AifcReader::new(&mut cursor), Err(AifcError::InvalidCommChunk)));

        // minimal AIFF with enough data
        let aifc = vec![
            b'F', b'O', b'R', b'M', 0, 0, 0, 30, b'A', b'I', b'F', b'F',
            b'C', b'O', b'M', b'M', 0, 0, 0, 18,   0, 1,        // num_channels
            0, 0, 0, 8,                                         // num_sample_frames
            0, 8,                                               // sample_size
            0x40, 0x0E, 0xAC, 0x44, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // sample_rate
        ];
        let mut cursor = std::io::Cursor::new(&aifc);
        let reader = AifcReader::new(&mut cursor);
        assert!(reader.is_ok());

        // minimal AIFF-C, not enough data
        let aifc = vec![
            b'F', b'O', b'R', b'M', 0, 0, 0, 46, b'A', b'I', b'F', b'C',
            b'F', b'V', b'E', b'R', 0, 0, 0, 4, 0xA2, 0x80, 0x51, 0x40,
            b'C', b'O', b'M', b'M', 0, 0, 0, 22,   0, 1,        // num_channels
            0, 0, 0, 8,                                         // num_sample_frames
            0, 8,                                               // sample_size
            0x40, 0x0E, 0xAC, 0x44, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // sample_rate
            b'N', b'O', b'N', b'E',
            // compression name missing
        ];
        let mut cursor = std::io::Cursor::new(&aifc);
        assert!(matches!(AifcReader::new(&mut cursor), Err(AifcError::InvalidCommChunk)));

        // minimal AIFF-C, compression name padding byte missing
        let aifc = vec![
            b'F', b'O', b'R', b'M', 0, 0, 0, 47, b'A', b'I', b'F', b'C',
            b'F', b'V', b'E', b'R', 0, 0, 0, 4, 0xA2, 0x80, 0x51, 0x40,
            b'C', b'O', b'M', b'M', 0, 0, 0, 23,   0, 1,        // num_channels
            0, 0, 0, 8,                                         // num_sample_frames
            0, 8,                                               // sample_size
            0x40, 0x0E, 0xAC, 0x44, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // sample_rate
            b'N', b'O', b'N', b'E',
            0    // compression name
        ];
        let mut cursor = std::io::Cursor::new(&aifc);
        let reader = AifcReader::new(&mut cursor);
        assert!(reader.is_ok());

        // minimal AIFF-C, compression name with a padding byte
        let aifc = vec![
            b'F', b'O', b'R', b'M', 0, 0, 0, 48, b'A', b'I', b'F', b'C',
            b'F', b'V', b'E', b'R', 0, 0, 0, 4, 0xA2, 0x80, 0x51, 0x40,
            b'C', b'O', b'M', b'M', 0, 0, 0, 24,   0, 1,        // num_channels
            0, 0, 0, 8,                                         // num_sample_frames
            0, 8,                                               // sample_size
            0x40, 0x0E, 0xAC, 0x44, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // sample_rate
            b'N', b'O', b'N', b'E',
            0, 0    // compression name
        ];
        let mut cursor = std::io::Cursor::new(&aifc);
        let reader = AifcReader::new(&mut cursor);
        assert!(reader.is_ok());
        Ok(())
    }

    #[test]
    fn test_form_size() -> AifcResult<()> {
        let aifc = vec![
            b'F', b'O', b'R', b'M', 0, 0, 0, 72, b'A', b'I', b'F', b'C',
            b'F', b'V', b'E', b'R', 0, 0, 0, 4, 0xA2, 0x80, 0x51, 0x40,
            b'C', b'O', b'M', b'M', 0, 0, 0, 24,   0, 1,        // num_channels
            0, 0, 0, 8,                                         // num_sample_frames
            0, 8,                                               // sample_size
            0x40, 0x0E, 0xAC, 0x44, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // sample_rate
            b'N', b'O', b'N', b'E',
            0, 0,
            b'S', b'S', b'N', b'D', 0, 0, 0, 16,  0, 0, 0, 0,  0, 0, 0, 0,
            1, 2, 3, 4, 5, 6, 7, 8,
            b'M', b'A', b'R', b'K', 0, 0, 0, 10,  0, 1, 0, 1, 0, 0, 0, 2, 1, b'm'
        ];
        let mut cursor = std::io::Cursor::new(&aifc);
        let mut reader = AifcReader::new(&mut cursor)?;
        // the MARK chunk is outside the FORM size, so it isn't found
        assert!(reader.read_chunk_markers(&mut [])?.is_none());
        Ok(())
    }

    #[test]
    fn test_read_sample() -> AifcResult<()> {
        let aifc = vec![
            b'F', b'O', b'R', b'M', 0, 0, 0, 56, b'A', b'I', b'F', b'F',
            b'C', b'O', b'M', b'M', 0, 0, 0, 18,   0, 1,        // num_channels
            0, 0, 0, 8,                                         // num_sample_frames
            0, 32,                                              // sample_size
            0x40, 0x0E, 0xAC, 0x44, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // sample_rate
            b'S', b'S', b'N', b'D', 0, 0, 0, 18,  0, 0, 0, 0,  0, 0, 0, 0,
            0, 0, 0, 1, 0, 0, 0, 2, 98, 99,
        ];
        {
            let mut cursor = std::io::Cursor::new(&aifc);
            let mut reader = AifcReader::new(&mut cursor)?;
            let info = reader.info();
            assert_eq!(info.sample_len, Some(2));
            assert_eq!(reader.read_sample()?, Some(Sample::I32(1)));
            assert_eq!(reader.read_sample()?, Some(Sample::I32(2)));
            // last two bytes are a broken sample and won't be read
            assert_eq!(reader.read_sample()?, None);
        }
        Ok(())
    }

    #[test]
    fn test_samples_iterator() -> AifcResult<()> {
        let aifc = vec![
            b'F', b'O', b'R', b'M', 0, 0, 0, 72, b'A', b'I', b'F', b'C',
            b'F', b'V', b'E', b'R', 0, 0, 0, 4, 0xA2, 0x80, 0x51, 0x40,
            b'C', b'O', b'M', b'M', 0, 0, 0, 24,   0, 1,        // num_channels
            0, 0, 0, 8,                                         // num_sample_frames
            0, 8,                                               // sample_size
            0x40, 0x0E, 0xAC, 0x44, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // sample_rate
            b'N', b'O', b'N', b'E',
            0, 0,    // compression name
            b'S', b'S', b'N', b'D', 0, 0, 0, 16,  0, 0, 0, 0,  0, 0, 0, 0,
            1, 2, 3, 4, 5, 6, 7, 8
        ];
        let mut cursor = std::io::Cursor::new(&aifc);
        let mut reader = AifcReader::new(&mut cursor)?;
        // check that size_hint() and len() are decreasing
        {
            let mut siter = reader.samples()?;
            assert_eq!(siter.size_hint(), (8, Some(8)));
            assert_eq!(siter.next().expect("no sample")?, Sample::I8(1));
            assert_eq!(siter.size_hint(), (7, Some(7)));
        }
        assert_eq!(reader.read_sample()?.expect("sample missing"),
            Sample::I8(2));
        {
            let siter = reader.samples()?;
            assert_eq!(siter.size_hint(), (6, Some(6)));
        }
        // ensure the remaining samples are read
        let expected = vec![
            Sample::I8(3), Sample::I8(4),
            Sample::I8(5), Sample::I8(6), Sample::I8(7), Sample::I8(8),
        ];
        for (i, sample) in reader.samples()?.enumerate() {
            assert_eq!(sample?, expected[i]);
        }
        // size_hint(), len() and count() must be zero after all samples have been read
        {
            let siter = reader.samples()?;
            assert_eq!(siter.size_hint(), (0, Some(0)));
            assert_eq!(siter.count(), 0);
        }
        // reader can be accessed after Samples isn't used anymore
        assert!(reader.read_sample()?.is_none());

        // unsupported compression type
        let aifc = vec![
            b'F', b'O', b'R', b'M', 0, 0, 0, 72, b'A', b'I', b'F', b'C',
            b'F', b'V', b'E', b'R', 0, 0, 0, 4, 0xA2, 0x80, 0x51, 0x40,
            b'C', b'O', b'M', b'M', 0, 0, 0, 24,   0, 1,        // num_channels
            0, 0, 0, 8,                                         // num_sample_frames
            0, 8,                                               // sample_size
            0x40, 0x0E, 0xAC, 0x44, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // sample_rate
            b'X', b'Y', b'Z', b' ',
            0, 0,    // compression name
            b'S', b'S', b'N', b'D', 0, 0, 0, 16,  0, 0, 0, 0,  0, 0, 0, 0,
            1, 2, 3, 4, 5, 6, 7, 8
        ];
        let mut cursor = std::io::Cursor::new(&aifc);
        let mut reader = AifcReader::new(&mut cursor)?;
        assert!(matches!(reader.samples(), Err(AifcError::Unsupported)));

        Ok(())
    }

    #[test]
    fn test_read_compression_name() -> AifcResult<()> {
        // odd number of pstring chars
        let aifc = vec![
            b'F', b'O', b'R', b'M', 0, 0, 0, 82, b'A', b'I', b'F', b'C',
            b'F', b'V', b'E', b'R', 0, 0, 0, 4, 0xA2, 0x80, 0x51, 0x40,
            b'C', b'O', b'M', b'M', 0, 0, 0, 34,   0, 1,        // num_channels
            0, 0, 0, 8,                                         // num_sample_frames
            0, 8,                                               // sample_size
            0x40, 0x0E, 0xAC, 0x44, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // sample_rate
            b'N', b'O', b'N', b'E',
            11, b'n', b'o', b' ', b'c', b'o', b'm', b'p', b'r', b'e', b's', b's',
            b'S', b'S', b'N', b'D', 0, 0, 0, 16,  0, 0, 0, 0,  0, 0, 0, 0,
            1, 2, 3, 4, 5, 6, 7, 8
        ];
        let mut cursor = std::io::Cursor::new(&aifc);
        let mut reader = AifcReader::new(&mut cursor)?;
        assert_eq!(reader.read_sample()?.expect("no sample?"), Sample::I8(1));
        let clen = reader.read_compression_name(&mut [])?;
        let mut cname = vec![0u8; clen];
        reader.read_compression_name(&mut cname)?;
        assert_eq!(cname, b"no compress");
        assert_eq!(reader.read_sample()?.expect("no sample?"), Sample::I8(2));

        // even number of pstring chars
        let aifc = vec![
            b'F', b'O', b'R', b'M', 0, 0, 0, 86, b'A', b'I', b'F', b'C',
            b'F', b'V', b'E', b'R', 0, 0, 0, 4, 0xA2, 0x80, 0x51, 0x40,
            b'C', b'O', b'M', b'M', 0, 0, 0, 38,   0, 1,        // num_channels
            0, 0, 0, 8,                                         // num_sample_frames
            0, 8,                                               // sample_size
            0x40, 0x0E, 0xAC, 0x44, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // sample_rate
            b'N', b'O', b'N', b'E',
            14, b'n', b'o', b't', b' ', b'c', b'o', b'm', b'p', b'r', b'e', b's', b's', b'e', b'd', 0,
            b'S', b'S', b'N', b'D', 0, 0, 0, 16,  0, 0, 0, 0,  0, 0, 0, 0,
            1, 2, 3, 4, 5, 6, 7, 8
        ];
        let mut cursor = std::io::Cursor::new(&aifc);
        let mut reader = AifcReader::new(&mut cursor)?;
        let clen = reader.read_compression_name(&mut [])?;
        let mut cname = vec![0u8; clen];
        reader.read_compression_name(&mut cname)?;
        assert_eq!(cname, b"not compressed");

        // check compression name is clamped to COMM chunk size
        let aifc = vec![
            b'F', b'O', b'R', b'M', 0, 0, 0, 74, b'A', b'I', b'F', b'C',
            b'F', b'V', b'E', b'R', 0, 0, 0, 4, 0xA2, 0x80, 0x51, 0x40,
            b'C', b'O', b'M', b'M', 0, 0, 0, 26,   0, 1,        // num_channels
            0, 0, 0, 8,                                         // num_sample_frames
            0, 8,                                               // sample_size
            0x40, 0x0E, 0xAC, 0x44, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // sample_rate
            b'N', b'O', b'N', b'E',
            199, b'n', b'o', b't',
            b'S', b'S', b'N', b'D', 0, 0, 0, 16,  0, 0, 0, 0,  0, 0, 0, 0,
            1, 2, 3, 4, 5, 6, 7, 8
        ];
        let mut cursor = std::io::Cursor::new(&aifc);
        let mut reader = AifcReader::new(&mut cursor)?;
        let clen = reader.read_compression_name(&mut [])?;
        let mut cname = vec![0u8; clen];
        reader.read_compression_name(&mut cname)?;
        assert_eq!(cname, b"not");
        Ok(())
    }

    #[test]
    fn test_chunks_iterator() -> AifcResult<()> {
        let mut data = vec![ 11, 12, 13, 14, 15, 16 ];
        data.extend(create_aifc(8, 1, b"NONE", &[ 1, 2, 3, 4 ]));
        let mut cursor = std::io::Cursor::new(&data);
        read_dummy_data(&mut cursor, 6)?;
        let mut reader = AifcReader::new(&mut cursor)?;
        let mut iter = reader.chunks()?;
        let mut test_chunk_refs = vec![
            ChunkRef { pos: 12, size: 4, id: *b"FVER" },
            ChunkRef { pos: 24, size: 24, id: *b"COMM" },
            ChunkRef { pos: 56, size: 12, id: *b"SSND" },
        ];
        let mut test_chunk_buf = vec![
            vec![ 0xA2, 0x80, 0x51, 0x40, ],
            vec![ 0, 1,    // num_channels
            0, 0, 0, 4,        // num_sample_frames
            0, 8,                             // sample_size
            0x40, 0x0E, 0xAC, 0x44, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // sample_rate
            b'N', b'O', b'N', b'E',
            0, 0, ],
            vec![ 0, 0, 0, 0,  0, 0, 0, 0,  1, 2, 3, 4 ],
        ];
        while let Some(chuckref) = iter.next() {
            let chuckref = chuckref?;
            assert_eq!(chuckref, test_chunk_refs.remove(0));
            let mut buf = vec![0u8; usize::try_from(chuckref.size).expect("Chunk too large")];
            iter.read_data(&chuckref, &mut buf)?;
            // read data can be called twice
            iter.read_data(&chuckref, &mut buf)?;
            assert_eq!(buf, test_chunk_buf.remove(0));
        }
        assert!(iter.next().is_none());

        // chunks can be iterated while reading samples
        let mut data = vec![ 11, 12, 13, 14, 15, 16 ];
        data.extend(create_aifc(8, 1, b"NONE", &[ 1, 2, 3, 4 ]));
        let mut cursor = std::io::Cursor::new(&data);
        read_dummy_data(&mut cursor, 6)?;
        let mut reader = AifcReader::new(&mut cursor)?;
        assert_eq!(reader.read_sample()?.expect("no sample?"), Sample::I8(1));
        assert_eq!(reader.read_sample()?.expect("no sample?"), Sample::I8(2));
        let mut iter = reader.chunks()?;
        assert!(iter.next().is_some());
        assert_eq!(reader.read_sample()?.expect("no sample?"), Sample::I8(3));
        assert_eq!(reader.read_sample()?.expect("no sample?"), Sample::I8(4));

        Ok(())
    }

    #[test]
    fn test_read_chunk_markers() -> AifcResult<()> {
        let aifc = vec![
            b'F', b'O', b'R', b'M', 0, 0, 0, 90, b'A', b'I', b'F', b'C',
            b'F', b'V', b'E', b'R', 0, 0, 0, 4, 0xA2, 0x80, 0x51, 0x40,
            b'C', b'O', b'M', b'M', 0, 0, 0, 24,   0, 1,        // num_channels
            0, 0, 0, 8,                                         // num_sample_frames
            0, 8,                                               // sample_size
            0x40, 0x0E, 0xAC, 0x44, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // sample_rate
            b'N', b'O', b'N', b'E',
            0, 0,
            b'S', b'S', b'N', b'D', 0, 0, 0, 16,  0, 0, 0, 0,  0, 0, 0, 0,
            1, 2, 3, 4, 5, 6, 7, 8,
            b'M', b'A', b'R', b'K', 0, 0, 0, 10,  0, 1, 0, 1, 0, 0, 0, 2, 1, b'm'
        ];
        let mut cursor = std::io::Cursor::new(&aifc);
        let mut reader = AifcReader::new(&mut cursor)?;
        if let Some(size) = reader.read_chunk_markers(&mut [])? {
            let mut buf = vec![0; size];
            reader.read_chunk_markers(&mut buf)?;
            let mut marker_iter = crate::Markers::new(&buf)?;
            assert_eq!(marker_iter.size_hint(), (1, Some(1)));
            assert_eq!(marker_iter.len(), 1);
            assert!(matches!(marker_iter.next(),
                Some(Ok(crate::Marker { id: 1, position: 2, name: &[ b'm' ] }))));
            assert_eq!(marker_iter.size_hint(), (0, Some(0)));
            assert_eq!(marker_iter.len(), 0);
            assert!(marker_iter.next().is_none());
            assert_eq!(marker_iter.count(), 0);
            // count() can be called
            let marker_iter = crate::Markers::new(&buf)?;
            assert_eq!(marker_iter.count(), 1);
        }
        Ok(())
    }

    #[test]
    fn test_read_chunk_comments() -> AifcResult<()> {
        let aifc = vec![
            b'F', b'O', b'R', b'M', 0, 0, 0, 92, b'A', b'I', b'F', b'C',
            b'F', b'V', b'E', b'R', 0, 0, 0, 4, 0xA2, 0x80, 0x51, 0x40,
            b'C', b'O', b'M', b'M', 0, 0, 0, 24,   0, 1,        // num_channels
            0, 0, 0, 8,                                         // num_sample_frames
            0, 8,                                               // sample_size
            0x40, 0x0E, 0xAC, 0x44, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // sample_rate
            b'N', b'O', b'N', b'E',
            0, 0,
            b'S', b'S', b'N', b'D', 0, 0, 0, 16,  0, 0, 0, 0,  0, 0, 0, 0,
            1, 2, 3, 4, 5, 6, 7, 8,
            b'C', b'O', b'M', b'T', 0, 0, 0, 12,  0, 1, 0, 0, 0, 0,  0, 1,  0, 2, b'c', b'm'
        ];
        let mut cursor = std::io::Cursor::new(&aifc);
        let mut reader = AifcReader::new(&mut cursor)?;
        if let Some(size) = reader.read_chunk_comments(&mut [])? {
            let mut buf = vec![0; size];
            reader.read_chunk_comments(&mut buf)?;
            let mut comment_iter = crate::Comments::new(&buf)?;
            assert_eq!(comment_iter.size_hint(), (1, Some(1)));
            assert_eq!(comment_iter.len(), 1);
            assert!(matches!(comment_iter.next(),
                Some(Ok(crate::Comment { timestamp: 0, marker_id: 1, text: &[ b'c', b'm' ] }))));
            assert_eq!(comment_iter.size_hint(), (0, Some(0)));
            assert_eq!(comment_iter.len(), 0);
            assert!(comment_iter.next().is_none());
            assert_eq!(comment_iter.count(), 0);
            // count() can be called
            let comment_iter = crate::Comments::new(&buf)?;
            assert_eq!(comment_iter.count(), 1);
        }
        Ok(())
    }

    #[test]
    fn test_read_chunk_id3() -> AifcResult<()> {
        let aifc = vec![
            b'F', b'O', b'R', b'M', 0, 0, 0, 88, b'A', b'I', b'F', b'C',
            b'F', b'V', b'E', b'R', 0, 0, 0, 4, 0xA2, 0x80, 0x51, 0x40,
            b'C', b'O', b'M', b'M', 0, 0, 0, 24,   0, 1,        // num_channels
            0, 0, 0, 8,                                         // num_sample_frames
            0, 8,                                               // sample_size
            0x40, 0x0E, 0xAC, 0x44, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // sample_rate
            b'N', b'O', b'N', b'E',
            0, 0,
            b'S', b'S', b'N', b'D', 0, 0, 0, 16,  0, 0, 0, 0,  0, 0, 0, 0,
            1, 2, 3, 4, 5, 6, 7, 8,
            b'I', b'D', b'3', b' ', 0, 0, 0, 8,  6, 5, 4, 3,  1, 2, 3, 4,
        ];
        let mut cursor = std::io::Cursor::new(&aifc);
        let mut reader = AifcReader::new(&mut cursor)?;
        if let Some(size) = reader.read_chunk_id3(&mut [])? {
            let mut buf = vec![0; size];
            reader.read_chunk_id3(&mut buf)?;
            assert_eq!(buf, vec![ 6, 5, 4, 3,  1, 2, 3, 4, ]);
        }
        Ok(())
    }

    #[test]
    fn test_seek() -> AifcResult<()> {
        // seeking i16 samples
        // three dummy bytes at the start of the data to test that they don't affect seeking
        let mut aifc = vec![ 90, 91, 92 ];
        aifc.append(&mut create_aifc(16, 1, b"NONE",
            &[ 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8 ]));
        // audio data contains extra bytes, which should not affect seeking
        aifc.extend_from_slice(&[ 0x80, 0x81, 0x82, 0x83, 0x84, 0x85 ]);

        let mut cursor = Cursor::new(&aifc);
        read_dummy_data(&mut cursor, 3)?;
        let mut reader = AifcReader::new(&mut cursor)?;
        assert_eq!(reader.read_sample()?, Some(Sample::I16(1)));
        assert_eq!(reader.read_sample()?, Some(Sample::I16(2)));
        reader.seek(0)?;
        assert_eq!(reader.read_sample()?, Some(Sample::I16(1)));
        reader.seek(4)?;
        assert_eq!(reader.read_sample()?, Some(Sample::I16(5)));
        reader.seek(7)?;
        assert_eq!(reader.read_sample()?, Some(Sample::I16(8)));
        reader.seek(8)?;
        assert!(reader.read_sample()?.is_none());
        assert!(matches!(reader.seek(9), Err(AifcError::SeekError)));
        assert!(reader.read_sample()?.is_none());
        assert!(matches!(reader.seek(u64::MAX), Err(AifcError::SeekError)));
        assert!(reader.read_sample()?.is_none());
        reader.seek(4)?;
        assert_eq!(reader.read_sample()?, Some(Sample::I16(5)));

        // check seek fails for a custom format

        let aifc = create_aifc(8, 1, b"wxyz", &[ 1, 2, 3, 4, 5, 6, 7, 8 ]);
        let mut cursor = Cursor::new(&aifc);
        let mut reader = AifcReader::new(&mut cursor)?;
        assert!(matches!(reader.seek(0), Err(AifcError::Unsupported)));
        assert!(matches!(reader.seek(1), Err(AifcError::Unsupported)));

        Ok(())
    }

    fn create_ima4_samples(start_value: i16) -> [i16; 64] {
        let mut buf = [0i16; 64];
        for (i, b) in buf.iter_mut().enumerate() {
            *b = i as i16 + start_value;
        }
        buf
    }

    #[test]
    fn test_seek_ima4_ch1() -> AifcResult<()> {
        // seeking 1 channel ima4 compressed data
        let mut ima4data = vec![];
        let mut adpcm_state = AdpcmImaState::new();
        let mut compressed_ima4 = [0u8; 34];
        audio_codec_algorithms::encode_adpcm_ima_ima4(&[
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
            33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
            368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383,
        ], &mut adpcm_state, &mut compressed_ima4);
        ima4data.extend_from_slice(&compressed_ima4);
        audio_codec_algorithms::encode_adpcm_ima_ima4(&create_ima4_samples(384),
            &mut adpcm_state, &mut compressed_ima4);
        ima4data.extend_from_slice(&compressed_ima4);
        // three dummy bytes at the start of the data to test that they don't affect seeking
        let mut aifc = vec![ 90, 91, 92 ];
        // note: this creates an aifc file with invalid num_sample_frames, but it doesn't matter..
        aifc.append(&mut create_aifc(8, 1, b"ima4", &ima4data));
        // audio data contains extra bytes, which should not affect seeking
        aifc.extend_from_slice(&[ 0x80, 0x81, 0x82, 0x83, 0x84, 0x85 ]);

        let mut cursor = Cursor::new(&aifc);
        read_dummy_data(&mut cursor, 3)?;
        let mut reader = AifcReader::new(&mut cursor)?;
        let info = reader.info();
        assert_eq!(info.sample_len, Some(128));
        // seek the first packet (the first 64 samples)
        assert_eq!(reader.read_sample()?, Some(Sample::I16(1)));
        assert_eq!(reader.read_sample()?, Some(Sample::I16(2)));
        reader.seek(0)?;
        assert_eq!(reader.read_sample()?, Some(Sample::I16(1)));
        reader.seek(14)?;
        assert_eq!(reader.read_sample()?, Some(Sample::I16(15)));
        // seek to the second packet, the adpcm predictor state is reset (*)
        reader.seek(64)?;
        assert_eq!(reader.read_sample()?, Some(Sample::I16(378)));
        reader.seek(127)?;
        assert_eq!(reader.read_sample()?, Some(Sample::I16(443)));
        // seek back to the first packet
        reader.seek(63)?;
        assert_eq!(reader.read_sample()?, Some(Sample::I16(388)));
        // reading advances to the second packet here (sample position 64)
        // values differ from above (*) because the adpcm predictor state isn't reset here
        assert_eq!(reader.read_sample()?, Some(Sample::I16(382)));
        assert_eq!(reader.read_sample()?, Some(Sample::I16(387)));
        // seek past sample_len
        reader.seek(128)?;
        assert!(reader.read_sample()?.is_none());
        assert!(matches!(reader.seek(129), Err(AifcError::SeekError)));
        assert!(reader.read_sample()?.is_none());
        assert!(matches!(reader.seek(u64::MAX), Err(AifcError::SeekError)));
        assert!(reader.read_sample()?.is_none());
        // back to the first packet
        reader.seek(4)?;
        assert_eq!(reader.read_sample()?, Some(Sample::I16(5)));

        Ok(())
    }

    #[test]
    fn test_seek_ima4_ch2() -> AifcResult<()> {
        // seeking 2 channel ima4 compressed data
        let mut ima4data = vec![];
        let mut adpcm_state0 = AdpcmImaState::new();
        let mut adpcm_state1 = AdpcmImaState::new();
        let mut compressed_ima4 = [0u8; 34];
        audio_codec_algorithms::encode_adpcm_ima_ima4(&create_ima4_samples(1001),
            &mut adpcm_state0, &mut compressed_ima4);
        ima4data.extend_from_slice(&compressed_ima4);
        audio_codec_algorithms::encode_adpcm_ima_ima4(&create_ima4_samples(-1001),
            &mut adpcm_state1, &mut compressed_ima4);
        ima4data.extend_from_slice(&compressed_ima4);
        audio_codec_algorithms::encode_adpcm_ima_ima4(&create_ima4_samples(1501),
            &mut adpcm_state0, &mut compressed_ima4);
        ima4data.extend_from_slice(&compressed_ima4);
        audio_codec_algorithms::encode_adpcm_ima_ima4(&create_ima4_samples(-1501),
            &mut adpcm_state1, &mut compressed_ima4);
        ima4data.extend_from_slice(&compressed_ima4);
        // three dummy bytes at the start of the data to test that they don't affect seeking
        let mut aifc = vec![ 90, 91, 92 ];
        // note: this creates an aifc file with invalid num_sample_frames, but it doesn't matter..
        aifc.append(&mut create_aifc(8, 2, b"ima4", &ima4data));
        // audio data contains extra bytes, which should not affect seeking
        aifc.extend_from_slice(&[ 0x80, 0x81, 0x82, 0x83, 0x84, 0x85 ]);

        let mut cursor = Cursor::new(&aifc);
        read_dummy_data(&mut cursor, 3)?;
        let mut reader = AifcReader::new(&mut cursor)?;
        let info = reader.info();
        assert_eq!(info.sample_len, Some(256));
        // seek the first 2 packets (the first 128 samples for left and right)
        assert_eq!(reader.read_sample()?, Some(Sample::I16(11))); // left
        assert_eq!(reader.read_sample()?, Some(Sample::I16(-11))); // right
        assert_eq!(reader.read_sample()?, Some(Sample::I16(41)));
        assert_eq!(reader.read_sample()?, Some(Sample::I16(-41)));
        reader.seek(0)?;
        assert_eq!(reader.read_sample()?, Some(Sample::I16(11)));
        reader.seek(14)?;
        assert_eq!(reader.read_sample()?, Some(Sample::I16(1001)));
        // seek to new packets, the adpcm predictor state is reset (*)
        reader.seek(128)?;
        assert_eq!(reader.read_sample()?, Some(Sample::I16(1035))); // left
        assert_eq!(reader.read_sample()?, Some(Sample::I16(-1035))); // right
        reader.seek(255)?;
        assert_eq!(reader.read_sample()?, Some(Sample::I16(-1524)));
        // seek back to first packets
        reader.seek(127)?;
        assert_eq!(reader.read_sample()?, Some(Sample::I16(-938)));
        // reading advances to packets 3-4 here (sample positions 128-129)
        // values differ from above (*) because the adpcm predictor state isn't reset here
        assert_eq!(reader.read_sample()?, Some(Sample::I16(1075))); // left
        assert_eq!(reader.read_sample()?, Some(Sample::I16(-949))); // right
        // seek past sample_len
        reader.seek(256)?;
        assert!(reader.read_sample()?.is_none());
        assert!(matches!(reader.seek(257), Err(AifcError::SeekError)));
        assert!(reader.read_sample()?.is_none());
        assert!(matches!(reader.seek(u64::MAX), Err(AifcError::SeekError)));
        assert!(reader.read_sample()?.is_none());
        // back to the first packets
        reader.seek(4)?;
        assert_eq!(reader.read_sample()?, Some(Sample::I16(104))); // left
        assert_eq!(reader.read_sample()?, Some(Sample::I16(-104))); // right

        Ok(())
    }

    #[test]
    fn test_reading_after_dummy_data() -> AifcResult<()> {
        // the stream contains 6 dummy bytes in the beginning
        let aifc = create_aifc(8, 1, b"NONE", &[ 1, 2, 3, 4 ]);
        let mut data = vec![ 11, 12, 13, 14, 15, 16 ];
        data.extend(aifc);
        let mut cursor = std::io::Cursor::new(&data);
        // read dummy data before aifc data
        read_dummy_data(&mut cursor, 6)?;
        let mut reader = AifcReader::new(&mut cursor)?;
        let clen = reader.read_compression_name(&mut [])?;
        let mut cname = vec![0u8; clen];
        reader.read_compression_name(&mut cname)?;
        assert_eq!(cname, b"");
        assert_eq!(reader.read_sample()?.expect("no sample?"), Sample::I8(1));
        assert_eq!(reader.read_sample()?.expect("no sample?"), Sample::I8(2));
        assert_eq!(reader.chunks()?.count(), 3);
        assert_eq!(reader.read_sample()?.expect("no sample?"), Sample::I8(3));
        assert_eq!(reader.read_sample()?.expect("no sample?"), Sample::I8(4));
        Ok(())
    }

    #[test]
    fn test_comment_timestamp() -> AifcResult<()> {
        let mut c = crate::Comment {
            timestamp: 0,
            marker_id: 0,
            text: &[]
        };
        // zero values
        assert_eq!(c.unix_timestamp(), -2082844800);
        assert!(c.set_unix_timestamp(0).is_ok());
        assert_eq!(c.timestamp, 2082844800);

        // ok values
        assert!(c.set_unix_timestamp(i64::from(u32::MAX) - 2082844800).is_ok());
        assert!(c.set_unix_timestamp(-2082844800).is_ok());

        // out-of-bounds values return errors
        assert!(matches!(c.set_unix_timestamp(i64::from(u32::MAX) - 2082844799),
            Err(AifcError::TimestampOutOfBounds)));
        assert!(matches!(c.set_unix_timestamp(-2082844801), Err(AifcError::TimestampOutOfBounds)));
        assert!(matches!(c.set_unix_timestamp(i64::MIN), Err(AifcError::TimestampOutOfBounds)));
        assert!(matches!(c.set_unix_timestamp(i64::MAX), Err(AifcError::TimestampOutOfBounds)));
        assert!(matches!(c.set_unix_timestamp(i64::from(u32::MAX)),
            Err(AifcError::TimestampOutOfBounds)));

        // 2023-11-10T07:26:29
        assert!(c.set_unix_timestamp(1699601189).is_ok());
        assert_eq!(c.timestamp, 3782445989);
        assert_eq!(c.unix_timestamp(), 1699601189);

        // 2040-02-06T06:28:15 (u32::MAX == 4294967295)
        c.timestamp = u32::MAX;
        assert_eq!(c.unix_timestamp(), 2212122495);
        Ok(())
    }
}
