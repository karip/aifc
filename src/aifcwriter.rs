
use audio_codec_algorithms::AdpcmImaState;
use crate::{cast, AifcError, AifcResult, ChunkId, FileFormat, Marker, Comment,
    SampleFormat, Seek, SeekFrom, Write, CountingWrite};

#[derive(Debug, Clone, Copy, PartialEq)]
enum WriteState {
    SamplesNotWritten,
    WritingSamples,
    WritingRawSamples,
    SamplesDone,
    Finalized
}

/// Checked add for writing.
#[inline(always)]
fn wchecked_add(lhs: u64, rhs: u64) -> AifcResult<u64> {
    lhs.checked_add(rhs).ok_or(AifcError::SizeTooLarge)
}

/// Audio info for `AifcWriter`.
#[derive(Debug, Clone, PartialEq)]
pub struct AifcWriteInfo {

    /// File format: AIFF or AIFF-C.
    pub file_format: FileFormat,

    /// Number of channels. This must be greater than zero.
    pub channels: i16,

    /// Sample rate, samples per second. This must be greater than or equal to 0.001.
    /// Infinite or NaN values are not allowed.
    pub sample_rate: f64,

    /// Sample format to be written. AIFF-C allows any sample format, but
    /// AIFF allows only signed big-endian integer sample formats.
    pub sample_format: SampleFormat,
}

impl Default for AifcWriteInfo {
    /// Default values: AIFF-C with 2 channels, sample rate 44100 and sample format I16.
    fn default() -> Self {
        AifcWriteInfo {
            file_format: FileFormat::Aifc,
            channels: 2,
            sample_rate: 44100.0,
            sample_format: SampleFormat::I16,
        }
    }
}

/// AIFF / AIFF-C writer.
///
/// `AifcWriter` writes audio samples and chunk data to the given writer implementing `Write+Seek`.
/// When writing audio samples, channel data is interleaved.
///
/// Chunks (markers, comments, id3 data, ..) can be written before any samples have been written
/// or after all samples have been written. It is not possible to write samples, chunk data and
/// then more samples.
///
/// **At the end, the [`finalize()`](AifcWriter::finalize()) method must be called to
/// finish writing.**
/// It isn't enough to just drop `AifcWriter` or call [`flush()`](AifcWriter::flush()),
/// because the stream wouldn't be padded and updated correctly.
///
/// The writer doesn't perform any buffering, so it's recommended to use a buffered writer with it.
///
/// # Errors
///
/// If any of the methods returns an error, then the writer shouldn't be used anymore.
///
/// # Examples
///
/// Writing AIFF-C with 2 channels, sample rate 48000 and signed 16-bit integer samples:
///
/// ```no_run
/// # fn example() -> aifc::AifcResult<()> {
/// let mut buf = std::io::Cursor::new(vec![0; 10000]);
/// let info = aifc::AifcWriteInfo {
///     file_format: aifc::FileFormat::Aifc,
///     channels: 2,
///     sample_rate: 48000.0,
///     sample_format: aifc::SampleFormat::I16,
/// };
/// let mut w = aifc::AifcWriter::new(&mut buf, &info)?;
/// w.write_samples_i16(&[ 0, 0, 10, 10, 20, 20, 30, 30, 40, 40 ])?;
/// w.finalize()?;
/// # Ok(())
/// # }
/// ```
pub struct AifcWriter<W> where W: Write + Seek {
    /// The underlying writer.
    handle: CountingWrite<W>,

    /// Info written to the stream.
    info: AifcWriteInfo,

    /// Write state to track which methods can be called.
    state: WriteState,

    /// Sample count written to the stream. For raw data writing, this is byte count written.
    samples_written: u64,

    /// FORM start position relative to the start of the stream (absolute position).
    initial_stream_pos: u64,

    /// SSND position relative to the FORM start position.
    ssnd_chunk_size_pos: Option<u64>,

    /// Position to update the num_sampleframes value, relative to the FORM start position.
    comm_numsampleframes_pos: u64,

    /// Stores frame count for write_samples_raw().
    raw_data_frame_count: Option<u32>,

    /// Sample buffer for ima4 compression, 2 channels.
    sample_buffer: [[i16; 64]; 2],
    /// Sample position in the compression sample buffer.
    sample_buffer_pos: usize,
    /// Channel position in the compression sample buffer.
    sample_buffer_ch: usize,
    /// IMA ADPCM states for 2 channels.
    ima4_state: [AdpcmImaState; 2],
}

impl<W: Write + Seek> AifcWriter<W> {
    /// Create a new `AifcWriter`. The given stream must implement the `Write` and `Seek`
    /// traits.
    ///
    /// The header data is written immediately to the inner writer.
    pub fn new(mut stream: W, info: &AifcWriteInfo) -> AifcResult<AifcWriter<W>> {
        let initial_stream_pos = stream.stream_position()?;
        let mut write = CountingWrite::new(stream);
        let sample_size = info.sample_format.bits_per_sample();
        let comm_numsampleframes_pos = write_header(&mut write, info, sample_size)?;
        Ok(AifcWriter {
            handle: write,

            info: info.clone(),
            state: WriteState::SamplesNotWritten,
            samples_written: 0,

            initial_stream_pos,
            ssnd_chunk_size_pos: None,
            comm_numsampleframes_pos,
            raw_data_frame_count: None,

            sample_buffer: [[0i16; 64]; 2],
            sample_buffer_pos: 0,
            sample_buffer_ch: 0,
            ima4_state: [ AdpcmImaState::new(), AdpcmImaState::new() ],
        })
    }

    fn update_header(&mut self) -> AifcResult<()> {
        let spos = self.handle.handle.stream_position()?;
        let stream_size = self.handle.bytes_written;

        // update FORM size
        let form_pos = wchecked_add(self.initial_stream_pos, 4)?;
        self.handle.handle.seek(SeekFrom::Start(form_pos))?;
        // ensure that stream_size fits in 32-bit integer
        // (it would be possible to have u32::MAX byte FORM data and 8 bytes for
        // FORM id and size fields, but macOS QuickTime can't play those files)
        let mut form_size = match u32::try_from(stream_size) {
            Ok(v) => v,
            Err(_) => { return Err(AifcError::SizeTooLarge); }
        };
        form_size -= 8; // decrement FORM id and size fields (8 bytes)
        self.handle.write_not_counted(&form_size.to_be_bytes())?;

        // update COMM numSampleFrames
        let sf_pos = wchecked_add(self.initial_stream_pos, self.comm_numsampleframes_pos)?;
        self.handle.handle.seek(SeekFrom::Start(sf_pos))?;
        // num_channels is never zero, because it would be InvalidParameter
        let num_channels = cast::clamp_i16_to_u64(self.info.channels);
        let num_sample_frames = calculate_num_sample_frames(self.info.sample_format,
            self.samples_written, num_channels)?;
        let num_sample_frames = self.raw_data_frame_count.unwrap_or(num_sample_frames);
        self.handle.write_not_counted(&num_sample_frames.to_be_bytes())?;

        // update SSND chunk size
        if let Some(ssnd_pos) = self.ssnd_chunk_size_pos {
            let ssnd_pos = wchecked_add(self.initial_stream_pos, ssnd_pos)?;
            self.handle.handle.seek(SeekFrom::Start(ssnd_pos))?;
            let ssnd_size = calculate_ssnd_size(self.info.sample_format, self.samples_written,
                    self.raw_data_frame_count.is_some())?;
            self.handle.write_not_counted(&ssnd_size.to_be_bytes())?;
        }
        self.handle.handle.seek(SeekFrom::Start(spos))?;
        Ok(())
    }

    /// Flushes the stream. This will also update headers to match the currently written data.
    ///
    /// Note: it isn't enough to call `flush()` to finish writing, because the stream wouldn't
    /// be padded correctly. Use `finalize()` to finish writing.
    pub fn flush(&mut self) -> AifcResult<()> {
        self.update_header()?;
        self.handle.flush()?;
        Ok(())
    }

    /// Finalizes the writer.
    /// Call this method after all sample and chunk data has been written to write padding bytes and
    /// to update headers.
    ///
    /// The write methods must not be called after this method has been called.
    ///
    /// Returns an error if writing fails.
    pub fn finalize(&mut self) -> AifcResult<()> {
        if self.state == WriteState::Finalized {
            return Ok(());
        }
        if self.state == WriteState::WritingSamples || self.state == WriteState::WritingRawSamples {
            self.finish_ssnd()?;
        }
        self.update_header()?;
        self.handle.flush()?;
        self.state = WriteState::Finalized;
        Ok(())
    }

    fn check_state(&mut self) -> AifcResult<()> {
        if self.state == WriteState::Finalized {
            return Err(AifcError::InvalidWriteState);
        }
        if self.state == WriteState::WritingSamples || self.state == WriteState::WritingRawSamples {
            self.finish_ssnd()?;
            self.state = WriteState::SamplesDone;
        }
        Ok(())
    }

    /// Writes the given chunk data to the stream.
    ///
    /// Note that some chunks should be written only once to the stream, but
    /// this method doesn't perform any validation for that.
    pub fn write_chunk(&mut self, id: &ChunkId, data: &[u8]) -> AifcResult<()> {
        self.check_state()?;
        let chunksize = u32::try_from(data.len()).map_err(|_| AifcError::SizeTooLarge)?;
        self.handle.write_all(id)?;
        self.handle.write_all(&chunksize.to_be_bytes())?;
        self.handle.write_all(data)?;
        if !crate::is_even_u32(chunksize) { // write pad byte
            self.handle.write_all(&[ 0 ])?;
        }
        Ok(())
    }

    /// Writes the given markers to the stream.
    ///
    /// Note that markers should be written only once to the stream, but
    /// this method doesn't check that.
    pub fn write_chunk_markers(&mut self, markers: &[Marker]) -> AifcResult<()> {
        self.check_state()?;
        let chunksize = Marker::chunk_data_size(markers)?;
        self.handle.write_all(&crate::CHUNKID_MARK)?;
        self.handle.write_all(&chunksize.to_be_bytes())?;
        Marker::write_chunk_data(&mut self.handle, markers)
    }

    /// Writes the given comments to the stream.
    ///
    /// Note that comments should be written only once to the stream, but
    /// this method doesn't check that.
    pub fn write_chunk_comments(&mut self, comments: &[Comment]) -> AifcResult<()> {
        self.check_state()?;
        let chunksize = Comment::chunk_data_size(comments)?;
        self.handle.write_all(&crate::CHUNKID_COMT)?;
        self.handle.write_all(&chunksize.to_be_bytes())?;
        Comment::write_chunk_data(&mut self.handle, comments)
    }

    /// Writes the given ID3 data to the stream.
    ///
    /// Note that ID3 data should be written only once to the stream, but
    /// this method doesn't check that.
    pub fn write_chunk_id3(&mut self, id3_data: &[u8]) -> AifcResult<()> {
        self.write_chunk(&crate::CHUNKID_ID3, id3_data)
    }

    fn increment_samples_written(&mut self, data_len: usize) -> AifcResult<()> {
        self.samples_written = wchecked_add(self.samples_written,
            cast::usize_to_u64(data_len, AifcError::SizeTooLarge)?)?;
        // check that total number of bytes written fits in the 32-bit FORM size field
        // (it would be possible to write u32::MAX+8 bytes, but macOS QuickTime
        // can't play those files)
        if self.handle.bytes_written > u64::from(u32::MAX) {
            return Err(AifcError::SizeTooLarge);
        }
        Ok(())
    }

    /// Writes u8 samples. Call this method if the `AifcWriteInfo::sample_format` is set
    /// to `SampleFormat::U8`.
    pub fn write_samples_u8(&mut self, data: &[u8]) -> AifcResult<()> {
        let current_pos = self.handle.bytes_written;
        if self.info.sample_format != SampleFormat::U8 {
            return Err(AifcError::InvalidSampleFormat);
        }
        if self.state != WriteState::SamplesNotWritten && self.state != WriteState::WritingSamples {
            return Err(AifcError::InvalidWriteState);
        }
        self.state = WriteState::WritingSamples;
        self.write_ssnd_header(data.len(), current_pos)?;
        self.handle.write_all(data)?;
        self.increment_samples_written(data.len())
    }

    /// Writes i8 samples. Call this method if the `AifcWriteInfo::sample_format` is set
    /// to `SampleFormat::I8`.
    pub fn write_samples_i8(&mut self, data: &[i8]) -> AifcResult<()> {
        let current_pos = self.handle.bytes_written;
        if self.info.sample_format != SampleFormat::I8 {
            return Err(AifcError::InvalidSampleFormat);
        }
        if self.state != WriteState::SamplesNotWritten && self.state != WriteState::WritingSamples {
            return Err(AifcError::InvalidWriteState);
        }
        self.state = WriteState::WritingSamples;
        self.write_ssnd_header(data.len(), current_pos)?;
        for d in data {
            self.handle.write_all(&[ cast::i8_to_u8(*d) ])?;
        }
        self.increment_samples_written(data.len())
    }

    /// Writes i16 samples. Call this method if the `AifcWriteInfo::sample_format` is set to
    /// `SampleFormat::I16`, `SampleFormat::CompressedUlaw`, `SampleFormat::CompressedAlaw` or
    /// `SampleFormat::CompressedIma4`.
    pub fn write_samples_i16(&mut self, data: &[i16]) -> AifcResult<()> {
        let current_pos = self.handle.bytes_written;
        if self.info.sample_format != SampleFormat::I16 &&
            self.info.sample_format != SampleFormat::I16LE &&
            self.info.sample_format != SampleFormat::CompressedUlaw &&
            self.info.sample_format != SampleFormat::CompressedAlaw &&
            self.info.sample_format != SampleFormat::CompressedIma4 {
                return Err(AifcError::InvalidSampleFormat);
        }
        if self.state != WriteState::SamplesNotWritten && self.state != WriteState::WritingSamples {
            return Err(AifcError::InvalidWriteState);
        }
        self.state = WriteState::WritingSamples;
        self.write_ssnd_header(data.len(), current_pos)?;
        if self.info.sample_format == SampleFormat::I16 {
            for d in data {
                self.handle.write_all(&d.to_be_bytes())?;
            }

        } else if self.info.sample_format == SampleFormat::I16LE {
            for d in data {
                self.handle.write_all(&d.to_le_bytes())?;
            }

        } else if self.info.sample_format == SampleFormat::CompressedUlaw {
            for d in data {
                let encoded = audio_codec_algorithms::encode_ulaw(*d);
                self.handle.write_all(&[ encoded ])?;
            }

        } else if self.info.sample_format == SampleFormat::CompressedAlaw {
            for d in data {
                let encoded = audio_codec_algorithms::encode_alaw(*d);
                self.handle.write_all(&[ encoded ])?;
            }

        } else if self.info.sample_format == SampleFormat::CompressedIma4 {
            self.write_samples_ima4(data)?;
        }
        self.increment_samples_written(data.len())
    }

    /// Writes out ima4 compressed samples.
    fn write_samples_ima4(&mut self, data: &[i16]) -> AifcResult<()> {
        // fill buffer and when it is full, compress it and write it out
        for d in data {
            self.sample_buffer[self.sample_buffer_ch][self.sample_buffer_pos] = *d;
            self.sample_buffer_ch += 1;
            let num_channels = cast::clamp_i16_to_usize(self.info.channels);
            if self.sample_buffer_ch >= num_channels {
                self.sample_buffer_ch = 0;
                self.sample_buffer_pos += 1;
                if self.sample_buffer_pos == 64 {
                    self.sample_buffer_pos = 0;
                    for ch in 0..num_channels {
                        let mut compressed_buf = [0u8; 34];
                        audio_codec_algorithms::encode_adpcm_ima_ima4(&self.sample_buffer[ch],
                            &mut self.ima4_state[ch], &mut compressed_buf);
                        self.handle.write_all(&compressed_buf)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Writes i24 samples.
    /// The highest 8 bits of the 32-bit data are ignored when converting it to 24 bits.
    /// Call this method if the `AifcWriteInfo::sample_format` is set to `SampleFormat::I24`.
    pub fn write_samples_i24(&mut self, data: &[i32]) -> AifcResult<()> {
        let current_pos = self.handle.bytes_written;
        if self.info.sample_format != SampleFormat::I24 {
            return Err(AifcError::InvalidSampleFormat);
        }
        if self.state != WriteState::SamplesNotWritten && self.state != WriteState::WritingSamples {
            return Err(AifcError::InvalidWriteState);
        }
        self.state = WriteState::WritingSamples;
        self.write_ssnd_header(data.len(), current_pos)?;
        for d in data {
            let i32bytes = &d.to_be_bytes();
            let i24bytes = [ i32bytes[1], i32bytes[2], i32bytes[3] ];
            self.handle.write_all(&i24bytes)?;
        }
        self.increment_samples_written(data.len())
    }

    /// Writes i32 samples.
    /// Call this method if the `AifcWriteInfo::sample_format` is set to `SampleFormat::I32`.
    pub fn write_samples_i32(&mut self, data: &[i32]) -> AifcResult<()> {
        let current_pos = self.handle.bytes_written;
        if self.info.sample_format != SampleFormat::I32 &&
            self.info.sample_format != SampleFormat::I32LE {
            return Err(AifcError::InvalidSampleFormat);
        }
        if self.state != WriteState::SamplesNotWritten && self.state != WriteState::WritingSamples {
            return Err(AifcError::InvalidWriteState);
        }
        self.state = WriteState::WritingSamples;
        self.write_ssnd_header(data.len(), current_pos)?;
        match self.info.sample_format {
            SampleFormat::I32 => {
                for d in data {
                    self.handle.write_all(&d.to_be_bytes())?;
                }
            },
            _ => { // SampleFormat::I32LE
                for d in data {
                    self.handle.write_all(&d.to_le_bytes())?;
                }
            }
        }
        self.increment_samples_written(data.len())
    }

    /// Writes f32 samples.
    /// Call this method if the `AifcWriteInfo::sample_format` is set to `SampleFormat::F32`.
    pub fn write_samples_f32(&mut self, data: &[f32]) -> AifcResult<()> {
        let current_pos = self.handle.bytes_written;
        if self.info.sample_format != SampleFormat::F32 {
            return Err(AifcError::InvalidSampleFormat);
        }
        if self.state != WriteState::SamplesNotWritten && self.state != WriteState::WritingSamples {
            return Err(AifcError::InvalidWriteState);
        }
        self.state = WriteState::WritingSamples;
        self.write_ssnd_header(data.len(), current_pos)?;
        for d in data {
            self.handle.write_all(&d.to_be_bytes())?;
        }
        self.increment_samples_written(data.len())
    }

    /// Writes f64 samples.
    /// Call this method if the `AifcWriteInfo::sample_format` is set to `SampleFormat::F64`.
    pub fn write_samples_f64(&mut self, data: &[f64]) -> AifcResult<()> {
        let current_pos = self.handle.bytes_written;
        if self.info.sample_format != SampleFormat::F64 {
            return Err(AifcError::InvalidSampleFormat);
        }
        if self.state != WriteState::SamplesNotWritten && self.state != WriteState::WritingSamples {
            return Err(AifcError::InvalidWriteState);
        }
        self.state = WriteState::WritingSamples;
        self.write_ssnd_header(data.len(), current_pos)?;
        for d in data {
            self.handle.write_all(&d.to_be_bytes())?;
        }
        self.increment_samples_written(data.len())
    }

    /// Writes raw sample data. Raw sample data can be written for any sample format.
    /// If this method is called once, then other write samples methods can't be called anymore.
    /// This method is the only way to write sample data for custom sample formats.
    ///
    /// The `frame_count` is passed in so that the writer can update the header (num_frames) to
    /// a correct value when `flush()` or `finalize()` is called.
    /// `frame_count` must be the total number of audio frames (not audio samples) written so far.
    pub fn write_samples_raw(&mut self, data: &[u8], frame_count: u32) -> AifcResult<()> {
        if self.state != WriteState::SamplesNotWritten &&
            self.state != WriteState::WritingRawSamples {
            return Err(AifcError::InvalidWriteState);
        }
        self.state = WriteState::WritingRawSamples;
        let current_pos = self.handle.bytes_written;
        self.raw_data_frame_count = Some(frame_count);
        self.write_ssnd_header(data.len(), current_pos)?;
        self.handle.write_all(data)?;
        self.increment_samples_written(data.len())
    }

    /// Consumes this `AifcWriter` and returns the underlying writer.
    pub fn into_inner(self) -> W {
        self.handle.handle
    }

    /// Gets a reference to the underlying writer.
    pub const fn get_ref(&self) -> &W {
        &self.handle.handle
    }

    /// Gets a mutable reference to the underlying writer.
    ///
    /// Care should be taken to avoid modifying the internal I/O state of the
    /// underlying writer as it may corrupt this writer's state.
    pub fn get_mut(&mut self) -> &mut W {
        &mut self.handle.handle
    }

    fn write_ssnd_header(&mut self, datalen: usize, current_pos: u64) -> AifcResult<()> {
        if self.samples_written == 0 && datalen > 0 {
            self.ssnd_chunk_size_pos = Some(wchecked_add(current_pos, 4)?);
            self.handle.write_all(b"SSND")?;
            const INITIAL_SSND_SIZE: u32 = 8;
            self.handle.write_all(&INITIAL_SSND_SIZE.to_be_bytes())?;
            self.handle.write_all(&[
                0, 0, 0, 0,  0, 0, 0, 0,
            ])?;
        }
        Ok(())
    }

    /// Writes out missing samples and possibly adds one pad byte to the SSND chunk.
    fn finish_ssnd(&mut self) -> AifcResult<()> {
        if self.info.sample_format == SampleFormat::CompressedIma4 {
            // fill in the sample buffers to write out the last compressed packets
            while self.sample_buffer_pos != 0 || self.sample_buffer_ch != 0 {
                self.write_samples_i16(&[ 0 ])?;
            }
        }
        self.state = WriteState::SamplesDone;
        let sample_bytesize = calculate_ssnd_size(self.info.sample_format, self.samples_written,
            self.raw_data_frame_count.is_some())?;
        if !crate::is_even_u32(sample_bytesize) { // write one pad byte for SSND
            self.handle.write_all(&[ 0 ])?;
        }
        Ok(())
    }
}

/// Returns the number of sample frames for the COMM chunk.
fn calculate_num_sample_frames(sample_format: SampleFormat, sample_count: u64, num_channels: u64)
    -> AifcResult<u32> {
    // divisions are rounded up so that partially written samples are included
    let num_sample_frames = match sample_format {
        SampleFormat::CompressedIma4 => {
            // ima4 uses packet count per channel as numSampleFrames
            sample_count.div_ceil(64*num_channels)
        },
        _ => {
            sample_count.div_ceil(num_channels)
        }
    };
    u32::try_from(num_sample_frames).map_err(|_| AifcError::SizeTooLarge)
}

/// Returns the ssnd size based on sample format and sample count.
/// This includes the 8 start bytes, but not the pad byte.
fn calculate_ssnd_size(sample_format: SampleFormat, sample_count: u64, writing_raw_data: bool)
    -> AifcResult<u32> {

    let sample_byte_size = if !writing_raw_data {
        match sample_format {
            SampleFormat::CompressedIma4 => {
                sample_count.div_ceil(64) * 34
            },
            _ => {
                sample_count.checked_mul(sample_format.encoded_size())
                    .ok_or(AifcError::SizeTooLarge)?
            }
        }
    } else {
        sample_count
    };
    let ssnd_size = wchecked_add(sample_byte_size, 8)?;
    u32::try_from(ssnd_size).map_err(|_| AifcError::SizeTooLarge)
}

/// Returns the stream position to update the num_sampleframes value.
fn write_header(write: &mut dyn Write, info: &AifcWriteInfo, sample_size: u8) -> AifcResult<u64> {
    if info.channels < 1 {
        return Err(AifcError::InvalidNumberOfChannels);
    }
    if info.channels > info.sample_format.maximum_channel_count() {
        return Err(AifcError::InvalidNumberOfChannels);
    }
    if !info.sample_rate.is_finite() || info.sample_rate < 0.001 {
        return Err(AifcError::InvalidParameter);
    }
    let comm_num_sample_frames_pos;
    match info.file_format {
        crate::FileFormat::Aiff => {
            match info.sample_format {
                SampleFormat::I8 | SampleFormat::I16 | SampleFormat::I24 | SampleFormat::I32 => {},
                _ => { return Err(AifcError::InvalidParameter); }
            };
            comm_num_sample_frames_pos = 22;
            write.write_all(&[ b'F', b'O', b'R', b'M', 0, 0, 0, 30, b'A', b'I', b'F', b'F' ])?;
            write.write_all(&[ b'C', b'O', b'M', b'M', 0, 0, 0, 18 ])?;
            write.write_all(&info.channels.to_be_bytes())?; // num_channels
            write.write_all(&[0, 0, 0, 0 ])?; // num_sample_frames
            write.write_all(&[ 0, sample_size ])?;  // sample_size
            write.write_all(&crate::f80::f64_to_f80(info.sample_rate))?; // sample_rate
        },
        crate::FileFormat::Aifc => {
            const COMPR_LEN: u8 = 0;
            const COMPR_NAME: &[u8] = &[ 0 ];
            comm_num_sample_frames_pos = 34;
            write.write_all(&[ b'F', b'O', b'R', b'M', 0, 0, 0, 48, b'A', b'I', b'F', b'C' ])?;
            write.write_all(&[ b'F', b'V', b'E', b'R', 0, 0, 0, 4, 0xA2, 0x80, 0x51, 0x40 ])?;
            write.write_all(&[ b'C', b'O', b'M', b'M', 0, 0, 0, 24 ])?;
            write.write_all(&info.channels.to_be_bytes())?; // num_channels
            write.write_all(&[0, 0, 0, 0 ])?; // num_sample_frames
            write.write_all(&[ 0, sample_size ])?;  // sample_size
            write.write_all(&crate::f80::f64_to_f80(info.sample_rate))?; // sample_rate
            let compression_type = match info.sample_format {
                SampleFormat::U8 => crate::COMPRESSIONTYPE_RAW,
                SampleFormat::I8 => crate::COMPRESSIONTYPE_NONE,
                SampleFormat::I16 => crate::COMPRESSIONTYPE_NONE,
                SampleFormat::I16LE => crate::COMPRESSIONTYPE_SOWT,
                SampleFormat::I24 => crate::COMPRESSIONTYPE_NONE,
                SampleFormat::I32 => crate::COMPRESSIONTYPE_NONE,
                SampleFormat::I32LE => crate::COMPRESSIONTYPE_23NI,
                SampleFormat::F32 => crate::COMPRESSIONTYPE_FL32,
                SampleFormat::F64 => crate::COMPRESSIONTYPE_FL64,
                SampleFormat::CompressedUlaw => crate::COMPRESSIONTYPE_ULAW,
                SampleFormat::CompressedAlaw => crate::COMPRESSIONTYPE_ALAW,
                SampleFormat::CompressedIma4 => crate::COMPRESSIONTYPE_IMA4,
                SampleFormat::Custom(chid) => chid,
            };
            write.write_all(&compression_type)?;
            write.write_all(&[ COMPR_LEN ])?;    // compression name length
            write.write_all(COMPR_NAME)?;        // compression name
        },
    }
    Ok(comm_num_sample_frames_pos)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Cursor};

    #[test]
    fn test_new_transferring_ownership() -> AifcResult<()> {
        let mut output = vec![];
        let cursor = Cursor::new(&mut output);
        let info = AifcWriteInfo::default();
        let mut writer = AifcWriter::new(cursor, &info)?;
        writer.finalize()?;
        assert_eq!(output[0..12], [ b'F', b'O', b'R', b'M', 0, 0, 0, 48, b'A', b'I', b'F', b'C' ]);
        assert_eq!(output.len(), 56);
        Ok(())
    }

    #[test]
    fn test_new_with_mut_ref() -> AifcResult<()> {
        let mut output = vec![];
        let mut cursor = Cursor::new(&mut output);
        let info = AifcWriteInfo::default();
        let mut writer = AifcWriter::new(&mut cursor, &info)?;
        writer.finalize()?;
        assert_eq!(output[0..12], [ b'F', b'O', b'R', b'M', 0, 0, 0, 48, b'A', b'I', b'F', b'C' ]);
        assert_eq!(output.len(), 56);
        Ok(())
    }

    #[test]
    fn test_into_inner() -> AifcResult<()> {
        let mut output = vec![];
        {
            let cursor = Cursor::new(&mut output);
            let info = AifcWriteInfo::default();
            let writer = AifcWriter::new(cursor, &info)?;
            let mut w = writer.into_inner();
            w.write(&[ 0xff ])?;
            assert_eq!(w.position(), 57);
        }
        assert_eq!(output[0..12], [ b'F', b'O', b'R', b'M', 0, 0, 0, 48, b'A', b'I', b'F', b'C' ]);
        Ok(())
    }

    #[test]
    fn test_get_ref() -> AifcResult<()> {
        let mut output = vec![];
        let cursor = Cursor::new(&mut output);
        let info = AifcWriteInfo::default();
        let mut writer = AifcWriter::new(cursor, &info)?;
        let w = writer.get_ref();
        assert_eq!(w.position(), 56);
        writer.finalize()?;
        assert_eq!(output[0..12], [ b'F', b'O', b'R', b'M', 0, 0, 0, 48, b'A', b'I', b'F', b'C' ]);
        assert_eq!(output.len(), 56);
        Ok(())
    }

    #[test]
    fn test_get_mut() -> AifcResult<()> {
        let mut output = vec![];
        let cursor = Cursor::new(&mut output);
        let info = AifcWriteInfo::default();
        let mut writer = AifcWriter::new(cursor, &info)?;
        let w: &mut Cursor<&mut Vec<u8>> = writer.get_mut();
        // nasty write - don't do this in real life, because this may confuse the writer!
        // it's ok to do it here to test get_mut()
        w.write(&[ 88 ])?;
        assert_eq!(w.position(), 57);
        writer.write_samples_i16(&[ 134 ])?;
        // no finalize has been called, so form size hasn't been updated
        assert_eq!(output[0..12], [ b'F', b'O', b'R', b'M', 0, 0, 0, 48, b'A', b'I', b'F', b'C' ]);
        assert_eq!(output.len(), 75);
        Ok(())
    }

    fn create_aiff_with_sample_format(sample_format: SampleFormat) -> AifcResult<()> {
        let mut output = vec![];
        let mut wr = std::io::Cursor::new(&mut output);
        let winfo = AifcWriteInfo {
            file_format: FileFormat::Aiff,
            sample_format,
            ..AifcWriteInfo::default()
        };
        let res = AifcWriter::new(&mut wr, &winfo);
        match res {
            Ok(_) => Ok(()),
            Err(e) => Err(e)
        }
    }

    #[test]
    fn test_new_aiff() {
        assert!(create_aiff_with_sample_format(SampleFormat::I8).is_ok());
        assert!(create_aiff_with_sample_format(SampleFormat::I16).is_ok());
        assert!(create_aiff_with_sample_format(SampleFormat::I24).is_ok());
        assert!(create_aiff_with_sample_format(SampleFormat::I32).is_ok());
        // AIFF files don't support floats, compressed formats or custom formats
        assert!(matches!(create_aiff_with_sample_format(SampleFormat::I16LE),
            Err(AifcError::InvalidParameter)));
        assert!(matches!(create_aiff_with_sample_format(SampleFormat::I32LE),
            Err(AifcError::InvalidParameter)));
        assert!(matches!(create_aiff_with_sample_format(SampleFormat::U8),
            Err(AifcError::InvalidParameter)));
        assert!(matches!(create_aiff_with_sample_format(SampleFormat::F32),
            Err(AifcError::InvalidParameter)));
        assert!(matches!(create_aiff_with_sample_format(SampleFormat::F64),
            Err(AifcError::InvalidParameter)));
        assert!(matches!(create_aiff_with_sample_format(SampleFormat::CompressedUlaw),
            Err(AifcError::InvalidParameter)));
        assert!(matches!(create_aiff_with_sample_format(SampleFormat::CompressedAlaw),
            Err(AifcError::InvalidParameter)));
        assert!(matches!(create_aiff_with_sample_format(SampleFormat::Custom(*b"bad ")),
            Err(AifcError::InvalidParameter)));
    }

    #[test]
    fn test_write_samples() -> AifcResult<()> {
        for sample_format in [
            SampleFormat::U8, SampleFormat::I8, SampleFormat::I16,
            SampleFormat::I16LE, SampleFormat::I24, SampleFormat::I32, SampleFormat::I32LE,
            SampleFormat::F32, SampleFormat::F64, SampleFormat::CompressedUlaw,
            SampleFormat::CompressedAlaw, SampleFormat::CompressedIma4 ] {

            let winfo = AifcWriteInfo {
                file_format: FileFormat::Aifc,
                sample_rate: 44100.0,
                sample_format,
                channels: 1,
            };
            let mut output = vec![];
            let mut wr = std::io::Cursor::new(&mut output);
            let mut aifcwr = AifcWriter::new(&mut wr, &winfo)?;
            // check samples can be written
            match sample_format {
                SampleFormat::U8 => aifcwr.write_samples_u8(&[ 0 ])?,
                SampleFormat::I8 => aifcwr.write_samples_i8(&[ 0 ])?,
                SampleFormat::I16 => aifcwr.write_samples_i16(&[ 0 ])?,
                SampleFormat::I16LE => aifcwr.write_samples_i16(&[ 0 ])?,
                SampleFormat::I24 => aifcwr.write_samples_i24(&[ 0 ])?,
                SampleFormat::I32 => aifcwr.write_samples_i32(&[ 0 ])?,
                SampleFormat::I32LE => aifcwr.write_samples_i32(&[ 0 ])?,
                SampleFormat::F32 => aifcwr.write_samples_f32(&[ 0.0 ])?,
                SampleFormat::F64 => aifcwr.write_samples_f64(&[ 0.0 ])?,
                SampleFormat::CompressedUlaw => aifcwr.write_samples_i16(&[ 0 ])?,
                SampleFormat::CompressedAlaw => aifcwr.write_samples_i16(&[ 0 ])?,
                SampleFormat::CompressedIma4 => aifcwr.write_samples_i16(&[ 0 ])?,
                SampleFormat::Custom(_) => {},
            }
            aifcwr.finalize()?;
        }
        Ok(())
    }

    /// Returns a tuple containing form size, num_frames and ssnd size.
    fn read_sizes(cursor: &mut std::io::Cursor<&mut Vec<u8>>) -> AifcResult<(u32, u32, u32)> {
        let pos = cursor.stream_position()?;
        cursor.seek(SeekFrom::Start(4))?;
        let mut form_size = [0u8; 4];
        cursor.read_exact(&mut form_size)?;
        cursor.seek(SeekFrom::Start(22))?;
        let mut num_frames = [0u8; 4];
        cursor.read_exact(&mut num_frames)?;
        cursor.seek(SeekFrom::Start(42))?;
        let mut ssnd_size = [0u8; 4];
        cursor.read_exact(&mut ssnd_size)?;
        cursor.seek(SeekFrom::Start(pos))?;
        Ok((u32::from_be_bytes(form_size), u32::from_be_bytes(num_frames),
            u32::from_be_bytes(ssnd_size)))
    }

    #[test]
    fn test_write_samples_raw() -> AifcResult<()> {
        // raw sample data for Aiff 16-bit format
        let winfo = AifcWriteInfo {
            file_format: FileFormat::Aiff,
            sample_rate: 44100.0,
            sample_format: SampleFormat::I16,
            channels: 1,
        };
        let mut output = vec![];
        {
            let mut wr = std::io::Cursor::new(&mut output);
            let mut aifcwr = AifcWriter::new(&mut wr, &winfo)?;
            aifcwr.flush()?;
            aifcwr.write_samples_raw(&[ 10, 128 ], 1)?;
            assert_eq!(read_sizes(aifcwr.get_mut())?, (30, 0, 8));
            assert!(matches!(aifcwr.write_samples_i16(&[ 101, 102, 103 ]),
                Err(AifcError::InvalidWriteState)));
            assert_eq!(read_sizes(aifcwr.get_mut())?, (30, 0, 8));
            aifcwr.flush()?;
            assert_eq!(read_sizes(aifcwr.get_mut())?, (48, 1, 10));
            aifcwr.write_samples_raw(&[ 98, 12, 129, 99 ], 3)?;
            assert_eq!(read_sizes(aifcwr.get_mut())?, (48, 1, 10));
            aifcwr.flush()?;
            assert_eq!(read_sizes(aifcwr.get_mut())?, (52, 3, 14));
            aifcwr.finalize()?;
        }
        assert_eq!(output, &[ b'F', b'O', b'R', b'M', 0, 0, 0, 52, b'A', b'I', b'F', b'F',
            b'C', b'O', b'M', b'M', 0, 0, 0, 18,
            0, 1, 0, 0, 0, 3, 0, 16, 64, 14, 172, 68, 0, 0, 0, 0, 0, 0,
            b'S', b'S', b'N', b'D', 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 0,
            10, 128, 98, 12, 129, 99 ]);

        // raw sample data for Aifc custom compressed format
        let winfo = AifcWriteInfo {
            file_format: FileFormat::Aifc,
            sample_rate: 44100.0,
            sample_format: SampleFormat::Custom(*b"test"),
            channels: 1,
        };
        let mut output = vec![];
        {
            let mut wr = std::io::Cursor::new(&mut output);
            let mut aifcwr = AifcWriter::new(&mut wr, &winfo)?;
            aifcwr.flush()?;
            assert!(matches!(aifcwr.write_samples_i8(&[ 101, 102, 103 ]),
                Err(AifcError::InvalidSampleFormat)));
            aifcwr.write_samples_raw(&[ 10, 128, 98 ], 3)?;
            aifcwr.flush()?;
            aifcwr.write_samples_raw(&[ 12, 129 ], 8)?;
            aifcwr.flush()?;
            aifcwr.finalize()?;
        }
        assert_eq!(output, &[ b'F', b'O', b'R', b'M', 0, 0, 0, 70, b'A', b'I', b'F', b'C',
            b'F', b'V', b'E', b'R', 0, 0, 0, 4, 162, 128, 81, 64,
            b'C', b'O', b'M', b'M', 0, 0, 0, 24,
            0, 1, 0, 0, 0, 8, 0, 0, 64, 14, 172, 68, 0, 0, 0, 0, 0, 0,
            b't', b'e', b's', b't',
            0, 0,
            b'S', b'S', b'N', b'D', 0, 0, 0, 13, 0, 0, 0, 0, 0, 0, 0, 0, 10, 128, 98, 12, 129, 0 ]);
        Ok(())
    }

    #[test]
    fn test_flush() -> AifcResult<()> {
        // checks that FORM, COMM and SSND chunks are updated correctly
        let winfo = AifcWriteInfo {
            file_format: FileFormat::Aiff,
            sample_rate: 44100.0,
            sample_format: SampleFormat::I8,
            channels: 1,
        };
        let mut output = vec![];
        let mut wr = std::io::Cursor::new(&mut output);
        let mut aifcwr = AifcWriter::new(&mut wr, &winfo)?;
        aifcwr.flush()?;
        aifcwr.write_samples_i8(&[ 10, 11 ])?;
        assert_eq!(read_sizes(aifcwr.get_mut())?, (30, 0, 8));
        aifcwr.flush()?;
        assert_eq!(read_sizes(aifcwr.get_mut())?, (48, 2, 10));
        aifcwr.write_samples_i8(&[ 12, 13 ])?;
        assert_eq!(read_sizes(aifcwr.get_mut())?, (48, 2, 10));
        aifcwr.flush()?;
        assert_eq!(read_sizes(aifcwr.get_mut())?, (50, 4, 12));
        aifcwr.finalize()?;
        assert_eq!(output, &[ b'F', b'O', b'R', b'M', 0, 0, 0, 50, b'A', b'I', b'F', b'F',
            b'C', b'O', b'M', b'M', 0, 0, 0, 18,
            0, 1, 0, 0, 0, 4, 0, 8, 64, 14, 172, 68, 0, 0, 0, 0, 0, 0,
            b'S', b'S', b'N', b'D', 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 10, 11, 12, 13 ]);
        Ok(())
    }

    #[test]
    fn test_finalize() -> AifcResult<()> {
        // checks that FORM, COMM and SSND chunks are updated correctly
        // FileFormat: Aiff
        let winfo = AifcWriteInfo {
            file_format: FileFormat::Aiff,
            sample_rate: 44100.0,
            sample_format: SampleFormat::I8,
            channels: 1,
        };
        let mut output = vec![];
        let mut wr = std::io::Cursor::new(&mut output);
        let mut aifcwr = AifcWriter::new(&mut wr, &winfo)?;
        aifcwr.write_samples_i8(&[ 10, 11, 12, 13 ])?;
        aifcwr.finalize()?;
        assert_eq!(aifcwr.get_ref().position(), 58);
        assert_eq!(output, &[ b'F', b'O', b'R', b'M', 0, 0, 0, 50, b'A', b'I', b'F', b'F',
            b'C', b'O', b'M', b'M', 0, 0, 0, 18,
            0, 1, 0, 0, 0, 4, 0, 8, 64, 14, 172, 68, 0, 0, 0, 0, 0, 0,
            b'S', b'S', b'N', b'D', 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 10, 11, 12, 13 ]);

        // FileFormat: Aifc
        let winfo = AifcWriteInfo {
            file_format: FileFormat::Aifc,
            sample_rate: 44100.0,
            sample_format: SampleFormat::I8,
            channels: 1,
        };
        let mut output = vec![];
        let mut wr = std::io::Cursor::new(&mut output);
        let mut aifcwr = AifcWriter::new(&mut wr, &winfo)?;
        aifcwr.write_samples_i8(&[ 10, 11, 12, 13 ])?;
        aifcwr.finalize()?;
        assert_eq!(aifcwr.get_ref().position(), 76);
        assert_eq!(output, &[ b'F', b'O', b'R', b'M', 0, 0, 0, 68, b'A', b'I', b'F', b'C',
            b'F', b'V', b'E', b'R', 0, 0, 0, 4, 162, 128, 81, 64,
            b'C', b'O', b'M', b'M', 0, 0, 0, 24,
            0, 1, 0, 0, 0, 4, 0, 8, 64, 14, 172, 68, 0, 0, 0, 0, 0, 0,
            b'N', b'O', b'N', b'E',
            0, 0,
            b'S', b'S', b'N', b'D', 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 10, 11, 12, 13 ]);
        Ok(())
    }

    #[test]
    fn test_finalize_for_ima4() -> AifcResult<()> {
        let winfo = AifcWriteInfo {
            file_format: FileFormat::Aifc,
            sample_rate: 44100.0,
            sample_format: SampleFormat::CompressedIma4,
            channels: 1,
        };
        let mut output = vec![];
        let mut wr = std::io::Cursor::new(&mut output);
        let mut aifcwr = AifcWriter::new(&mut wr, &winfo)?;
        aifcwr.write_samples_i16(&[ 10, 11, 12, 13 ])?;
        aifcwr.finalize()?;
        assert_eq!(output, [ b'F', b'O', b'R', b'M', 0, 0, 0, 98, b'A', b'I', b'F', b'C',
            b'F', b'V', b'E', b'R', 0, 0, 0, 4, 162, 128, 81, 64,
            b'C', b'O', b'M', b'M', 0, 0, 0, 24,
            0, 1, 0, 0, 0, 1, 0, 0, 64, 14, 172, 68, 0, 0, 0, 0, 0, 0,
            b'i', b'm', b'a', b'4',
            0, 0,
            b'S', b'S', b'N', b'D', 0, 0, 0, 42, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0,
            6, 0, 13, 8, 8, 8, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]);
        Ok(())
    }

    #[test]
    fn test_finalize_ssnd_padding() -> AifcResult<()> {
        let mut output = vec![];
        let winfo = AifcWriteInfo {
            file_format: FileFormat::Aifc,
            channels: 1,
            sample_format: SampleFormat::I8,
            ..AifcWriteInfo::default()
        };
        let mut wr = std::io::Cursor::new(&mut output);
        let mut writer = AifcWriter::new(&mut wr, &winfo)?;
        writer.write_samples_i8(&[ 11, 12, 13 ])?;
        writer.finalize()?;
        assert_eq!(output, &[ b'F', b'O', b'R', b'M', 0, 0, 0, 68, b'A', b'I', b'F', b'C',
            b'F', b'V', b'E', b'R', 0, 0, 0, 4, 162, 128, 81, 64,
            b'C', b'O', b'M', b'M', 0, 0, 0, 24,
            0, 1, 0, 0, 0, 3, 0, 8, 64, 14, 172, 68, 0, 0, 0, 0, 0, 0,
            b'N', b'O', b'N', b'E',
            0, 0,
            b'S', b'S', b'N', b'D', 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 0, 0,
            11, 12, 13, 0 ]);
        Ok(())
    }

    #[test]
    fn test_finalize_with_missing_samples() -> AifcResult<()> {
        // SampleFormat::I16 and 2 channels writing 6 samples out of 6 (no missing samples)
        let mut output = vec![];
        {
            let winfo = AifcWriteInfo {
                file_format: FileFormat::Aifc,
                channels: 2,
                ..AifcWriteInfo::default()
            };
            let mut wr = std::io::Cursor::new(&mut output);
            let mut writer = AifcWriter::new(&mut wr, &winfo)?;
            writer.write_samples_i16(&[ 11, 12, 13, 14, 15, 16 ])?;
            writer.finalize()?;
        }
        assert_eq!(output, &[ b'F', b'O', b'R', b'M', 0, 0, 0, 76, b'A', b'I', b'F', b'C',
            b'F', b'V', b'E', b'R', 0, 0, 0, 4, 162, 128, 81, 64,
            b'C', b'O', b'M', b'M', 0, 0, 0, 24,
            0, 2, 0, 0, 0, 3, 0, 16, 64, 14, 172, 68, 0, 0, 0, 0, 0, 0,
            b'N', b'O', b'N', b'E',
            0, 0,
            b'S', b'S', b'N', b'D', 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 11, 0, 12, 0, 13, 0, 14, 0, 15, 0, 16 ]);

        // SampleFormat::I16 and 5 channels writing 7 samples out of 10 (3 samples missing)
        let mut output = vec![];
        {
            let winfo = AifcWriteInfo {
                file_format: FileFormat::Aifc,
                channels: 5,
                ..AifcWriteInfo::default()
            };
            let mut wr = std::io::Cursor::new(&mut output);
            let mut writer = AifcWriter::new(&mut wr, &winfo)?;
            writer.write_samples_i16(&[ 11, 12, 13, 14, 15, 16, 17 ])?;
            writer.finalize()?;
        }
        assert_eq!(output, &[ b'F', b'O', b'R', b'M', 0, 0, 0, 78, b'A', b'I', b'F', b'C',
            b'F', b'V', b'E', b'R', 0, 0, 0, 4, 162, 128, 81, 64,
            b'C', b'O', b'M', b'M', 0, 0, 0, 24,
            0, 5, 0, 0, 0, 2, 0, 16, 64, 14, 172, 68, 0, 0, 0, 0, 0, 0,
            b'N', b'O', b'N', b'E',
            0, 0,
            b'S', b'S', b'N', b'D', 0, 0, 0, 22, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 11, 0, 12, 0, 13, 0, 14, 0, 15, 0, 16, 0, 17 ]);

        // SampleFormat::I16 and 5 channels writing 7 raw sample bytes out of 20 bytes
        let mut output = vec![];
        {
            let winfo = AifcWriteInfo {
                file_format: FileFormat::Aifc,
                channels: 5,
                ..AifcWriteInfo::default()
            };
            let mut wr = std::io::Cursor::new(&mut output);
            let mut writer = AifcWriter::new(&mut wr, &winfo)?;
            writer.write_samples_raw(&[ 11, 12, 13, 14, 15, 16, 17 ], 4)?;
            writer.finalize()?;
        }
        assert_eq!(output, &[ b'F', b'O', b'R', b'M', 0, 0, 0, 72, b'A', b'I', b'F', b'C',
            b'F', b'V', b'E', b'R', 0, 0, 0, 4, 162, 128, 81, 64,
            b'C', b'O', b'M', b'M', 0, 0, 0, 24,
            0, 5, 0, 0, 0, 4, 0, 16, 64, 14, 172, 68, 0, 0, 0, 0, 0, 0,
            b'N', b'O', b'N', b'E',
            0, 0,
            b'S', b'S', b'N', b'D', 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0,
            11, 12, 13, 14, 15, 16, 17, 0 ]);

        // SampleFormat::I16 and 5 channels writing 0 samples (no missing samples)
        let mut output = vec![];
        {
            let winfo = AifcWriteInfo {
                file_format: FileFormat::Aifc,
                channels: 5,
                ..AifcWriteInfo::default()
            };
            let mut wr = std::io::Cursor::new(&mut output);
            let mut writer = AifcWriter::new(&mut wr, &winfo)?;
            writer.write_samples_i16(&[])?;
            writer.finalize()?;
        }
        assert_eq!(output, &[ b'F', b'O', b'R', b'M', 0, 0, 0, 48, b'A', b'I', b'F', b'C',
            b'F', b'V', b'E', b'R', 0, 0, 0, 4, 162, 128, 81, 64,
            b'C', b'O', b'M', b'M', 0, 0, 0, 24,
            0, 5, 0, 0, 0, 0, 0, 16, 64, 14, 172, 68, 0, 0, 0, 0, 0, 0,
            b'N', b'O', b'N', b'E',
            0, 0 ]);

        // SampleFormat::Custom and 5 channels writing 7 raw sample bytes (no missing samples)
        // SSND is padded with one byte
        let mut output = vec![];
        {
            let winfo = AifcWriteInfo {
                file_format: FileFormat::Aifc,
                channels: 5,
                sample_rate: 44100.0,
                sample_format: SampleFormat::Custom(*b"test"),
            };
            let mut wr = std::io::Cursor::new(&mut output);
            let mut writer = AifcWriter::new(&mut wr, &winfo)?;
            writer.write_samples_raw(&[ 11, 12, 13, 14, 15, 16, 17 ], 7)?;
            writer.finalize()?;
        }
        assert_eq!(output, &[ b'F', b'O', b'R', b'M', 0, 0, 0, 72, b'A', b'I', b'F', b'C',
            b'F', b'V', b'E', b'R', 0, 0, 0, 4, 162, 128, 81, 64,
            b'C', b'O', b'M', b'M', 0, 0, 0, 24,
            0, 5, 0, 0, 0, 7, 0, 0, 64, 14, 172, 68, 0, 0, 0, 0, 0, 0,
            b't', b'e', b's', b't',
            0, 0,
            b'S', b'S', b'N', b'D', 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0,
            11, 12, 13, 14, 15, 16, 17, 0 ]);

        Ok(())
    }

    #[test]
    fn test_write_chunk() -> AifcResult<()> {
        let mut output = vec![];
        let winfo = AifcWriteInfo {
            file_format: FileFormat::Aifc,
            ..AifcWriteInfo::default()
        };
        let mut wr = std::io::Cursor::new(&mut output);
        let mut writer = AifcWriter::new(&mut wr, &winfo)?;
        writer.write_chunk(b"TEST", &[ 11, 12, 13, 14, 15 ])?;
        writer.write_samples_i16(&[ 11, 12, 13, 14, 15, 16 ])?;
        writer.finalize()?;
        assert_eq!(output, &[ b'F', b'O', b'R', b'M', 0, 0, 0, 90, b'A', b'I', b'F', b'C',
            b'F', b'V', b'E', b'R', 0, 0, 0, 4, 162, 128, 81, 64,
            b'C', b'O', b'M', b'M', 0, 0, 0, 24,
            0, 2, 0, 0, 0, 3, 0, 16, 64, 14, 172, 68, 0, 0, 0, 0, 0, 0,
            b'N', b'O', b'N', b'E',
            0, 0,
            b'T', b'E', b'S', b'T', 0, 0, 0, 5, 11, 12, 13, 14, 15, 0,
            b'S', b'S', b'N', b'D', 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 11, 0, 12, 0, 13, 0, 14, 0, 15, 0, 16 ]);
        Ok(())
    }

    #[test]
    fn test_write_chunk_markers() -> AifcResult<()> {
        let mut output = vec![];
        let winfo = AifcWriteInfo {
            file_format: FileFormat::Aifc,
            ..AifcWriteInfo::default()
        };
        let markers = [ Marker { id: 8, position: 12468, name: &[ b'm', b'y', b'm' ] } ];
        let mut wr = std::io::Cursor::new(&mut output);
        let mut writer = AifcWriter::new(&mut wr, &winfo)?;
        writer.write_samples_i16(&[ 11, 12, 13, 14, 15, 16 ])?;
        writer.write_chunk_markers(&markers)?;
        writer.finalize()?;
        assert_eq!(output, &[ b'F', b'O', b'R', b'M', 0, 0, 0, 96, b'A', b'I', b'F', b'C',
            b'F', b'V', b'E', b'R', 0, 0, 0, 4, 162, 128, 81, 64,
            b'C', b'O', b'M', b'M', 0, 0, 0, 24,
            0, 2, 0, 0, 0, 3, 0, 16, 64, 14, 172, 68, 0, 0, 0, 0, 0, 0,
            b'N', b'O', b'N', b'E',
            0, 0,
            b'S', b'S', b'N', b'D', 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 11, 0, 12, 0, 13, 0, 14, 0, 15, 0, 16,
            b'M', b'A', b'R', b'K', 0, 0, 0, 12,  0, 1, 0, 8,  0, 0, 48, 180,  3, 109, 121, 109
        ]);
        Ok(())
    }

    #[test]
    fn test_write_chunk_comments() -> AifcResult<()> {
        let mut output = vec![];
        let winfo = AifcWriteInfo {
            file_format: FileFormat::Aifc,
            ..AifcWriteInfo::default()
        };
        let comments = [ Comment { timestamp: 0, marker_id: 6, text: &[ b'c', b'm' ] } ];
        let mut wr = std::io::Cursor::new(&mut output);
        let mut writer = AifcWriter::new(&mut wr, &winfo)?;
        writer.write_samples_i16(&[ 11, 12, 13, 14, 15, 16 ])?;
        writer.write_chunk_comments(&comments)?;
        writer.finalize()?;
        assert_eq!(output, &[ b'F', b'O', b'R', b'M', 0, 0, 0, 96, b'A', b'I', b'F', b'C',
            b'F', b'V', b'E', b'R', 0, 0, 0, 4, 162, 128, 81, 64,
            b'C', b'O', b'M', b'M', 0, 0, 0, 24,
            0, 2, 0, 0, 0, 3, 0, 16, 64, 14, 172, 68, 0, 0, 0, 0, 0, 0,
            b'N', b'O', b'N', b'E',
            0, 0,
            b'S', b'S', b'N', b'D', 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 11, 0, 12, 0, 13, 0, 14, 0, 15, 0, 16,
            b'C', b'O', b'M', b'T', 0, 0, 0, 12,  0, 1,  0, 0, 0, 0,  0, 6,  0, 2, 99, 109
        ]);
        Ok(())
    }

    #[test]
    fn test_write_chunk_id3() -> AifcResult<()> {
        let mut output = vec![];
        let winfo = AifcWriteInfo {
            file_format: FileFormat::Aifc,
            ..AifcWriteInfo::default()
        };
        let mut wr = std::io::Cursor::new(&mut output);
        let mut writer = AifcWriter::new(&mut wr, &winfo)?;
        writer.write_samples_i16(&[ 11, 12, 13, 14, 15, 16 ])?;
        // some id3 data (invalid data for a ID3 chunk, but we don't care)
        writer.write_chunk_id3(&[ 61, 62, 63, 64, 65, 66 ])?;
        writer.finalize()?;
        assert_eq!(output, &[ b'F', b'O', b'R', b'M', 0, 0, 0, 90, b'A', b'I', b'F', b'C',
            b'F', b'V', b'E', b'R', 0, 0, 0, 4, 162, 128, 81, 64,
            b'C', b'O', b'M', b'M', 0, 0, 0, 24,
            0, 2, 0, 0, 0, 3, 0, 16, 64, 14, 172, 68, 0, 0, 0, 0, 0, 0,
            b'N', b'O', b'N', b'E',
            0, 0,
            b'S', b'S', b'N', b'D', 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 11, 0, 12, 0, 13, 0, 14, 0, 15, 0, 16,
            b'I', b'D', b'3', b' ', 0, 0, 0, 6, 61, 62, 63, 64, 65, 66,
        ]);
        Ok(())
    }

    #[test]
    fn test_writing_after_dummy_data() -> AifcResult<()> {
        // checks that FORM, COMM and SSND chunks are updated correctly relative to the FORM start
        let winfo = AifcWriteInfo {
            file_format: FileFormat::Aiff,
            sample_rate: 44100.0,
            sample_format: SampleFormat::I8,
            channels: 1,
        };
        let mut output = vec![];
        let mut wr = std::io::Cursor::new(&mut output);
        wr.write_all(&[ 1, 2, 3, 4 ])?;
        let mut aifcwr = AifcWriter::new(&mut wr, &winfo)?;
        aifcwr.write_samples_i8(&[ 10, 11, 12, 13 ])?;
        aifcwr.finalize()?;
        assert_eq!(aifcwr.get_ref().position(), 62);
        assert_eq!(output, &[ 1, 2, 3, 4,
            b'F', b'O', b'R', b'M', 0, 0, 0, 50, b'A', b'I', b'F', b'F',
            b'C', b'O', b'M', b'M', 0, 0, 0, 18,
            0, 1, 0, 0, 0, 4, 0, 8, 64, 14, 172, 68, 0, 0, 0, 0, 0, 0,
            b'S', b'S', b'N', b'D', 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 10, 11, 12, 13 ]);
        Ok(())
    }

    #[test]
    fn test_drop() -> AifcResult<()> {
        // checks that finalize isn't called when AifcWriter is dropped
        let winfo = AifcWriteInfo {
            file_format: FileFormat::Aiff,
            sample_rate: 44100.0,
            sample_format: SampleFormat::I8,
            channels: 1,
        };
        let mut output = vec![];
        {
            let mut wr = std::io::Cursor::new(&mut output);
            let mut aifcwr = AifcWriter::new(&mut wr, &winfo)?;
            aifcwr.write_samples_i8(&[ 10, 11, 12 ])?;
            assert_eq!(aifcwr.get_ref().position(), 57);
        }
        assert_eq!(output, &[ b'F', b'O', b'R', b'M', 0, 0, 0, 30, b'A', b'I', b'F', b'F',
            b'C', b'O', b'M', b'M', 0, 0, 0, 18,
            0, 1, 0, 0, 0, 0, 0, 8, 64, 14, 172, 68, 0, 0, 0, 0, 0, 0,
            b'S', b'S', b'N', b'D', 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 10, 11, 12 ]);
        Ok(())
    }
}
