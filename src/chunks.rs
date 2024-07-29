// Data structs and iterators for markers, comments and instrument.

use crate::{cast, AifcError, AifcResult, MarkerId, Write};

fn read_u16_from_pos(data: &[u8], pos: &mut usize) -> AifcResult<u16> {
    let Some(pos_end) = pos.checked_add(2) else {
        return Err(AifcError::StdIoError(crate::unexpectedeof()));
    };
    if pos_end > data.len() {
        return Err(AifcError::StdIoError(crate::unexpectedeof()));
    }
    let mut buf = [0u8; 2];
    buf.copy_from_slice(&data[*pos..pos_end]);
    *pos = pos_end;
    Ok(u16::from_be_bytes(buf))
}

fn read_i16_from_pos(data: &[u8], pos: &mut usize) -> AifcResult<i16> {
    let Some(pos_end) = pos.checked_add(2) else {
        return Err(AifcError::StdIoError(crate::unexpectedeof()));
    };
    if pos_end > data.len() {
        return Err(AifcError::StdIoError(crate::unexpectedeof()));
    }
    let mut buf = [0u8; 2];
    buf.copy_from_slice(&data[*pos..pos_end]);
    *pos = pos_end;
    Ok(i16::from_be_bytes(buf))
}

fn read_u32_from_pos(data: &[u8], pos: &mut usize) -> AifcResult<u32> {
    let Some(pos_end) = pos.checked_add(4) else {
        return Err(AifcError::StdIoError(crate::unexpectedeof()));
    };
    if pos_end > data.len() {
        return Err(AifcError::StdIoError(crate::unexpectedeof()));
    }
    let mut buf = [0u8; 4];
    buf.copy_from_slice(&data[*pos..pos_end]);
    *pos = pos_end;
    Ok(u32::from_be_bytes(buf))
}

fn read_pstring_from_pos<'a>(data: &'a [u8], pos: &mut usize) -> AifcResult<&'a [u8]> {
    let Some(pos_end) = pos.checked_add(1) else {
        return Err(AifcError::StdIoError(crate::unexpectedeof()));
    };
    if pos_end > data.len() {
        return Err(AifcError::StdIoError(crate::unexpectedeof()));
    }
    let pstr_len = usize::from(data[*pos]);
    let Some(pos2_end) = pos_end.checked_add(pstr_len) else {
        return Err(AifcError::StdIoError(crate::unexpectedeof()));
    };
    if pos2_end > data.len() {
        return Err(AifcError::StdIoError(crate::unexpectedeof()));
    }
    let pstr = &data[*pos+1..pos2_end];
    *pos = pos2_end;
    if crate::is_even_usize(pstr_len) { // if pstr_len is even, read pad byte
        // there's no check that pos + 1 isn't outside data
        // to allow the pad byte to be shared with the chunk.
        // for instance, chunk size may be 9 and pstring len correctly ends at the same byte,
        // but there may be a pad byte for pstring which is also a pad byte for the chunk
        *pos = pos.checked_add(1).ok_or(AifcError::StdIoError(crate::unexpectedeof()))?;
    }
    Ok(pstr)
}

/// Marker data.
#[derive(Debug, Clone, PartialEq)]
pub struct Marker<'a> {
    /// Marker id. This must be a value greater than 0. Note: there isn't any validation that
    /// the marker id is actually a positive number.
    pub id: MarkerId,
    /// Marker frame position. The value 0 means the position before the first frame.
    /// The value 1 means the position after the first frame and before
    /// the second frame. The position is in audio frames, not samples or bytes.
    pub position: u32,
    /// Marker name. The maximum length is 255 bytes.
    pub name: &'a [u8],
}

impl<'a> Marker<'a> {
    /// Writes the given markers to the given stream. The maximum marker count is 65535.
    /// The first bytes written contains the marker count.
    pub fn write_chunk_data(write: &mut dyn Write, markers: &[Marker]) -> AifcResult<()> {
        let mlen = u16::try_from(markers.len()).map_err(|_| AifcError::SizeTooLarge)?;
        write.write_all(&mlen.to_be_bytes())?;
        for marker in markers {
            write.write_all(&marker.id.to_be_bytes())?;
            write.write_all(&marker.position.to_be_bytes())?;
            let namelen = u8::try_from(marker.name.len()).map_err(|_| AifcError::SizeTooLarge)?;
            write.write_all(&[ namelen ])?;
            write.write_all(marker.name)?;
            if crate::is_even_usize(marker.name.len()) { // pad byte for name
                write.write_all(&[ 0 ])?;
            }
        }
        Ok(())
    }

    /// Returns the byte size of the marker chunk for the given markers.
    pub fn chunk_data_size(markers: &[Marker]) -> AifcResult<u32> {
        let mut size_calc = SizeCalculator::new();
        Marker::write_chunk_data(&mut size_calc, markers)?;
        Ok(size_calc.size)
    }
}

/// Iterator to read markers from a slice of bytes.
///
/// The iterator returns [`Marker`]s or errors if the parser runs out of data.
/// The iterator returns at most 65535 items (`Marker`s or errors).
///
/// See [`AifcReader::read_chunk_markers()`](crate::AifcReader::read_chunk_markers())
/// for an example.
pub struct Markers<'a> {
    slice: &'a [u8],
    slice_pos: usize,
    num_markers: usize,
}

impl<'a> Markers<'a> {
    /// Creates a new marker iterator to read markers from the given slice of bytes.
    /// [`Marker`]s returned by `next()` have the same lifetimes as `slice`
    /// because they reference `slice` for [`Marker::name`].
    pub fn new(slice: &'a [u8]) -> AifcResult<Markers<'a>> {
        let mut slice_pos: usize = 0;
        let num_markers = usize::from(read_u16_from_pos(slice, &mut slice_pos)?);
        Ok(Markers {
            slice,
            slice_pos,
            num_markers,
        })
    }
}

/// Iterator implementation for markers.
impl<'a> Iterator for Markers<'a> {
    type Item = AifcResult<Marker<'a>>;

    /// Reads the next marker.
    fn next(&mut self) -> Option<Self::Item> {
        if self.num_markers == 0 {
            return None;
        }
        self.num_markers -= 1;
        let id = match read_i16_from_pos(self.slice, &mut self.slice_pos) {
            Ok(id) => id,
            Err(e) => { return Some(Err(e)); }
        };
        let position = match read_u32_from_pos(self.slice, &mut self.slice_pos) {
            Ok(position) => position,
            Err(e) => { return Some(Err(e)); }
        };
        let name = match read_pstring_from_pos(self.slice, &mut self.slice_pos) {
            Ok(name) => name,
            Err(e) => { return Some(Err(e)); }
        };
        Some(Ok(Marker { id, position, name }))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.num_markers, Some(self.num_markers))
    }
}

impl<'a> ExactSizeIterator for Markers<'a> {
    // just use the default implementation
}

/// Comment data.
#[derive(Debug, Clone, PartialEq)]
pub struct Comment<'a> {
    /// The timestamp of the comment. The value 0 usually means that no timestamp has been set.
    /// The value is the number of seconds since January 1, 1904.
    /// The [`unix_timestamp()`](Comment::unix_timestamp) and
    /// [`set_unix_timestamp()`](Comment::set_unix_timestamp) methods can be used to manipulate
    /// this value as a UNIX timestamp.
    pub timestamp: u32,
    /// Optional marker id for the comment. The value 0 means that there is no marker associated
    /// with the comment. The value should be a positive number, but there isn't any validation
    /// for it.
    ///
    /// If comments with non-zero marker ids are written to a stream, then also markers
    /// with the same marker ids should be written.
    pub marker_id: MarkerId,
    /// Text of the comment. The maximum length is 65535 bytes.
    pub text: &'a [u8],
}

impl<'a> Comment<'a> {
    /// Returns the comment timestamp (relative to 1904-01-01T00:00:00) as
    /// a UNIX timestamp (relative to 1970-01-01T00:00:00). The result can be a positive or
    /// a negative value.
    pub fn unix_timestamp(&self) -> i64 {
        i64::from(self.timestamp) - crate::UNIX_TIMESTAMP_OFFSET
    }

    /// Sets the comment timestamp (relative to 1904-01-01T00:00:00) to
    /// a UNIX timestamp (relative to 1970-01-01T00:00:00).
    /// If the UNIX timestamp is outside the bounds of the comment timestamp,
    /// the `TimestampOutOfBounds` error is returned.
    pub fn set_unix_timestamp(&mut self, unix_timestamp: i64) -> AifcResult<()> {
        self.timestamp = unix_timestamp
            .checked_add(crate::UNIX_TIMESTAMP_OFFSET)
            .and_then(|val| val.try_into().ok())
            .ok_or(AifcError::TimestampOutOfBounds)?;
        Ok(())
    }

    /// Writes the given comments to the given write stream. The maximum comment count is 65535.
    /// The first bytes written contains the comment count.
    pub fn write_chunk_data(write: &mut dyn Write, comments: &[Comment]) -> AifcResult<()> {
        let clen = u16::try_from(comments.len()).map_err(|_| AifcError::SizeTooLarge)?;
        write.write_all(&(clen).to_be_bytes())?;
        for comment in comments {
            write.write_all(&comment.timestamp.to_be_bytes())?;
            write.write_all(&comment.marker_id.to_be_bytes())?;
            let textlen = u16::try_from(comment.text.len())
                .map_err(|_| AifcError::SizeTooLarge)?;
            write.write_all(&textlen.to_be_bytes())?;
            write.write_all(comment.text)?;
            if !crate::is_even_usize(comment.text.len()) { // pad byte for text
                write.write_all(&[ 0 ])?;
            }
        }
        Ok(())
    }

    /// Returns the byte size of the comment chunk for the given comments.
    pub fn chunk_data_size(comments: &[Comment]) -> AifcResult<u32> {
        let mut size_calc = SizeCalculator::new();
        Comment::write_chunk_data(&mut size_calc, comments)?;
        Ok(size_calc.size)
    }
}

/// Iterator to read comments from a slice of bytes.
///
/// The iterator returns [`Comment`]s or errors if the parser runs out of data.
/// The iterator returns at most 65535 items (`Comment`s or errors).
///
/// See [`AifcReader::read_chunk_comments()`](crate::AifcReader::read_chunk_comments())
/// for an example.
pub struct Comments<'a> {
    slice: &'a [u8],
    slice_pos: usize,
    num_comments: usize,
}

impl<'a> Comments<'a> {
    /// Creates a new comment iterator to read comments from the given slice of bytes.
    /// [`Comment`]s returned by `next()` have the same lifetimes as `slice`
    /// because they reference `slice` for [`Comment::text`].
    pub fn new(slice: &'a [u8]) -> AifcResult<Comments<'a>> {
        let mut slice_pos: usize = 0;
        let num_comments = usize::from(read_u16_from_pos(slice, &mut slice_pos)?);
        Ok(Comments {
            slice,
            slice_pos,
            num_comments,
        })
    }
}

/// Iterator implementation for comments.
impl<'a> Iterator for Comments<'a> {
    type Item = AifcResult<Comment<'a>>;

    /// Reads the next comment.
    fn next(&mut self) -> Option<Self::Item> {
        if self.num_comments == 0 {
            return None;
        }
        self.num_comments -= 1;
        let timestamp = match read_u32_from_pos(self.slice, &mut self.slice_pos) {
            Ok(timestamp) => timestamp,
            Err(e) => { return Some(Err(e)); }
        };
        let marker_id = match read_i16_from_pos(self.slice, &mut self.slice_pos) {
            Ok(marker_id) => marker_id,
            Err(e) => { return Some(Err(e)); }
        };
        // read text
        let count = match read_u16_from_pos(self.slice, &mut self.slice_pos) {
            Ok(count) => count,
            Err(e) => { return Some(Err(e)); }
        };
        let count = usize::from(count);
        let Some(new_slice_pos) = self.slice_pos.checked_add(count) else {
            return Some(Err(AifcError::StdIoError(crate::unexpectedeof())));
        };
        if new_slice_pos > self.slice.len() {
            return Some(Err(AifcError::StdIoError(crate::unexpectedeof())));
        }
        let text = &self.slice[self.slice_pos..new_slice_pos];
        self.slice_pos = new_slice_pos;
        if !crate::is_even_usize(count) { // if count is odd, skip the pad byte
            // there's no check that pos + 1 isn't outside data (slice.len())
            // to allow the pad byte to be shared with the chunk.
            // (however, there is an overflow check)
            self.slice_pos = match self.slice_pos.checked_add(1) {
                Some(s) => s,
                None => { return Some(Err(AifcError::StdIoError(crate::unexpectedeof()))); }
            };
        }
        Some(Ok(Comment { timestamp, marker_id, text }))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.num_comments, Some(self.num_comments))
    }
}

impl<'a> ExactSizeIterator for Comments<'a> {
    // just use the default implementation
}

/// Instrument loop values.
///
/// The `begin_loop` marker position must be before the `end_loop` marker
/// position, otherwise the loop should be ignored and no looping happens.
#[derive(Debug, Clone, PartialEq)]
pub struct Loop {
    /// Play mode describing how to loop the instrument samples.
    ///
    /// 0 = no looping, 1 = the loop segment is repeatedly played from its start to end,
    /// 2 = the loop segment is repeatedly played from its start to end and then
    /// in reverse from its end to start.
    pub play_mode: i16,
    /// Marker id for the beginning of the loop segment.
    pub begin_loop: MarkerId,
    /// Marker id for the end of the loop segment.
    pub end_loop: MarkerId,
}

/// Instrument data.
#[derive(Debug, Clone, PartialEq)]
pub struct Instrument {
    /// The MIDI note number for the base note of the MIDI instrument.
    /// This should be a value in the range 0 to 127. A value of 60 represents MIDI middle C.
    pub base_note: i8,
    /// Tuning adjustment to the sound if it wasn't recorded exactly in tune.
    /// The value should be in the range -50 to 50 and its unit is cents (1/100 of a semitone).
    /// Negative values means that the pitch of the sound should be lowered
    /// and positive values mean that it should be raised.
    pub detune: i8,
    /// The lowest useful MIDI note value for this sound.
    pub low_note: i8,
    /// The highest useful MIDI note value for this sound.
    pub high_note: i8,
    /// The lowest useful MIDI velocity value for this sound.
    /// This should be a value in the range 1 (lowest velocity) to 127 (highest velocity).
    pub low_velocity: i8,
    /// The highest useful MIDI velocity value for this sound.
    /// This should be a value in the range 1 (lowest velocity) to 127 (highest velocity).
    pub high_velocity: i8,
    /// The amount to change the gain of the sound in decibels. The value 0 means no change.
    /// The value 6 means 6 db louder sound (sample values doubled) and the value -6 means 6 db
    /// quieter sound (sample values halved).
    pub gain: i16,
    /// Loop to use for the MIDI instrument's "sustain" stage.
    pub sustain_loop: Loop,
    /// Loop to use for the MIDI instrument's "release" stage.
    pub release_loop: Loop
}

impl Instrument {

    /// Reads instrument data from a byte slice. The first byte should be the base note value
    /// and the slice length should be 20.
    ///
    /// This method doesn't check that the returned values are in a valid range.
    pub fn from_bytes(slice: &[u8]) -> AifcResult<Instrument> {
        if slice.len() < 20 {
            return Err(AifcError::StdIoError(crate::unexpectedeof()));
        }
        Ok(Instrument {
            base_note: cast::u8_to_i8(slice[0]),
            detune: cast::u8_to_i8(slice[1]),
            low_note: cast::u8_to_i8(slice[2]),
            high_note: cast::u8_to_i8(slice[3]),
            low_velocity: cast::u8_to_i8(slice[4]),
            high_velocity: cast::u8_to_i8(slice[5]),
            gain: cast::u16_to_i16(u16::from(slice[6]) << 8 | u16::from(slice[7])),
            sustain_loop: Loop {
                play_mode: cast::u16_to_i16(u16::from(slice[8]) << 8 | u16::from(slice[9])),
                begin_loop: cast::u16_to_i16(u16::from(slice[10]) << 8 | u16::from(slice[11])),
                end_loop: cast::u16_to_i16(u16::from(slice[12]) << 8 | u16::from(slice[13])),
            },
            release_loop: Loop {
                play_mode: cast::u16_to_i16(u16::from(slice[14]) << 8 | u16::from(slice[15])),
                begin_loop: cast::u16_to_i16(u16::from(slice[16]) << 8 | u16::from(slice[17])),
                end_loop: cast::u16_to_i16(u16::from(slice[18]) << 8 | u16::from(slice[19])),
            }
        })
    }

    fn write_loop(write: &mut dyn Write, iloop: &Loop) -> AifcResult<()> {
        write.write_all(&iloop.play_mode.to_be_bytes())?;
        write.write_all(&iloop.begin_loop.to_be_bytes())?;
        write.write_all(&iloop.end_loop.to_be_bytes())?;
        Ok(())
    }

    /// Writes instrument data to the given write stream. The first byte written
    /// is the base note value. This method writes always 20 bytes.
    ///
    /// This method doesn't check that the instrument values are in a valid range.
    pub fn write_chunk_data(&self, write: &mut dyn Write) -> AifcResult<()> {
        write.write_all(&[
            cast::i8_to_u8(self.base_note),
            cast::i8_to_u8(self.detune),
            cast::i8_to_u8(self.low_note),
            cast::i8_to_u8(self.high_note),
            cast::i8_to_u8(self.low_velocity),
            cast::i8_to_u8(self.high_velocity),
        ])?;
        write.write_all(&self.gain.to_be_bytes())?;
        Instrument::write_loop(write, &self.sustain_loop)?;
        Instrument::write_loop(write, &self.release_loop)?;
        Ok(())
    }

    /// Copies instrument data to the given byte array. The first byte will contain
    /// the base note value.
    ///
    /// This method doesn't check that the instrument values are in a valid range.
    pub fn copy_to_slice(&self, slice: &mut [u8; 20]) {
        slice[0] = cast::i8_to_u8(self.base_note);
        slice[1] = cast::i8_to_u8(self.detune);
        slice[2] = cast::i8_to_u8(self.low_note);
        slice[3] = cast::i8_to_u8(self.high_note);
        slice[4] = cast::i8_to_u8(self.low_velocity);
        slice[5] = cast::i8_to_u8(self.high_velocity);
        let buf = self.gain.to_be_bytes();
        slice[6] = buf[0];
        slice[7] = buf[1];
        let buf = self.sustain_loop.play_mode.to_be_bytes();
        slice[8] = buf[0];
        slice[9] = buf[1];
        let buf = self.sustain_loop.begin_loop.to_be_bytes();
        slice[10] = buf[0];
        slice[11] = buf[1];
        let buf = self.sustain_loop.end_loop.to_be_bytes();
        slice[12] = buf[0];
        slice[13] = buf[1];
        let buf = self.release_loop.play_mode.to_be_bytes();
        slice[14] = buf[0];
        slice[15] = buf[1];
        let buf = self.release_loop.begin_loop.to_be_bytes();
        slice[16] = buf[0];
        slice[17] = buf[1];
        let buf = self.release_loop.end_loop.to_be_bytes();
        slice[18] = buf[0];
        slice[19] = buf[1];
    }
}

/// SizeCalculator counts the bytes written to it. It doesn't actually write anything.
struct SizeCalculator {
    size: u32
}

impl SizeCalculator {
    pub fn new() -> SizeCalculator {
        SizeCalculator { size: 0 }
    }
}

impl Write for SizeCalculator {
    fn write(&mut self, buf: &[u8]) -> Result<usize, std::io::Error> {
        let Ok(buflen_u32) = u32::try_from(buf.len()) else {
            return Err(std::io::Error::from(std::io::ErrorKind::Other));
        };
        self.size = match self.size.checked_add(buflen_u32) {
            Some(s) => s,
            None => {
                return Err(std::io::Error::from(std::io::ErrorKind::Other));
            }
        };
        Ok(buf.len())
    }
    fn flush(&mut self) -> Result<(), std::io::Error> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_instrument_from_bytes() -> AifcResult<()> {
        assert_eq!(Instrument::from_bytes(&[
            50, 250, 20, 120, 40, 80, 1, 5, 0, 1, 0, 2, 0, 3, 0, 0, 0, 7, 0, 9
        ])?, Instrument {
            base_note: 50,
            detune: -6,
            low_note: 20,
            high_note: 120,
            low_velocity: 40,
            high_velocity: 80,
            gain: 261,
            sustain_loop: Loop { play_mode: 1, begin_loop: 2, end_loop: 3 },
            release_loop: Loop { play_mode: 0, begin_loop: 7, end_loop: 9 },
        });
        Ok(())
    }

    #[test]
    fn test_instrument_copy_to_slice() {
        let mut bytes = [0; 20];
        Instrument {
            base_note: 53,
            detune: -7,
            low_note: 21,
            high_note: 121,
            low_velocity: 10,
            high_velocity: 70,
            gain: 258,
            sustain_loop: Loop { play_mode: 0, begin_loop: 3, end_loop: 9 },
            release_loop: Loop { play_mode: 1, begin_loop: 2, end_loop: 3 },
        }.copy_to_slice(&mut bytes);
        assert_eq!(bytes, [ 53, 249, 21, 121, 10, 70, 1, 2, 0, 0, 0, 3, 0, 9, 0, 1, 0, 2, 0, 3 ]);
    }
}
