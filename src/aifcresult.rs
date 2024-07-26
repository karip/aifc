use core::result;

/// Error values.
#[derive(Debug)]
pub enum AifcError {
    /// Unrecognized format.
    UnrecognizedFormat,
    /// The operation is not supported for the current compression type.
    Unsupported,
    /// The COMM header chunk does not exist or it contains invalid values.
    InvalidCommChunk,
    /// Invalid sample size.
    InvalidSampleSize,
    /// The COMM chunk was not found before the SSND chunk in the stream,
    /// which means that a single-pass reader can't read it.
    CommChunkNotFoundBeforeSsndChunk,
    /// Invalid number of channels.
    InvalidNumberOfChannels,
    /// Invalid read state.
    InvalidReadState,
    /// Invalid write state.
    InvalidWriteState,
    /// The size is too large.
    SizeTooLarge,
    /// Invalid parameter.
    InvalidParameter,
    /// Invalid sample format.
    InvalidSampleFormat,
    /// Timestamp is out of bounds.
    TimestampOutOfBounds,
    /// Read error.
    ReadError,
    /// Seek error.
    SeekError,
    /// Standard IO error.
    StdIoError(std::io::Error),
}

impl From<std::io::Error> for AifcError {
    fn from(e: std::io::Error) -> Self {
        AifcError::StdIoError(e)
    }
}

/// Library Result type.
pub type AifcResult<T> = result::Result<T, AifcError>;
