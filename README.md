# AIFC

[![Cross-platform tests](https://github.com/karip/aifc/actions/workflows/cross-test.yml/badge.svg)](https://github.com/karip/aifc/actions/workflows/cross-test.yml)

Rust library to read and write AIFF and AIFF-C (AIFC) audio format.

Features:

 - reading AIFF and AIFF-C
 - writing AIFF and AIFF-C
 - no heap memory allocations
 - no unsafe code
 - no panicking
 - supports uncompressed integer and floating point samples
 - supports compression types: μ-law, A-law and IMA ADPCM ("ima4")
 - supports audio files up to 4 gigabytes

Out of scope:

 - conversion between different sample formats (e.g., i16 to f32). There's
   [so many ways](https://web.archive.org/web/20240224192658/http://blog.bjornroche.com/2009/12/int-float-int-its-jungle-out-there.html)
   to do the conversion that it's better that this crate doesn't do it.

## Usage

Reading AIFF/AIFF-C file:

```rust, no_run
let mut stream = std::io::BufReader::new(std::fs::File::open("test.aiff").expect("Open failed"));
let mut reader = aifc::AifcReader::new(&mut stream).expect("Can't create reader");
let info = reader.read_info().expect("Can't read header");
for sample in reader.samples().expect("Can't iterate samples") {
    println!("Got sample {:?}", sample.expect("Sample read error"));
}
```

Writing AIFF-C file (with the default 2 channels, 16 bits/sample, sample rate 44100):

```rust, no_run
let mut stream = std::io::BufWriter::new(std::fs::File::create("test.aiff").expect("Open failed"));
let info = aifc::AifcWriteInfo::default();
let mut writer = aifc::AifcWriter::new(&mut stream, &info).expect("Can't create writer");
writer.write_samples_i16(&[ 1, 2, 3, 4 ]).expect("Can't write samples");
writer.finalize().expect("Can't finalize");
```

See [the AIFC API documentation](https://docs.rs/aifc/) for details.

## Examples

A simple AIFF player using `aifc::AifcReader` and [tinyaudio](https://crates.io/crates/tinyaudio):

```sh
cd examples/aifc-tinyaudio
cargo run filename.aiff
```

## Testing

[Toisto AIFF Test Suite](https://github.com/karip/toisto-aiff-test-suite) is a submodule and
needs to be fetched before running the tests.

```sh
cd aifc
git submodule update --init
./tools/test.sh
```

The test should end with `--- All tests OK.`.

Performance testing:

```sh
cargo bench
```

There is a GitHub Action called "Cross-platform tests" (cross-test.yml), which automatically
runs `./tools/test.sh` for little-endian 64-bit x64_86 and big-endian 32-bit PowerPC.

## References

 - [Wikipedia: AIFF file format](https://en.wikipedia.org/wiki/Audio_Interchange_File_Format)
 - [AIFF/AIFF-C Specifications](https://www-mmsp.ece.mcgill.ca/Documents/AudioFormats/AIFF/AIFF.html)

## License

Licensed under either of <a href="LICENSE-APACHE">Apache License, Version
2.0</a> or <a href="LICENSE-MIT">MIT license</a> at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
