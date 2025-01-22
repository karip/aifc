use criterion::{criterion_group, criterion_main, black_box, Criterion};
use std::io::Cursor;
use aifc::AifcReader;

const SAMPLE_FRAME_COUNT: u32 = 512;

/// Returns nth byte from the given u32 value.
fn bp(value: u32, byte_num: u32) -> u8 {
    ((value >> (byte_num * 8)) & 0xff) as u8
}

/// Creates AIFF-C with floating point samples.
fn create_aifc(frame_count: u32) -> Vec<u8> {
    let channels: u8 = 1;
    let bits_per_sample: u8 = 32;
    let bytes_per_sample = u32::from(bits_per_sample.div_ceil(8));
    let ssnd_size = frame_count * bytes_per_sample * u32::from(channels) + 8;
    let form_size = 56 + ssnd_size;
    let mut aifc = vec![
        b'F', b'O', b'R', b'M',
        bp(form_size, 3), bp(form_size, 2), bp(form_size, 1), bp(form_size, 0),
        b'A', b'I', b'F', b'C',
        b'F', b'V', b'E', b'R', 0, 0, 0, 4, 0xA2, 0x80, 0x51, 0x40,
        b'C', b'O', b'M', b'M', 0, 0, 0, 24,
        0, channels,                                                    // num_channels
        bp(frame_count, 3), bp(frame_count, 2), bp(frame_count, 1), bp(frame_count, 0),
        0, bits_per_sample,                                             // sample_size
        0x40, 0x0E, 0xAC, 0x44, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,     // sample_rate
        b'f', b'l', b'3', b'2',
        0, 0,
        b'S', b'S', b'N', b'D',
        bp(ssnd_size, 3), bp(ssnd_size, 2), bp(ssnd_size, 1), bp(ssnd_size, 0),
        0, 0, 0, 0,  0, 0, 0, 0,
    ];
    // samples
    for _ in 0..frame_count {
        aifc.extend_from_slice(&[ 0, 0, 0, 0 ]);
    }
    aifc
}

fn criterion_benchmark(c: &mut Criterion) {
    let aifc_data = create_aifc(SAMPLE_FRAME_COUNT);

    c.bench_function("samples_iterator", |b| b.iter(|| {
        let cursor = Cursor::new(&aifc_data);
        let mut reader = AifcReader::new(cursor).expect("reader failed");
        for s in reader.samples().expect("no iterator") {
            black_box(match s.expect("iter failed") {
                aifc::Sample::U8(_) => {},
                aifc::Sample::I8(_) => {},
                aifc::Sample::I16(_) => {},
                aifc::Sample::I24(_) => {},
                aifc::Sample::I32(_) => {},
                aifc::Sample::F32(_) => {},
                aifc::Sample::F64(_) => {},
            })
        }
    }));

    c.bench_function("read_sample", |b| b.iter(|| {
        let cursor = Cursor::new(&aifc_data);
        let mut reader = AifcReader::new(cursor).expect("reader failed");
        while let Some(s) = reader.read_sample().expect("sample error") {
            black_box(match s {
                aifc::Sample::U8(_) => {},
                aifc::Sample::I8(_) => {},
                aifc::Sample::I16(_) => {},
                aifc::Sample::I24(_) => {},
                aifc::Sample::I32(_) => {},
                aifc::Sample::F32(_) => {},
                aifc::Sample::F64(_) => {},
            })
        }
    }));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
