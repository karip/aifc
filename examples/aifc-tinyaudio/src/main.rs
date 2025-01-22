/*!

Plays AIFF audio files using aifc and tinyaudio.

Example run:

    cd examples/aifc-tinyaudio
    cargo run filename.aiff

*/
use std::env;
use std::fs::File;
use std::io::BufReader;
use tinyaudio::prelude::*;

fn main() {

    // open aiff / aifc file for reading
    let filename = env::args().nth(1)
        .expect("Filename missing");
    let bufreader = BufReader::new(File::open(filename)
        .expect("Can't open file"));
    let mut reader = aifc::AifcReader::new(bufreader)
        .expect("Can't create reader");
    let info = reader.info();

    if !info.sample_rate.is_finite() || info.sample_rate < 0.01 {
        eprintln!("Invalid sample rate: {}", info.sample_rate);
        return;
    }
    let total_samples = info.sample_len
        .expect("Unsupported sample format");
    let duration = total_samples as f64 / info.channels as f64 / info.sample_rate as f64;
    println!("Audio file: {} channels, sample rate {}, format {:?}, duration {:.3} secs.",
        info.channels, info.sample_rate, info.sample_format, duration);

    // audio buffer size = total sample len or max 2048 bytes
    let channel_sample_count = 2048.min(total_samples as usize);

    // play the samples using tinyaudio
    let _device = run_output_device(
        OutputDeviceParameters {
            channels_count: info.channels as usize,
            sample_rate: info.sample_rate as usize,
            channel_sample_count,
        },
        move |data| {
            // read samples from the audio file to fill the audio buffer
            // ideally, the samples should be buffered to avoid glitches in audio
            for sample in data.iter_mut() {
                *sample = match reader.read_sample().expect("Error reading samples") {
                    Some(s) => simple_sample_to_f32(s),
                    None => 0.0
                };
            }
        },
    ).expect("Audio output failed");

    // sleep until playback is done - this should be improved so that the program is exited
    // when all samples have been played instead of estimating the duration and latency
    let latency = channel_sample_count as f64 / info.sample_rate as f64 * 2.0;
    std::thread::sleep(std::time::Duration::from_millis(((duration+latency) * 1000.0) as u64));
}

// simple conversion algorithm to convert Sample to f32
fn simple_sample_to_f32(s: aifc::Sample) -> f32 {
    match s {
        aifc::Sample::U8(s) => { s as f32 / (1u32 << 7) as f32 - 1.0 },
        aifc::Sample::I8(s) => { s as f32 / (1u32 << 7) as f32 },
        aifc::Sample::I16(s) => { s as f32 / (1u32 << 15) as f32 },
        aifc::Sample::I24(s) => { s as f32 / (1u32 << 23) as f32 },
        aifc::Sample::I32(s) => { s as f32 / (1u32 << 31) as f32 },
        aifc::Sample::F32(s) => { s },
        aifc::Sample::F64(s) => { s as f32 },
    }
}
