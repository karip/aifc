//!
//! Test to write AIFF files based on input json spec file.
//!
use std::ffi::OsStr;
use std::fs::File;
use std::io::{Cursor, Error, ErrorKind, Read, Write};
use std::path::Path;
use serde::Deserialize;
use aifc::{AifcError, SampleFormat};

#[test]
fn toisto_writer() {
    // print a warning if the toisto-aiff-test-suite folder isn't found
    match Path::new("toisto-aiff-test-suite/tests").try_exists() {
        Ok(true) => {},
        _ => {
            // write directly to stderr() so that the warning is not captured by `cargo test`
            std::io::stderr()
                .write_all(b" * WARNING: Can't read folder 'toisto-aiff-test-suite/tests'\n")
                .expect("Can't read folder 'toisto-aiff-test-suite/tests'");
            std::process::exit(0);
        }
    }

    let mut json_filenames = Vec::new();
    glob_json_files("toisto-aiff-test-suite/tests", &mut json_filenames)
        .expect("Can't get json filenames");
    json_filenames.sort();
    run_test_for_files(&json_filenames, true, false);
}

#[path = "shared/jsonhelper.rs"]
mod jsonhelper;

fn ignored_tests() -> Vec<&'static str> {
    vec![
        "toisto-aiff-test-suite/tests/compressed/compressed-alaw-uppercase.json",
        "toisto-aiff-test-suite/tests/compressed/compressed-dwvw-16bit.json",
        "toisto-aiff-test-suite/tests/compressed/compressed-dwvw-24bit.json",
        "toisto-aiff-test-suite/tests/compressed/compressed-g722-ch1.json",
        "toisto-aiff-test-suite/tests/compressed/compressed-g722-ch2.json",
        "toisto-aiff-test-suite/tests/compressed/compressed-g722-ch3.json",
        "toisto-aiff-test-suite/tests/compressed/compressed-gsm.json",
        "toisto-aiff-test-suite/tests/compressed/compressed-mac3-ch1.json",
        "toisto-aiff-test-suite/tests/compressed/compressed-mac3-ch2.json",
        "toisto-aiff-test-suite/tests/compressed/compressed-mac6-ch1.json",
        "toisto-aiff-test-suite/tests/compressed/compressed-mac6-ch2.json",
        "toisto-aiff-test-suite/tests/compressed/compressed-qclp.json",
        "toisto-aiff-test-suite/tests/compressed/compressed-qdm2-ch1.json",
        "toisto-aiff-test-suite/tests/compressed/compressed-qdm2-ch2.json",
        "toisto-aiff-test-suite/tests/compressed/compressed-qdmc-ch1.json",
        "toisto-aiff-test-suite/tests/compressed/compressed-qdmc-ch2.json",
        "toisto-aiff-test-suite/tests/compressed/compressed-ulaw-uppercase.json",
        "toisto-aiff-test-suite/tests/exported/audacity-ima-adpcm.json",
        "toisto-aiff-test-suite/tests/exported/ffmpeg-id3.json",
        "toisto-aiff-test-suite/tests/exported/ffmpeg-metadata.json",
        "toisto-aiff-test-suite/tests/invalid/invalid-aifc-no-comm.json",
        "toisto-aiff-test-suite/tests/invalid/invalid-aiff-no-comm.json",
        "toisto-aiff-test-suite/tests/invalid/invalid-chunk-comm-short.json",
        "toisto-aiff-test-suite/tests/invalid/invalid-compression-type.json",
        "toisto-aiff-test-suite/tests/invalid/invalid-channels-0.json",
        "toisto-aiff-test-suite/tests/invalid/invalid-samplerate-0.json",
        "toisto-aiff-test-suite/tests/invalid/invalid-samplerate-inf.json",
        "toisto-aiff-test-suite/tests/invalid/invalid-samplerate-nan.json",
        "toisto-aiff-test-suite/tests/invalid/invalid-samplesize-0.json",
        "toisto-aiff-test-suite/tests/invalid/invalid-samplesize-33.json",
        "toisto-aiff-test-suite/tests/invalid/unspecified-chunk-anno-non-ascii.json",
        "toisto-aiff-test-suite/tests/invalid/unspecified-chunk-auth-non-ascii.json",
        "toisto-aiff-test-suite/tests/invalid/unspecified-chunk-comments-non-ascii.json",
        "toisto-aiff-test-suite/tests/invalid/unspecified-chunk-markers-non-ascii.json",
        "toisto-aiff-test-suite/tests/invalid/unspecified-chunk-name-non-ascii.json",
    ]
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
struct JsonMarker {
    id: i16,
    position: f32,
    name: String,
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
struct JsonComment {
    time_stamp: u32,
    marker: i16,
    text: String,
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
struct JsonInstrument {
    base_note: i8,
    detune: i8,
    low_note: i8,
    high_note: i8,
    low_velocity: i8,
    high_velocity: i8,
    gain: i16,
    sustain_loop: JsonLoop,
    release_loop: JsonLoop
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct JsonLoop {
    pub play_mode: i16,
    pub begin_loop: i16,
    pub end_loop: i16,
}

#[derive(Debug, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct JsonChunks {
    markers: Option<Vec<JsonMarker>>,
    comments: Option<Vec<JsonComment>>,
    inst: Option<JsonInstrument>,
    midi: Option<Vec<Vec<u8>>>,
    aesd: Option<Vec<u8>>,
    appl: Option<Vec<Vec<u8>>>,
    name: Option<String>,
    auth: Option<String>,
    copy: Option<String>,
    anno: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AiffJson {
    result: Option<String>,
    format: String,
    sample_rate: serde_json::Value,
    channels: i16,
    codec: String,
    sample_size: u32,
    samples_per_channel: usize,
    start_samples: Vec<Vec<serde_json::Value>>,
    end_samples: Vec<Vec<serde_json::Value>>,
    chunks: Option<JsonChunks>
}

impl AiffJson {
    pub fn sample_rate(&self) -> f64 {
        serde_value_to_f64(&self.sample_rate)
    }
}

fn run_test_for_files(json_filenames: &[String], verbose: bool, no_errors: bool) {
    let ignored = ignored_tests();
    let mut count_ok = 0;
    let mut count_fail = 0;
    let mut count_ignore = 0;
    for json_filename in json_filenames {
        let json_filename_dashed = json_filename.replace('\\', "/"); // for Windows
        if ignored.contains(&json_filename_dashed.as_ref()) {
            count_ignore += 1;
            if verbose {
                println!("IGNORE: {}", json_filename);
            }
            continue;
        }
        let src_json = parse_json_file(&json_filename);
        if src_json.result == Some("ignore".to_string()) ||
            src_json.result == Some("invalid".to_string()) {
            count_ignore += 1;
            if verbose {
                println!("IGNORE: {}", json_filename);
            }
            continue;
        }
        match test(&src_json) {
            Ok(()) => {
                count_ok += 1;
                if verbose {
                    println!("OK  : {}", json_filename);
                }
            },
            Err(e) => {
                count_fail += 1;
                if !no_errors {
                    println!("FAIL: {}", json_filename);
                    eprintln!(" * ERROR: {:?}", e);
                } else {
                    println!("(FAIL): {}", json_filename);
                    eprintln!(" * WARNING: {:?}", e);
                }
            }
        }
    }
    println!("Total write tests {}: {count_ok} passed, \
              {count_fail} failed, {count_ignore} ignored.",
        count_ok + count_fail + count_ignore);
    if count_fail > 0 {
        panic!("{} tests failed!", count_fail);
    }
}

fn test(src_json: &AiffJson) -> aifc::AifcResult<()> {
    let src_start_samples = convert_json_samples_to_f64(&src_json.start_samples);
    let src_end_samples = convert_json_samples_to_f64(&src_json.end_samples);

    let buf = match write_aifc(&src_json, &src_start_samples, &src_end_samples) {
        Err(e) => {
            return other_error(format!("Can't create AIFF file: {:?}", e));
        },
        Ok(b) => { b }
    };

    // open for reading

    let mut cursor = std::io::Cursor::new(&buf);
    let mut reader = aifc::AifcReader::new(&mut cursor)?;

    // create json

    let json = jsonhelper::jsonify(&mut reader).expect("Failed to jsonify");

    // compare source and generated jsons

    let gen_json: AiffJson = parse_json(&json, "generated-file");
    if src_json.format != gen_json.format {
        return other_error(format!("format mismatch {:?} != {:?}",
            src_json.format, gen_json.format));
    }
    if src_json.sample_rate() != gen_json.sample_rate() {
        return other_error(format!("sample_rate mismatch {:?} != {:?}",
            src_json.sample_rate(), gen_json.sample_rate()));
    }
    if src_json.channels != gen_json.channels {
        return other_error(format!("channels mismatch {:?} != {:?}",
            src_json.channels, gen_json.channels));
    }
    if src_json.codec != gen_json.codec {
        return other_error(format!("codec mismatch {:?} != {:?}", src_json.codec, gen_json.codec));
    }
    if src_json.sample_size != gen_json.sample_size {
        if (src_json.sample_size as f64 / 8.0).ceil() !=
            (gen_json.sample_size as f64 / 8.0).ceil() {
            return other_error(format!("sample_size mismatch {:?} != {:?}",
                src_json.sample_size, gen_json.sample_size));
        }
    }
    if src_json.samples_per_channel != gen_json.samples_per_channel {
        return other_error(format!("samples_per_channel mismatch {:?} != {:?}",
            src_json.samples_per_channel, gen_json.samples_per_channel));
    }
    if let Some(src_chunks) = &src_json.chunks {
        let gen_chunks = match gen_json.chunks {
            Some(chs) => chs,
            None => JsonChunks::default()
        };
        if src_chunks.markers != gen_chunks.markers {
            return other_error(format!("markers mismatch {:?} != {:?}",
                src_chunks.markers, gen_chunks.markers));
        }
        if src_chunks.comments != gen_chunks.comments {
            return other_error(format!("comments mismatch {:?} != {:?}",
                src_chunks.comments, gen_chunks.comments));
        }
        if src_chunks.inst != gen_chunks.inst {
            return other_error(format!("inst mismatch {:?} != {:?}", src_chunks.inst, gen_chunks.inst));
        }
        if src_chunks.midi != gen_chunks.midi {
            return other_error(format!("midi mismatch {:?} != {:?}", src_chunks.midi, gen_chunks.midi));
        }
        if src_chunks.aesd != gen_chunks.aesd {
            return other_error(format!("aesd mismatch {:?} != {:?}", src_chunks.aesd, gen_chunks.aesd));
        }
        if src_chunks.appl != gen_chunks.appl {
            return other_error(format!("appl mismatch {:?} != {:?}", src_chunks.appl, gen_chunks.appl));
        }
        if src_chunks.name != gen_chunks.name {
            return other_error(format!("name mismatch {:?} != {:?}", src_chunks.name, gen_chunks.name));
        }
        if src_chunks.auth != gen_chunks.auth {
            return other_error(format!("auth mismatch {:?} != {:?}", src_chunks.auth, gen_chunks.auth));
        }
        if src_chunks.copy != gen_chunks.copy {
            return other_error(format!("copy mismatch {:?} != {:?}", src_chunks.copy, gen_chunks.copy));
        }
        if src_chunks.anno != gen_chunks.anno {
            return other_error(format!("anno mismatch {:?} != {:?}", src_chunks.anno, gen_chunks.anno));
        }
    }
    let gen_start_samples = convert_json_samples_to_f64(&src_json.start_samples);
    let gen_end_samples = convert_json_samples_to_f64(&src_json.end_samples);

    for ch in 0..src_start_samples.len() {
        for i in 0..src_start_samples[ch].len() {
            if !is_f64_equal(src_start_samples[ch][i], gen_start_samples[ch][i]) {
                return other_error(format!("start_samples mismatch {i}:{ch} {:?} != {:?}",
                    src_start_samples[ch][i], gen_start_samples[ch][i]));
            }
        }
    }
    for ch in 0..src_end_samples.len() {
        for i in 0..src_end_samples[ch].len() {
            if !is_f64_equal(src_end_samples[ch][i], gen_end_samples[ch][i]) {
                return other_error(format!("end_samples mismatch {i}:{ch} {:?} != {:?}",
                    src_end_samples[ch][i], gen_end_samples[ch][i]));
            }
        }
    }
    // additional test for desc.num_sample_frames
    let mut cursor = std::io::Cursor::new(&buf);
    let mut reader = aifc::AifcReader::new(&mut cursor)?;
    let desc = match reader.read_info() {
        Err(error) => {
            return other_error(format!("Can't read the created AIFF file: {:?}", error));
        },
        Ok(val) => val
    };
    // source num_sample_frames is either samples_per_channel or packet count per channel for ima4
    let src_num_sample_frames = if desc.sample_format == SampleFormat::CompressedIma4 {
        src_json.samples_per_channel.div_ceil(64)
    } else {
        src_json.samples_per_channel
    };
    if desc.comm_num_sample_frames as usize != src_num_sample_frames {
        return other_error(format!("num_sample_frames mismatch {:?} != {:?}",
            desc.comm_num_sample_frames, src_num_sample_frames));
    }
    Ok(())
}

fn other_error(s: String) -> aifc::AifcResult<()> {
    return Err(AifcError::from(Error::new(ErrorKind::Other, s)));
}

fn is_f64_equal(val1: f64, val2: f64) -> bool {
    (val1.is_nan() && val2.is_nan()) || (val1 == val2)
}

fn parse_json_file(filename: &str) -> AiffJson {
    let mut file = File::open(filename).expect("Can't open spec json file");
    let mut txt = String::new();
    file.read_to_string(&mut txt).expect("Can't read spec json file");
    parse_json(&txt, filename)
}

fn parse_json(spectxt: &str, filename: &str) -> AiffJson {
    match serde_json::from_str(spectxt) {
        Ok(d) => d,
        Err(e) => {
            panic!(" * ERROR: invalid json file: {}: {}", filename, e);
        }
    }
}

fn convert_json_samples_to_f64(samples: &Vec<Vec<serde_json::Value>>) -> Vec<Vec<f64>> {
    let mut result = vec![];
    for ch in samples {
        let mut out = vec![];
        for s in ch {
            out.push(serde_value_to_f64(s));
        }
        result.push(out);
    }
    result
}

fn serde_value_to_f64(val: &serde_json::Value) -> f64 {
    match val {
        serde_json::Value::String(s) => {
            if s == "nan" { f64::NAN }
            else if s == "inf" { f64::INFINITY }
            else if s == "-inf" { -f64::INFINITY }
            else { panic!(" * ERROR: invalid value: {}", s); }
        },
        serde_json::Value::Number(n) => { n.as_f64().unwrap_or(0.0) },
        _ => { panic!(" * ERROR: invalid value: {:?}", val); }
    }
}

fn write_aifc(json: &AiffJson, src_start_samples: &Vec<Vec<f64>>, src_end_samples: &Vec<Vec<f64>>)
    -> aifc::AifcResult<Vec<u8>> {

    let file_format = if json.format == "aiff" {
        aifc::FileFormat::Aiff
    } else {
        aifc::FileFormat::Aifc
    };
    let sample_format = match (json.codec.as_str(), json.sample_size) {
        ("pcm_beu", 8) => SampleFormat::U8,
        ("pcm_bei", 1..=8) => SampleFormat::I8,
        ("pcm_bei", 9..=16) => SampleFormat::I16,
        ("pcm_lei", 9..=16) => SampleFormat::I16LE,
        ("pcm_bei", 17..=24) => SampleFormat::I24,
        ("pcm_bei", 25..=32) => SampleFormat::I32,
        ("pcm_lei", 25..=32) => SampleFormat::I32LE,
        ("pcm_bef", 32) => SampleFormat::F32,
        ("pcm_bef", 64) => SampleFormat::F64,
        ("ulaw", _) => SampleFormat::CompressedUlaw,
        ("alaw", _) => SampleFormat::CompressedAlaw,
        ("ima4", _) => SampleFormat::CompressedIma4,
        _ => {
            return Err(AifcError::from(Error::new(ErrorKind::Other,
                format!("Unsupported sample format {:?}, sample size {:?}",
                json.codec.as_str(), json.sample_size))));
        }
    };
    let winfo = aifc::AifcWriteInfo {
        file_format,
        sample_rate: json.sample_rate(),
        sample_format,
        channels: json.channels,
        ..aifc::AifcWriteInfo::default()
    };

    let mut markers: Option<Vec<aifc::Marker>> = None;
    let mut comments: Option<Vec<aifc::Comment>> = None;
    let mut chunk_data: Vec<(aifc::ChunkId, &[u8])> = vec![];
    let mut instrument_buf = [0u8; 20];
    if let Some(json_chunks) = &json.chunks {
        if let Some(jmarkers) = &json_chunks.markers {
            let ms: Vec<aifc::Marker> = jmarkers.iter().map(|m| {
                aifc::Marker {
                    id: m.id,
                    position: m.position as u32,
                    name: m.name.as_bytes()
                }
            }).collect();
            markers = Some(ms);
        };
        if let Some(jcomments) = &json_chunks.comments {
            let cs: Vec<aifc::Comment> = jcomments.iter().map(|c| {
                aifc::Comment {
                    timestamp: c.time_stamp,
                    marker_id: c.marker,
                    text: c.text.as_bytes()
                }
            }).collect();
            comments = Some(cs);
        }
        if let Some(i) = &json_chunks.inst {
            let iii = aifc::Instrument {
                base_note: i.base_note,
                detune: i.detune,
                low_note: i.low_note,
                high_note: i.high_note,
                low_velocity: i.low_velocity,
                high_velocity: i.high_velocity,
                gain: i.gain,
                sustain_loop: aifc::Loop {
                    play_mode: i.sustain_loop.play_mode,
                    begin_loop: i.sustain_loop.begin_loop,
                    end_loop: i.sustain_loop.end_loop
                },
                release_loop: aifc::Loop {
                    play_mode: i.release_loop.play_mode,
                    begin_loop: i.release_loop.begin_loop,
                    end_loop: i.release_loop.end_loop
                }
            };
            iii.copy_to_slice(&mut instrument_buf);
            chunk_data.push((aifc::CHUNKID_INST, &instrument_buf));
        }
        if let Some(midis) = &json_chunks.midi {
            for midi in midis {
                chunk_data.push((aifc::CHUNKID_MIDI, midi.as_slice()));
            }
        }
        if let Some(aesd) = &json_chunks.aesd {
            chunk_data.push((aifc::CHUNKID_AESD, aesd.as_slice()));
        }
        if let Some(appls) = &json_chunks.appl {
            for appl in appls {
                chunk_data.push((aifc::CHUNKID_APPL, appl.as_slice()));
            }
        }
        if let Some(name) = &json_chunks.name {
            let bytes = name.as_bytes();
            chunk_data.push((aifc::CHUNKID_NAME, bytes));
        }
        if let Some(auth) = &json_chunks.auth {
            let bytes = auth.as_bytes();
            chunk_data.push((aifc::CHUNKID_AUTH, bytes));
        }
        if let Some(copy) = &json_chunks.copy {
            let bytes = copy.as_bytes();
            chunk_data.push((aifc::CHUNKID_COPY, bytes));
        }
        if let Some(annos) = &json_chunks.anno {
            for anno in annos {
                let bytes = anno.as_bytes();
                chunk_data.push((aifc::CHUNKID_ANNO, bytes));
            }
        }
    }

    let mut buf = vec![];
    let cursor = Cursor::new(&mut buf);
    let mut aifc_writer = aifc::AifcWriter::new(cursor, &winfo)?;
    let Ok(channels) = usize::try_from(json.channels) else {
        return Err(AifcError::from(Error::new(ErrorKind::Other, "Too many channels")));
    };
    if let Some(ms) = markers {
        aifc_writer.write_chunk_markers(&ms)?;
    }
    if let Some(cs) = comments {
        aifc_writer.write_chunk_comments(&cs)?;
    }
    for chdata in chunk_data {
        aifc_writer.write_chunk(&chdata.0, chdata.1)?;
    }
    match sample_format {
        SampleFormat::U8 => {
            let samples = create_samples::<u8>(json.samples_per_channel,
                channels, src_start_samples, src_end_samples,
                |x| { x as u8 });
            aifc_writer.write_samples_u8(&samples)?;
        },
        SampleFormat::I8 => {
            let samples = create_samples::<i8>(json.samples_per_channel,
                channels, src_start_samples, src_end_samples,
                |x| { x as i8 });
            aifc_writer.write_samples_i8(&samples)?;
        },
        SampleFormat::I16 | SampleFormat::I16LE => {
            let samples = create_samples::<i16>(json.samples_per_channel,
                channels, src_start_samples, src_end_samples,
                |x| { x as i16 });
            aifc_writer.write_samples_i16(&samples)?;
        },
        SampleFormat::I24 => {
            let samples = create_samples::<i32>(json.samples_per_channel,
                channels, src_start_samples, src_end_samples,
                |x| { x as i32 });
            aifc_writer.write_samples_i24(&samples)?;
        },
        SampleFormat::I32 | SampleFormat::I32LE => {
            let samples = create_samples::<i32>(json.samples_per_channel,
                channels, src_start_samples, src_end_samples,
                |x| { x as i32 });
            aifc_writer.write_samples_i32(&samples)?;
        },
        SampleFormat::F32 => {
            let samples = create_samples::<f32>(json.samples_per_channel,
                channels, src_start_samples, src_end_samples,
                |x| { x as f32 });
            aifc_writer.write_samples_f32(&samples)?;
        },
        SampleFormat::F64 => {
            let samples = create_samples::<f64>(json.samples_per_channel,
                channels, src_start_samples, src_end_samples,
                |x| { x as f64 });
            aifc_writer.write_samples_f64(&samples)?;
        },
        SampleFormat::CompressedUlaw | SampleFormat::CompressedAlaw |
        SampleFormat::CompressedIma4 => {
            let samples = create_samples::<i16>(json.samples_per_channel,
                channels, src_start_samples, src_end_samples,
                |x| { x as i16 });
            aifc_writer.write_samples_i16(&samples)?;
        },
        SampleFormat::Custom(_) => {
            let sample_count = json.samples_per_channel as u32 * json.channels as u32;
            let samples = vec![0u8; sample_count as usize];
            aifc_writer.write_samples_raw(&samples, sample_count)?;
        },
    }
    aifc_writer.finalize()?;
    Ok(buf)
}

fn create_samples<T>(samples_per_channel: usize, channels: usize, start_samples: &Vec<Vec<f64>>,
    end_samples: &Vec<Vec<f64>>, converter: impl Fn(f64) -> T) -> Vec<T> where T: Default+Clone {
    let mut samples = vec![T::default(); samples_per_channel*channels];
    if samples_per_channel == 0 || channels == 0 {
        return samples;
    }
    for ch in 0..channels {
        for i in 0..start_samples[ch].len() {
            samples[i*channels + ch] = converter(start_samples[ch][i]);
        }
        for i in 0..end_samples[ch].len() {
            let pos = (samples_per_channel - end_samples[ch].len() + i)*channels + ch;
            samples[pos] = converter(end_samples[ch][i]);
        }
    }
    samples
}

fn glob_json_files(folder: impl AsRef<Path>, jsons: &mut Vec<String>) -> std::io::Result<()> {
    for entry in std::fs::read_dir(folder)? {
        let entry = entry?;
        if entry.path().is_dir() {
            glob_json_files(entry.path(), jsons)?;

        } else if entry.path().extension() == Some(OsStr::new("json")) {
            jsons.push(entry.path().to_string_lossy().to_string());
        }
    }
    Ok(())
}
