// Helper for testing.

use aifc::{AifcError, SampleFormat, Marker, Comment};
use id3::TagLike;

/// Reads data from the given AifcReader and returns it as JSON.
pub fn jsonify<T>(reader: &mut aifc::AifcReader<T>) -> aifc::AifcResult<String>
    where T: std::io::Read + std::io::Seek {

    let mut json = String::new();

    let info = match reader.read_info() {
        Err(AifcError::StdIoError(e)) => {
            return Ok(format!("{{ \"error\": {:?} }}", e.to_string()));
        },
        Err(error) => {
            let msg = format!("{:?}", error);
            return Ok(format!("{{ \"error\": \"{}\" }}", msg.replace("\"", "'")));
        },
        Ok(val) => val
    };

    // print json

    json += &format!("{{\n");
    match info.file_format {
        aifc::FileFormat::Aiff => { json += &format!("    \"format\": \"aiff\",\n"); },
        aifc::FileFormat::Aifc => { json += &format!("    \"format\": \"aiff-c\",\n"); }
    }
    if info.sample_rate.is_finite() {
        json += &format!("    \"sampleRate\": {},\n", info.sample_rate);
    } else {
        json += &format!("    \"sampleRate\": \"{}\",\n",
            format!("{}", info.sample_rate).to_lowercase());
    }
    json += &format!("    \"channels\": {},\n", info.channels);
    json += &format!("    \"codec\": {},\n", sampleformat_to_codec(&info.sample_format));
    let sample_size = if info.comm_compression_type == *b"NONE" {
        info.comm_sample_size as usize
    } else {
        info.sample_format.decoded_size()*8
    };
    json += &format!("    \"sampleSize\": {},\n", sample_size);

    json += &format!("    \"chunks\": {{\n");

    if let Ok(Some(size)) = reader.read_chunk_markers(&mut []) {
        let mut data = vec![0; size];
        reader.read_chunk_markers(&mut data)?;
        let markers: Vec<aifc::AifcResult<aifc::Marker>> = aifc::Markers::new(&data)?
            .collect();
        if !markers.is_empty() {
            let empty_marker = Marker { id: -1, position: 0, name: &[ b'B', b'A', b'D' ] };
            let strlist = markers.iter()
                .map(|marker| {
                    let m = marker.as_ref().unwrap_or(&empty_marker);
                    format!("{{ \"id\": {}, \"position\": {}, \"name\": {} }}",
                        m.id,
                        m.position,
                        bytes_to_json_string(&m.name))
                })
                .collect::<Vec<String>>()
                .join(", ");
            json += &format!("        \"markers\": [ {} ],\n", strlist);
        }
    }

    if let Ok(Some(size)) = reader.read_chunk_comments(&mut []) {
        let mut data = vec![0; size];
        reader.read_chunk_comments(&mut data)?;
        let comments: Vec<aifc::AifcResult<aifc::Comment>> = aifc::Comments::new(&data)?
            .collect();
        if !comments.is_empty() {
            let empty_comment = Comment {
                timestamp: 0, marker_id: -1, text: &[ b'B', b'A', b'D' ]
            };
            let strlist = comments.iter()
                .map(|comment| {
                    let c = comment.as_ref().unwrap_or(&empty_comment);
                    format!("{{ \"timeStamp\": {}, \"marker\": {}, \"text\": {} }}",
                        c.timestamp,
                        c.marker_id,
                        bytes_to_json_string(&c.text))
                })
                .collect::<Vec<String>>()
                .join(", ");
            json += &format!("        \"comments\": [ {} ],\n", strlist);
        }
    }

    match reader.chunks() {
        Ok(mut chunks) => {
            let mut midis = vec![];
            let mut appls = vec![];
            let mut annos = vec![];
            while let Some(d) = chunks.next() {
                let d = d.expect("Can't read chunk");
                match &d.id {
                    &aifc::CHUNKID_INST => {
                        let mut buf = vec![0u8; usize::try_from(d.size)
                            .expect("Chunk too large")];
                        chunks.read_data(&d, &mut buf)?;
                        let inst = aifc::Instrument::from_bytes(&buf)?;
                        json += &format!("        \"inst\": {{\n");
                        json += &format!("            \"baseNote\": {},\n", inst.base_note);
                        json += &format!("            \"detune\": {},\n", inst.detune);
                        json += &format!("            \"lowNote\": {},\n", inst.low_note);
                        json += &format!("            \"highNote\": {},\n", inst.high_note);
                        json += &format!("            \"lowVelocity\": {},\n", inst.low_velocity);
                        json += &format!("            \"highVelocity\": {},\n", inst.high_velocity);
                        json += &format!("            \"gain\": {},\n", inst.gain);
                        json += &format!("            \"sustainLoop\": {},\n",
                            loop_to_str(&inst.sustain_loop));
                        json += &format!("            \"releaseLoop\": {}\n",
                            loop_to_str(&inst.release_loop));
                        json += &format!("        }},\n");
                    },
                    &aifc::CHUNKID_NAME | &aifc::CHUNKID_AUTH => {
                        let mut buf = vec![0u8; usize::try_from(d.size)
                            .expect("Chunk too large")];
                        chunks.read_data(&d, &mut buf)?;
                        json += &format!("        \"{}\": {},\n",
                            String::from_utf8_lossy(&d.id).to_ascii_lowercase(),
                            bytes_to_json_string(&buf));
                    },
                    &aifc::CHUNKID_MIDI => {
                        let mut buf = vec![0u8; usize::try_from(d.size)
                            .expect("Chunk too large")];
                        chunks.read_data(&d, &mut buf)?;
                        midis.push(buf);
                    },
                    &aifc::CHUNKID_APPL => {
                        let mut buf = vec![0u8; usize::try_from(d.size)
                            .expect("Chunk too large")];
                        chunks.read_data(&d, &mut buf)?;
                        appls.push(buf);
                    },
                    &aifc::CHUNKID_COPY => {
                        let mut buf = vec![0u8; usize::try_from(d.size)
                            .expect("Chunk too large")];
                        chunks.read_data(&d, &mut buf)?;
                        json += &format!("        \"(c)\": {},\n", bytes_to_json_string(&buf));
                    },
                    &aifc::CHUNKID_ANNO => {
                        let mut buf = vec![0u8; usize::try_from(d.size)
                            .expect("Chunk too large")];
                        chunks.read_data(&d, &mut buf)?;
                        annos.push(buf);
                    },
                    &aifc::CHUNKID_AESD | b"hash" => {
                        let mut buf = vec![0u8; usize::try_from(d.size)
                            .expect("Chunk too large")];
                        chunks.read_data(&d, &mut buf)?;
                        json += &format!("        \"{}\": {},\n",
                            String::from_utf8_lossy(&d.id).to_ascii_lowercase(),
                            bytes_to_json_number_array(&buf));
                    },
                    _=> {
                    }
                }
            }
            if !midis.is_empty() {
                let strlist = midis.iter()
                    .map(|data| format!("{}", bytes_to_json_number_array(&data)))
                    .collect::<Vec<String>>()
                    .join(", ");
                json += &format!("        \"midi\": [ {} ],\n", strlist);
            }
            if !appls.is_empty() {
                let strlist = appls.iter()
                    .map(|data| format!("{}", bytes_to_json_number_array(&data)))
                    .collect::<Vec<String>>()
                    .join(", ");
                json += &format!("        \"appl\": [ {} ],\n", strlist);
            }
            if !annos.is_empty() {
                let strlist = annos.iter()
                    .map(|data| bytes_to_json_string(&data))
                    .collect::<Vec<String>>()
                    .join(", ");
                json += &format!("        \"anno\": [ {} ],\n", strlist);
            }

            if let Ok(Some(size)) = reader.read_chunk_id3(&mut []) {
                let mut data = vec![0; size];
                reader.read_chunk_id3(&mut data)?;
                let tag = id3::Tag::read_from(std::io::Cursor::new(&data))
                    .expect("bad id3 data");
                json += &format!("        \"id3\": {{");
                let mut framelist = vec![];
                if let Some(title) = tag.title() {
                    framelist.push(format!("            \"ATT2\": \"{}\"", title));
                }
                if let Some(artist) = tag.artist() {
                    framelist.push(format!("            \"TP1\": \"{}\"", artist));
                }
                if let Some(album) = tag.album() {
                    framelist.push(format!("            \"TAL\": \"{}\"", album));
                }
                if let Some(track) = tag.track() {
                    if let Some(total) = tag.total_tracks() {
                        framelist.push(format!("            \"TRK\": \"{}/{}\"", track, total));
                    } else {
                        framelist.push(format!("            \"TRK\": \"{}\"", track));
                    }
                }
                if let Some(year) = tag.year() {
                    framelist.push(format!("            \"TYE\": \"{}\"", year));
                }
                if let Some(tco) = tag.genre_parsed() {
                    framelist.push(format!("            \"TCO\": \"{}\"", tco));
                }
                // print the first comment only, remove possible NUL chars
                for com in tag.comments() {
                    framelist.push(format!("            \"COM\": \"{}\"",
                        com.text.replace("\u{0}", "")));
                    break;
                }
                json += &format!("{}\n        }},\n", framelist.join(",\n"));
            }

        },
        Err(_) => {
            json += &format!("        \"markers\": \"-unsupported-\",\n");
            json += &format!("        \"comments\": \"-unsupported-\",\n");
            json += &format!("        \"inst\": \"-unsupported-\",\n");
            json += &format!("        \"midi\": \"-unsupported-\",\n");
            json += &format!("        \"aesd\": \"-unsupported-\",\n");
            json += &format!("        \"appl\": \"-unsupported-\",\n");
            json += &format!("        \"name\": \"-unsupported-\",\n");
            json += &format!("        \"auth\": \"-unsupported-\",\n");
            json += &format!("        \"(c)\": \"-unsupported-\",\n");
            json += &format!("        \"anno\": \"-unsupported-\",\n");
            json += &format!("        \"id3\": \"-unsupported-\",\n");
            json += &format!("        \"hash\": \"-unsupported-\",\n");
        }
    }
    json += &format!("        \"chan\": \"-unsupported-\"\n");

    json += &format!("    }},\n");

    let sample_frames = if let Some(slen) = info.sample_len {
        if info.channels > 0 {
            slen / info.channels as u64
        } else {
            info.comm_num_sample_frames as u64
        }
    } else {
        info.comm_num_sample_frames as u64
    };
    json += &format!("    \"samplesPerChannel\": {}", sample_frames);

    match read_samples(reader) {
        Ok(samples) => {
            // print sample data
            json += &format!(",\n    \"startSamples\": [\n");
            let start_einx = samples.len().min(300*info.channels as usize);
            print_sample_data(info.channels as usize, &samples[0..start_einx], &mut json);
            json += &format!("    ],\n");

            json += &format!("    \"endSamples\": [\n");
            let mut end_sinx = 0;
            if samples.len() > 30*info.channels as usize {
                end_sinx = samples.len() - 30*info.channels as usize;
            }
            print_sample_data(info.channels as usize, &samples[end_sinx..], &mut json);
            json += &format!("    ]\n");
        },
        Err(AifcError::Unsupported) => {
            json += &format!(",\n    \"startSamples\": \"-unsupported-\",\n");
            json += &format!("    \"endSamples\": \"-unsupported-\"\n");
        },
        Err(e) => {
            let msg = format!("{:?}", e).replace("\"", "'");
            json += &format!(",\n    \"error\": \"{}\"\n", msg);
        }
    }

    json += &format!("}}\n");
    Ok(json)
}

fn read_samples<T>(reader: &mut aifc::AifcReader<T>) -> Result<Vec<f64>, AifcError>
    where T: std::io::Read+std::io::Seek {
    let mut samples = vec![];
    while let Some(sample) = reader.read_sample()? {
        match sample {
            aifc::Sample::U8(s) => { samples.push(s as f64); },
            aifc::Sample::I8(s) => { samples.push(s as f64); },
            aifc::Sample::I16(s) => { samples.push(s as f64); },
            aifc::Sample::I24(s) => { samples.push(s as f64); },
            aifc::Sample::I32(s) => { samples.push(s as f64); },
            aifc::Sample::F32(s) => { samples.push(s as f64); },
            aifc::Sample::F64(s) => { samples.push(s as f64); }
        }
    }
    Ok(samples)
}

/// Converts bytes to JSON array: [ numbers.. ]
fn bytes_to_json_number_array(data: &[u8]) -> String {
    let list = data.iter()
        .map(|b| format!("{}", b))
        .collect::<Vec<String>>()
        .join(", ");
    format!("[ {} ]", list)
}

/// Converts bytes to JSON string.
fn bytes_to_json_string(data: &[u8]) -> String {
    let mut s = String::new();
    for b in data {
        if b.is_ascii() {
            s.push(char::from(*b));
        } else {
            s.push(char::REPLACEMENT_CHARACTER);
        }
    }
    serde_json::to_string(&s).expect("JSON encode error")
}

fn sampleformat_to_codec(sample_format: &SampleFormat) -> String {
    match sample_format {
        aifc::SampleFormat::U8 => { "\"pcm_beu\"".to_string() },
        aifc::SampleFormat::I8 => { "\"pcm_bei\"".to_string() },
        aifc::SampleFormat::I16 => { "\"pcm_bei\"".to_string() },
        aifc::SampleFormat::I24 => { "\"pcm_bei\"".to_string() },
        aifc::SampleFormat::I32 => { "\"pcm_bei\"".to_string() },
        aifc::SampleFormat::I16LE => { "\"pcm_lei\"".to_string() },
        aifc::SampleFormat::I32LE => { "\"pcm_lei\"".to_string() },
        aifc::SampleFormat::F32 => { "\"pcm_bef\"".to_string() },
        aifc::SampleFormat::F64 => { "\"pcm_bef\"".to_string() },
        aifc::SampleFormat::CompressedUlaw => { "\"ulaw\"".to_string() },
        aifc::SampleFormat::CompressedAlaw => { "\"alaw\"".to_string() },
        aifc::SampleFormat::CompressedIma4 => { "\"ima4\"".to_string() },
        aifc::SampleFormat::Custom(chid) => { bytes_to_json_string(chid) },
    }
}

fn loop_to_str(inst_loop: &aifc::Loop) -> String {
    format!("{{ \"playMode\": {:?}, \"beginLoop\": {}, \"endLoop\": {} }}",
        inst_loop.play_mode,
        inst_loop.begin_loop,
        inst_loop.end_loop
    )
}

fn print_sample_data(num_channels: usize, samples: &[f64], json: &mut String) {
    if num_channels == 0 {
        return;
    }
    let samples_per_channel = samples.len() / num_channels;
    for ch in 0..num_channels {
        *json += "        [ ";
        let mut pos = 0;
        while pos < samples_per_channel {
            if pos != 0 {
                *json += ", ";
            }
            let s = samples[pos * num_channels + ch];
            if s.is_finite() {
                *json += &format!("{:.6}", s);
            } else {
                let str = format!("\"{:.6}\"", s);
                *json += &format!("{}", str.to_lowercase());
            }
            pos += 1;
        }
        if ch < num_channels-1 {
            *json += " ],\n";
        } else {
            *json += " ]\n";
        }
    }
}
