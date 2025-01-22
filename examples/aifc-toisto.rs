/*!

Test command to read AIFF files and output json for Toisto AIFF test suite.
The "single" mode tests the single-pass reader and the "multi" mode tests the multi-pass reader.

Example run:

    cargo run --example aifc-toisto multi filename.aiff

*/
use std::fs::File;
use std::io::BufReader;
use std::env;

#[path = "../tests/shared/jsonhelper.rs"]
mod jsonhelper;

fn main() {
    let num_args = env::args().count();
    if num_args != 3 {
        println!("Usage: aifc-toisto {{single|multi}} <path-to-aifc-file>");
        return;
    }

    let reader_type = env::args().nth(1).expect("Cannot get reader type");

    // open file for reading

    let filename = env::args().nth(2)
        .expect("Cannot get filename");
    let mut rd = BufReader::new(File::open(filename)
        .expect("Failed to open file for reading"));
    let mut reader = match reader_type.as_ref() {
        "single" => match aifc::AifcReader::new_single_pass(&mut rd) {
            Ok(r) => r,
            Err(e) => { print_error(e); std::process::exit(0); }
        },
        "multi" => match aifc::AifcReader::new(&mut rd) {
            Ok(r) => r,
            Err(e) => { print_error(e); std::process::exit(0); }
        },
        _ => { eprintln!("Unknown reader type: {}", reader_type); std::process::exit(1); }
    };

    // print json

    let json = jsonhelper::jsonify(&mut reader).expect("Failed to jsonify");
    println!("{}", json);
}

fn print_error(e: aifc::AifcError) {
    match e {
        aifc::AifcError::StdIoError(e) => {
            println!("{{ \"error\": {:?} }}", e.to_string());
        },
        _ => {
            let msg = format!("{:?}", e);
            println!("{{ \"error\": \"{}\" }}", msg.replace("\"", "'"));
        },
    };
}
