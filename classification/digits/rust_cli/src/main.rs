use clap::Parser;
use std::process::Command;
use std::path::PathBuf;
use serde_json::Value;

#[derive(Parser)]
struct Args{
	paths: Vec<PathBuf>
}

fn main(){
	let args = Args::parse();

	if args.paths.is_empty(){
		eprintln!("Provide at least one image or directory");
		return;
	}

	let mut images = Vec::new();

	for path in args.paths{
		if path.is_dir(){
			for entry in std::fs::read_dir(path).unwrap(){
				let p = entry.unwrap().path();
				if let Some(ext) = p.extension(){
					if ext == "png" || ext == "jpg" || ext == "jpeg"{
						images.push(p);
					}
				}
			}
		} else{
			images.push(path);
		}
	}

	let mut cmd = Command::new("python");
	cmd.arg("classification/digits/infer.py");

	for img in &images{
		cmd.arg(img);
	}

	let output = cmd.output().expect("failed to run python");

	let json: Value = serde_json::from_slice(&output.stdout).unwrap();
	
	for(k,v) in json.as_object().unwrap(){
		println!("{} -> {}", k, v);
	}
}
