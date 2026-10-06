fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        match std::env::var("CARGO_CFG_TARGET_ENV").as_deref() {
            Ok("msvc") => println!("cargo:rustc-link-arg-bin=subhooper=/STACK:8388608"),
            Ok("gnu") => println!("cargo:rustc-link-arg-bin=subhooper=-Wl,--stack,8388608"),
            _ => {}
        }
    }
    tauri_build::build()
}
