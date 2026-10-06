#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    subhooper_lib::startup_trace("main-enter");
    subhooper_lib::run();
    subhooper_lib::startup_trace("main-return");
}
