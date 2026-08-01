mod app;
mod forward;
mod kernels;
mod model;
mod sample;

fn main() {
    std::process::exit(app::run());
}
