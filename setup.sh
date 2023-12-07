## Install rustup and common components
curl https://sh.rustup.rs -sSf | sh -s -- -y 
rustup install stable
rustup default stable
rustup component add rustfmt
rustup component add clippy 


cargo install cargo-expand
cargo install cargo-edit
