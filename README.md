# softfloat-wrapper
softfloat-wrapper is a safe wrapper of [Berkeley SoftFloat](https://github.com/ucb-bar/berkeley-softfloat-3) based on [softfloat-sys](https://crates.io/crates/softfloat-sys).

[![Actions Status](https://github.com/dalance/softfloat-wrapper/workflows/Regression/badge.svg)](https://github.com/dalance/softfloat-wrapper/actions)
[![Crates.io](https://img.shields.io/crates/v/softfloat-wrapper.svg)](https://crates.io/crates/softfloat-wrapper)
[![Docs.rs](https://docs.rs/softfloat-wrapper/badge.svg)](https://docs.rs/softfloat-wrapper)

## Usage

```Cargo.toml
[dependencies]
softfloat-wrapper = "0.1.2"
```

## Example

```rust
use softfloat_wrapper::{Float, F16, RoundingMode};

fn main() {
    let a = 0x1234;
    let b = 0x1479;

    let a = F16::from_bits(a);
    let b = F16::from_bits(b);
    let d = a.add(b, RoundingMode::TiesToEven);

    let a = f32::from_bits(a.to_f32(RoundingMode::TiesToEven).bits());
    let b = f32::from_bits(b.to_f32(RoundingMode::TiesToEven).bits());
    let d = f32::from_bits(d.to_f32(RoundingMode::TiesToEven).bits());

    println!("{} + {} = {}", a, b, d);
}
```

## Feature

Some architectures are supported:

* 8086
* 8086-SSE (default)
* ARM-VFPv2
* ARM-VFPv2-DefaultNaN
* RISCV

You can specify architecture through feature like below:

```Cargo.toml
[dependencies.softfloat-wrapper]
version = "0.1.2"
default-features = false
features = ["riscv"]
```

## License

Licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or
conditions.
