//! WGSL kernel generators (f32 activations; bf16 weights unpacked in-shader).
//!
//! One module per kernel family. The model stages keep addressing them as
//! `crate::shaders::NAME`, so how the families are cut up here is free to
//! change -- and a shader only has to be re-read where its family lives.

mod attention;
mod elementwise;
mod gemv;
mod gemv_t1;
mod gemv_t6;
mod prefill;

pub use attention::{ATTEND, ATTEND_SIMPLE, KV_WRITE, ROPE};
pub use elementwise::{
    ADD, CFM_STEP, GELU_TANH, GELU_TANH_STABLE, PACK3, PACK6, RMSNORM, SILU_MUL,
};
pub use gemv::{GEMV_BF16W, GEMV_MROW, GEMV_MROW_T4, GEMV_MROW_T8};
pub use gemv_t1::gemv_t1_vec4;
pub use gemv_t6::{gemv_t6_vec4, GEMV_V4_T6};
pub use prefill::GEMM_PREFILL_48X64;
