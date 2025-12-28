pub mod linear;
pub mod sparselinear;

pub(crate) fn init_weight() -> f32 {
    // 簡易擬似乱数
    static mut SEED: u32 = 123456789;
    unsafe {
        SEED = SEED.wrapping_mul(1103515245).wrapping_add(12345);
        (SEED as f32 / u32::MAX as f32 - 0.5) * 0.1
    }
}
