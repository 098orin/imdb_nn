pub mod linear;
pub mod sparselinear;

pub(crate) fn init_weight(fan_in: usize) -> f32 {
    // 簡易擬似乱数（LCG）
    static mut SEED: u32 = 123456789;
    let rand_0_1 = unsafe {
        SEED = SEED.wrapping_mul(1103515245).wrapping_add(12345);
        (SEED as f32) / (u32::MAX as f32) // [0,1)
    };

    // He初期化の範囲
    let std = (2.0 / fan_in as f32).sqrt();
    (rand_0_1 * 2.0 - 1.0) * std // [-std, +std]
}
