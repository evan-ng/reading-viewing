
use opencv::{
    core, imgproc, prelude::*,  boxed_ref
};

pub fn compute_entropy(roi: &boxed_ref::BoxedRef<'_, core::Mat>) -> f32 {
    // Number of bins
    let hist_size = 256;

    // Range for histogram calculation (0 to 256 for pixel values)
    let hist_range = vec![0.0, 256.0];

    // Calculate the histogram
    let mut hist_vec: core::Vector<core::Mat> = core::Vector::new();
    hist_vec.push(roi.try_clone().expect("")); // We need to pass a vector of Mats

    let mut hist_out = core::Mat::default();
    
    imgproc::calc_hist(
        &hist_vec,
        &core::Vector::from_iter(vec![0]),
        &core::no_array(),
        &mut hist_out,
        &core::Vector::from_iter(vec![hist_size]),
        &core::Vector::from_iter(hist_range),
        false
    ).expect("calculate histogram failed");

    // Normalize the histogram
    let mut normalized_hist = core::Mat::default();
    core::normalize(
        &hist_out, 
        &mut normalized_hist, 
        1.0, 
        0.0, 
        core::NORM_L1, 
        -1, 
        &core::no_array()
    ).expect("normalize histogram failed");

    let mut normalized_add_hist = core::Mat::default();
    core::add(
        &normalized_hist, 
        &core::Scalar::all(1e-4), 
        &mut normalized_add_hist, 
        &core::no_array(),
        -1
    ).expect("add failed");

    // Take the log of the histogram
    let mut log_p = Mat::default();
    opencv::core::log(&normalized_add_hist, &mut log_p).expect("log failed");

    // Compute the entropy: -sum(hist * log(hist))
    let mut entropy = 0.0;
    for i in 0..hist_size {
        let hist_val = *normalized_hist.at::<f32>(i as i32).expect(""); // Get histogram value
        let log_val = *log_p.at::<f32>(i as i32).expect(""); // Get log value
        if hist_val > 0.0 {
            entropy -= hist_val * log_val; // Sum of the entropy formula
        }
    }
    return entropy;
}