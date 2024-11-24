pub mod entropy;
pub mod letters;

use entropy::compute_entropy;
use letters::Letters;

use std::{collections::VecDeque, env};
use opencv::{
    core, highgui, imgproc, prelude::*, videoio
};

const GRID_CHANGE_COUNT: i32 = 5;
const DEPTH_LIMIT: i32 = 5;
const ENTROPY_THRESH: f32 = 4.4;
const ENTROPY_EXIT_THRESH: f32 = 4.8;
const MAX_HIGHLIGHTED_RECTS: usize = 14;
const NON_HIGHLIGHT_ADJUST: i32 = 4;
const NON_HIGHLIGHT_MIN: i32 = 16;
const HIGHLIGHT_ADJUST: i32 = 1;

struct Grid {
    value: u8,
    entropy: f32,
    subgrid: Option<[Box<Grid>; 4]>,
    rect: core::Rect,
}

impl Grid {
    fn new(value: u8, entropy: f32, subgrid: Option<[Box<Grid>; 4]>, rect: core::Rect) -> Self {
        return Self {
            value,
            entropy,
            subgrid,
            rect,
        };
    }
}

// determines whether the grid should be split
// given the entropy (using threshold constants)
fn is_split(entropy: f32, grid: &Grid) -> bool {
    if grid.subgrid.is_none() {
        return entropy > ENTROPY_THRESH;
    } else {
        return entropy > ENTROPY_EXIT_THRESH;
    }
}

// returns true if the previous grid allows a split
// i.e. previous grid exists for corresponding current grid
fn can_split(prev_grid: &Option<&Box<Grid>>) -> bool {
    // can split if previous grid exists
    // (previous grid had been split, or 
    // curr grid is leaf split of previous grid)
    return prev_grid.is_some();
}

// returns true if the previous grid allows a merge
// i.e. all subgrids do not contain subgrids of their own
fn can_merge(prev_grid: &Option<&Box<Grid>>) -> bool {
    // can merge if:
    // prev_grid exists, prev_grid has subgrids,
    // and none of the subgrids have subgrids
    let mut can_merge = false;
    if prev_grid.is_some() {
        let subgrid = &prev_grid.as_ref().expect("msg").subgrid;
        if subgrid.is_some() {
            can_merge = true;
            for s in subgrid.as_ref().expect("msg") {
                if s.subgrid.is_some() {
                    return false;
                }
            }
        }
    }
    return can_merge;
}

// returns a fixed length array of `prev_grid`'s subgrids
// if they exist, otherwise a fixed length array of none
fn get_sub_prev_grids<'a>(prev_grid: &'a Option<&'a Box<Grid>>) -> [Option<&'a Box<Grid>>; 4] {
    
    let mut sub_prev_grids: [Option<&Box<Grid>>; 4] = [None; 4];

    if prev_grid.is_some() {
        let subgrid = &prev_grid.as_ref().expect("msg").subgrid;
        if subgrid.is_some() {
            for (i, s) in subgrid.as_ref().expect("msg").into_iter().enumerate() {
                sub_prev_grids[i] = Some(s);
            }
        }
    }

    return sub_prev_grids;
}

fn create_grid(
    mat: &core::Mat, 
    dest: &mut core::Mat, 
    depth: i32, 
    rect: core::Rect, 
    letters: &mut Letters, 
    prev_grid: &Option<&Box<Grid>>,
) -> Box<Grid> {
    let offset = if rect.width % 2 == 0 {0} else {1};
    let size = rect.width >> 1;
    let mid_x = rect.x + size - offset;
    let mid_y = rect.y + size - offset;

    let roi = core::Mat::roi(mat, rect).expect("roi failed");
    let entropy = compute_entropy(&roi);
    let value = core::mean(&roi, &core::no_array()).unwrap()[0] as u8;

    let sub_rects = [
        core::Rect::new(rect.x, rect.y, size, size),
        core::Rect::new(mid_x, rect.y, size, size),
        core::Rect::new(rect.x, mid_y, size, size),
        core::Rect::new(mid_x, mid_y, size, size),
    ];

    let mut c = vec![];
    c.push(value);

    let mut grid = Grid::new(value, entropy, None, rect);

    if depth < DEPTH_LIMIT && is_split(entropy, &grid) && can_split(prev_grid) {
        let sub_prev_grids = get_sub_prev_grids(prev_grid);
        let _ = grid.subgrid.insert([
            create_grid(mat, dest, depth+1, sub_rects[0], letters, &sub_prev_grids[0]), 
            create_grid(mat, dest, depth+1, sub_rects[1], letters, &sub_prev_grids[1]), 
            create_grid(mat, dest, depth+1, sub_rects[2], letters, &sub_prev_grids[2]), 
            create_grid(mat, dest, depth+1, sub_rects[3], letters, &sub_prev_grids[3]),
        ]);
    } else if grid.subgrid.is_some() && can_merge(prev_grid) {
        grid.subgrid = None;
    }

    if grid.subgrid.is_none() {
        let mut sub_means: [f64; 4] = Default::default();
        for (i, l) in sub_rects.into_iter().enumerate() {
            let sub_roi = core::Mat::roi(mat, l).expect("roi failed");
            sub_means[i] = core::mean(&sub_roi, &core::no_array()).unwrap()[0];
        }

        let letter_mat = letters.get_img(sub_means);
        let mut resized_letter_mat = core::Mat::default();
        imgproc::resize(
            &letter_mat, 
            &mut resized_letter_mat, 
            core::Size::new(rect.width, rect.width), 
            0.0, 
            0.0, 
            imgproc::INTER_AREA
        ).expect("resized_frame failed");

        core::copy_to(&resized_letter_mat, &mut core::Mat::roi_mut(dest, rect).expect("roi failed"), &core::no_array()).expect("");
    }

    return Box::new(grid);
}

fn distance(x1: i32, y1: i32, x2: i32, y2: i32) -> i32 {
    return (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1);
}

fn get_next_rect(grid: &Box<Grid>, rect: core::Rect, size: i32) -> core::Rect {
    let mut queue: VecDeque<&Box<Grid>> = VecDeque::new();

    let mut best_rect = rect;
    let mut best_dist = size.pow(3);

    queue.push_back(grid);

    while queue.len() > 0 {
        let grid_entry = queue.pop_front();
        if grid_entry.is_none() {
            continue;
        }

        let g = grid_entry.expect("");
        let rect_end_x = rect.x + rect.width;
        let rect_end_y = rect.y + rect.height;
        
        if rect_end_x < size && g.rect.x + g.rect.width <= rect_end_x {
            continue;
        }

        if g.subgrid.is_some() {
            for s in g.subgrid.as_ref().expect("") {
                queue.push_back(s);
            }
        } else {
            if rect_end_x >= size { // wrap to next line
                // wrap to top, return top left rect
                if rect_end_y >= size {
                    if g.rect.x == 0 && g.rect.y == 0 {
                        return g.rect;
                    }
                }
                // must be below rect
                if g.rect.y >= rect_end_y {
                    // minimize distance between (0, bottom of rect)
                    // and top left of best_rect
                    let g_dist = distance(0, rect_end_y, g.rect.x, g.rect.y);
                    if g_dist < best_dist {
                        best_rect = g.rect;
                        best_dist = g_dist;
                    }
                }
            } else { // continue on line
                // must be right of rect
                if g.rect.x + g.rect.width > rect_end_x {
                    // minimize distance between bottom right of rect
                    // and bottom left of best_rect
                    let mut g_dist = distance(rect_end_x, rect_end_y, g.rect.x, g.rect.y + g.rect.height);
                    if g.rect.y + g.rect.height < rect_end_y {
                        g_dist = g_dist * 3; // favour rects below rect
                    }
                    if g_dist < best_dist {
                        best_rect = g.rect;
                        best_dist = g_dist;
                    }
                }
            }
        }
    }

    return best_rect;

}

fn highlight_rects(
    src: &core::Mat, 
    dest: &mut core::Mat, 
    rects: &VecDeque<core::Rect>, 
) {
    let dark = NON_HIGHLIGHT_ADJUST * MAX_HIGHLIGHTED_RECTS as i32;

    let mut subtracted = core::Mat::default();
    core::subtract(
        &src, 
        &core::Scalar::all(dark as f64),
        &mut subtracted, 
        &core::no_array(),
        -1,
    ).expect("subtract failed");

    core::max(
        &subtracted, 
        &core::Scalar::all(NON_HIGHLIGHT_MIN as f64),
        dest, 
    ).expect("max failed");

    let adjust_less_max = MAX_HIGHLIGHTED_RECTS - rects.len();
    let half_highlight_rects = MAX_HIGHLIGHTED_RECTS >> 1;
    for (i, rect) in rects.into_iter().enumerate() {
        if i > half_highlight_rects {
            let scale = (HIGHLIGHT_ADJUST * (i + adjust_less_max - half_highlight_rects) as i32).pow(3) >> 1;

            let mut added = core::Mat::default();
            core::add(
                &core::Mat::roi(src, *rect).expect("roi failed"),
                &core::Scalar::all(scale as f64),
                &mut added,
                &core::no_array(),
                -1,
            ).expect("add failed");
            
            core::copy_to(
                &added, 
                &mut core::Mat::roi_mut(dest, *rect).expect("roi failed"), 
                &core::no_array(),
            ).expect("");
        } else {
            let scale = dark >> i as i32;
            let min = NON_HIGHLIGHT_MIN - (i << 1) as i32;

            let mut subtracted = core::Mat::default();
            core::subtract(
                &core::Mat::roi(src, *rect).expect("roi failed"),
                &core::Scalar::all(scale as f64),
                &mut subtracted, 
                &core::no_array(),
                -1,
            ).expect("subtract failed");

            let mut max = core::Mat::default();
            core::max(
                &subtracted, 
                &core::Scalar::all(min as f64),
                &mut max, 
            ).expect("max failed");

            core::copy_to(
                &max, 
                &mut core::Mat::roi_mut(dest, *rect).expect("roi failed"), 
                &core::no_array(),
            ).expect("");
        }
    }
}


fn run() -> opencv::Result<()> {
    let args: Vec<String> = env::args().collect();
    let one_window = args.len() > 1 && args[1].clone() == "one";

    // initialize letter images from assets directory
    let mut letters = Letters::init();

    // initialize video capture from camera
    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;  // 0 is the default camera
    let opened = videoio::VideoCapture::is_opened(&cam)?;
    if !opened {
        panic!("Unable to open default camera!");
    }
    
    // compute size of video and necessary 
    let width = cam.get(videoio::CAP_PROP_FRAME_WIDTH)?.round() as u64;
    let height = cam.get(videoio::CAP_PROP_FRAME_HEIGHT)?.round() as u64;
    let offset = ((width).abs_diff(height) >> 1) as i32;
    let orig_size = height as i32;
    let size = height.next_power_of_two() as i32;

    // create windows for display
    let window1 = if one_window {"reading viewing"} else {"image"};
    let window2 = "word";
    highgui::named_window(window1, highgui::WINDOW_KEEPRATIO)?; 
    if one_window {
        highgui::resize_window(window1, size, size>>1)?;
    } else {
        highgui::resize_window(window1, size>>1, size>>1)?;
        highgui::named_window(window2, highgui::WINDOW_KEEPRATIO)?; 
        highgui::resize_window(window2, size>>1, size>>1)?;
    }

    let mut curr_rect_queue: VecDeque<core::Rect> = VecDeque::new();
    let mut curr_rect: core::Rect = core::Rect::new(0, 0, size, size);

    let mut frame = core::Mat::default();
    let mut resized_frame = core::Mat::default();
    let mut flipped_frame = core::Mat::default();
    let mut grey_frame = core::Mat::default();
    let mut normal_frame = core::Mat::default();
    let mut prev_grid: Box<Grid> = Box::new(Grid::new(0, 0.0, None, core::Rect::new(0, 0, 0, 0)));
    let mut grid_change_count = 0;

    while 
        highgui::get_window_property(window1, 0)? >= 0.0 &&
        (one_window || highgui::get_window_property(window2, 0)? >= 0.0)
    {
        cam.read(&mut frame)?;

        if frame.size()?.width > 0 {
            // crop video frame to square
            let cropped_frame = core::Mat::roi(&frame, core::Rect {
                x: offset,
                y: 0,
                width: orig_size,
                height: orig_size,
            }).expect("cropped_frame failed");
            
            // resize to nearest power of 2 size
            imgproc::resize(&cropped_frame, &mut resized_frame, core::Size::new(size, size), 0.0, 0.0, imgproc::INTER_LINEAR).expect("resized_frame failed");
            // mirror video
            core::flip(&resized_frame, &mut flipped_frame, 1)?;

            // convert to black-and-white
            imgproc::cvt_color(&flipped_frame, &mut grey_frame, imgproc::COLOR_BGR2GRAY, 1)?;
            
            // normalize to min 0, max 255 for word square
            core::normalize(&grey_frame, &mut normal_frame, 255.0, 0.0, core::NORM_MINMAX, -1, &core::no_array())?;
            
            // create word frame using grid
            let mut word_frame = core::Mat::zeros(size, size, core::CV_8U).expect("").to_mat().expect("msg");
            let curr_grid = create_grid(
                &normal_frame, 
                &mut word_frame, 
                0, 
                core::Rect::new(0,0,size,size), 
                &mut letters, 
                &Some(&prev_grid)
            );

            // draw highlights for image frame
            curr_rect = get_next_rect(&curr_grid, curr_rect, size);
            curr_rect_queue.push_back(curr_rect);
            if curr_rect_queue.len() > MAX_HIGHLIGHTED_RECTS {
                curr_rect_queue.pop_front();
            }
            let mut image_frame = core::Mat::default();
            highlight_rects(&grey_frame, &mut image_frame, &curr_rect_queue);

            // show image and word frames
            if one_window {
                let mut concat = core::Mat::default();
                core::hconcat2(&image_frame, &word_frame, &mut concat).expect("concat failed");
                highgui::imshow(window1, &mut concat)?;
            } else {
                highgui::imshow(window1, &mut image_frame)?;
                highgui::imshow(window2, &mut word_frame)?;
            }

            // update grid change count or reset and update grid
            if grid_change_count >= GRID_CHANGE_COUNT {
                prev_grid = curr_grid;
                grid_change_count = 0;
            } else {
                grid_change_count += 1;
            }
        }

        // 48ms before fetching next frame, if pressed ESC key (27), exit
        if highgui::wait_key(48 as i32)? == 27 {
            break;
        }
    }
    Ok(())
}

fn main() {
    run().unwrap()
}
