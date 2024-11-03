use opencv::{
    core,
    highgui,
    prelude::*,
    videoio,
    imgproc,
};

struct Grid<'a> {
    value: i32,
    subgrid: Option<[&'a Grid<'a>; 4]>,
}


fn abc(mat: &core::Mat) -> Grid<'static> {
    let value = core::mean(&mat, &core::no_array()).unwrap()[0] as i32;

    let mut grid = Grid {
        value,
        subgrid: None,
    };

    return grid;
}


fn run() -> opencv::Result<()> {
    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;  // 0 is the default camera
    let opened = videoio::VideoCapture::is_opened(&cam)?;
    if !opened {
        panic!("Unable to open default camera!");
    }
    
    let width = cam.get(videoio::CAP_PROP_FRAME_WIDTH)?.round() as i32;
    let height = cam.get(videoio::CAP_PROP_FRAME_HEIGHT)?.round() as i32;
    let offset = (width - height) / 2;

    let window1 = "letters";
    let window2 = "video";
    highgui::named_window(window1, highgui::WINDOW_KEEPRATIO)?; 
    highgui::named_window(window2, highgui::WINDOW_KEEPRATIO)?; 
    highgui::resize_window(window1, height, height)?;
    highgui::resize_window(window2, height, height)?;

    let mut frame = core::Mat::default();
    let mut normal_frame = core::Mat::default();
    while 
        highgui::get_window_property(window1, 0)? >= 0.0 &&
        highgui::get_window_property(window2, 0)? >= 0.0
    {
        cam.read(&mut frame)?;

        if frame.size()?.width > 0 {
            let mut cropped_frame = core::Mat::roi(&frame, core::Rect {
                x: offset,
                y: 0,
                width: height,
                height,
            }).unwrap();
            let mut grey_frame = core::Mat::default();

            imgproc::cvt_color(&cropped_frame, &mut grey_frame, imgproc::COLOR_BGR2GRAY, 1)?;
            core::normalize(&grey_frame, &mut normal_frame, 255.0, 0.0, core::NORM_MINMAX, -1, &core::no_array())?;
            
            highgui::imshow(window1, &mut cropped_frame)?;
            highgui::imshow(window2, &mut normal_frame)?;
            
            let a = abc(&grey_frame);
            println!("{}", a.value);
        }


        // 10ms before fetching next frame, if pressed ESC key (27), exit
        if highgui::wait_key(10 as i32)? == 27 {
            break;
        }
    }
    Ok(())
}

fn main() {
    run().unwrap()
}
