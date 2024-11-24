use std::fs::read_to_string;
use serde_json::{self};
use opencv::{
    core, prelude::*, imgcodecs
};
use kiddo::{KdTree, SquaredEuclidean};

const NUM_IMG: usize = 26;

pub struct Letters {
    kdtree: KdTree<f64, 4>,
    letter_imgs: [core::Mat; NUM_IMG],
}

impl Letters {
    pub fn init() -> Self  {

        let mut kdtree: KdTree<f64, 4> = KdTree::new();

        let mut letter_imgs: [Mat; NUM_IMG] = Default::default();

        let str = read_to_string("assets/data.json").unwrap();
        print!("{}\n", str);
        let json: serde_json::Value = serde_json::from_str(&str).expect("JSON was not well-formatted");

        let letter_averages = json["letters"].as_array().expect("JSON is incorrect type (letters is not array)");

        for (i, l) in letter_averages.iter().enumerate() {
            if i > NUM_IMG { break; }

            let letter = l.as_array().expect("JSON is incorrect type (letters entry is not array)");
            let filename = letter[0].as_str().expect("JSON is incorrect type (filenames is not string)");
            let avg1 = letter[1].as_f64().expect("JSON is incorrect type (average is not f64)");
            let avg2 = letter[2].as_f64().expect("JSON is incorrect type (average is not f64)");
            let avg3 = letter[3].as_f64().expect("JSON is incorrect type (average is not f64)");
            let avg4 = letter[4].as_f64().expect("JSON is incorrect type (average is not f64)");

            let src = imgcodecs::imread(
                &("assets/letters/".to_string() + &filename), 
                imgcodecs::IMREAD_GRAYSCALE
            ).expect("Read image failed");
        
            letter_imgs[i] = src;

            kdtree.add(&[avg1, avg2, avg3, avg4], i as u64);

        }

        return Self{
            kdtree,
            letter_imgs,
        };
    }

    pub fn get_img(&mut self, val: [f64; 4]) -> &core::Mat {
        let index = self.kdtree.nearest_one::<SquaredEuclidean>(&val).item;
        return &self.letter_imgs[index as usize];
    }
}
