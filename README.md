# TwoStepper

Hi! Learning a new dance move can be tough at any skill level. I developed TwoStepper to help dancers learn new moves. I always found that it was difficult when I was out at my favorite honkytonk to see a new move and remember how to do it later. I would always mess something up, or it wasn't quite right. TwoStepper exists so you can see the move, identify it by its name, and learn how to do it with a curated search in YouTube.

TwoStepper was built as a part of a three-week project for Insight Data Science in January 2020.

Visit TwoStepper at [www.twostepper.xyz](www.twostepper.xyz)!

## Working with the Data

Classifying dance moves has several challenges. Dancers can be viewed from any angle, any one move can take different amounts of time, and collapsing a time series into an input that is consistently recognizable is not an intuitive task.

Initially, I tried to collect a dataset of dancers dancing a select number of moves by recording the time that I saw it happen in existing tutorials on YouTube for three different moves:

* The Turn
* The Cuddle
* The Shadow

These YouTube queries resulted in lots of tutorials, and lots of instances within each video of the dancers moving around a studio from a variety of angles and proximity to the camera. Still, the process was labor-intensive, and required a lot of scrubbing back and forward within videos.

To augment the dataset, I decided to create a moving window around each known instance of the move happening in the video. One that started late, one that started early, and one that was a little long. With my clip database quadrupled in its number of elements, I then split each clip into a maximum of 30 evenly timed frames. For more details, look for the function videos_from_database in, 'classifier/frame_processer_functions.py'
