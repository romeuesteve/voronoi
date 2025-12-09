# Voronoi Presentation

Voronoi Diagram and Fortune's algorithm presentation for the GEOC Course (25-26 Q1).

---

## Video
<video width="800" controls>
  <source src="video.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

## Usage

The video uses mainly the Manim Community library for its animated slides.

Run this script to use
```sh
# create venv
python -m venv venv
. venv/bin/activate

# install requirements (might take a while)
pip install -r requirements.txt

# render the slides
manim render -ql vor.py

# view slides
manim-slides
```


## References

### Resources
- clideo.com for editing
- [voronoi with cones](https://ophysics.com/fs5.html)
- [fortune's algorithm video](https://www.youtube.com/watch?v=FQjUFLy6s6w)

### Research
- [en.wikipedia.org](https://en.wikipedia.org/wiki/Voronoi_diagram)
- [cs.princeton.edu](https://www.cs.princeton.edu/courses/archive/spring12/cos423/bib/vor.pdf#:~:text=%E2%80%A2%20The%20number%20of%20Voronoi,%E2%88%88%20S%2C%20then%20the%20Voronoi)
- [cs.umd.edu](https://www.cs.umd.edu/class/spring2020/cmsc754/Lects/lect11-vor.pdf#:~:text=The%20analysis%20follows%20a%20typical,n)