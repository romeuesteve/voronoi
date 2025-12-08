from manim import *
from manim_slides import Slide
import numpy as np
from scipy.spatial import Voronoi

# --- SCENE 1: DEFINITIONS & PROPERTIES (FIXED) ---
class VoronoiIntro(Slide):
    def construct(self):
        # 1. Title & Definition
        title = Title("Voronoi Diagrams")
        def_text = Tex(
            r"Partitions the plane so each site's region (cell) contains \\ all points closer to it than to any other.",
            font_size=32
        ).next_to(title, DOWN)
        
        sites = [np.array(p) for p in [[-3, -1, 0], [-1, 2, 0], [2, 1, 0], [3, -2, 0], [0, 0, 0]]]
        site_dots = VGroup(*[Dot(p, color=YELLOW, radius=0.1) for p in sites])
        
        self.play(Write(title), Write(def_text))
        self.play(LaggedStart(*[GrowFromCenter(d) for d in site_dots], lag_ratio=0.1))
        self.next_slide()

        # 2. Construction by Bisectors
        self.play(FadeOut(def_text))
        bisector_text = Tex(
            r"Boundaries are \textbf{perpendicular bisectors} between sites.",
            font_size=32
        ).to_edge(DOWN)
        
        # Demo specific bisector
        p1, p2 = sites[4], sites[2]
        line_connect = Line(p1, p2, color=GRAY, stroke_opacity=0.5)
        mid = (p1+p2)/2
        v = p2-p1
        
        # Calculate normal vector (perpendicular to v) and normalize
        normal_dir = np.array([-v[1], v[0], 0], dtype=float)
        normal_dir /= np.linalg.norm(normal_dir) 
        
        # Scale for better visibility (Line of length 14 units)
        scale = 7 
        start_point = mid - normal_dir * scale
        end_point = mid + normal_dir * scale

        bisector = Line(start_point, end_point, color=BLUE)
        
        self.play(Write(bisector_text))
        self.play(Create(line_connect), Create(bisector))
        self.next_slide()
        
        self.play(FadeOut(line_connect), FadeOut(bisector), FadeOut(bisector_text))

        # 3. Full Diagram & Convexity
        points_2d = [p[:2] for p in sites]
        vor = Voronoi(points_2d)
        voronoi_lines = VGroup()
        
        # Draw Finite ridges
        for v_pair in vor.ridge_vertices:
            if v_pair[0] >= 0 and v_pair[1] >= 0:
                v0 = vor.vertices[v_pair[0]]
                v1 = vor.vertices[v_pair[1]]
                voronoi_lines.add(Line([v0[0], v0[1], 0], [v1[0], v1[1], 0], color=BLUE, stroke_width=4))

        # Draw Infinite ridges (FIXED: ensured element-wise addition with NumPy arrays)
        center_mass = vor.points.mean(axis=0)
        for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
            v_pair = simplex
            if v_pair[0] < 0 or v_pair[1] < 0:
                i = v_pair[0] if v_pair[0] >= 0 else v_pair[1]
                v0 = vor.vertices[i]
                p0, p1 = vor.points[pointidx[0]], vor.points[pointidx[1]]
                tangent = p1 - p0
                normal = np.array([-tangent[1], tangent[0]])
                normal /= np.linalg.norm(normal)
                if np.dot(normal, v0 - center_mass) < 0: normal = -normal
                
                # --- FIX Applied Here ---
                start_point = np.array([v0[0],v0[1],0])
                # Use np.array for the addition to ensure element-wise operation (not list concatenation)
                direction_vector = np.array([normal[0]*10, normal[1]*10, 0]) 
                end_point = start_point + direction_vector
                voronoi_lines.add(Line(start_point, end_point, color=BLUE, stroke_width=4))
                # --- End Fix ---

        prop_text = Tex(
            r"Each cell $V(p_i)$ is a \textbf{Convex Polygon}.",
            font_size=32
        ).to_edge(DOWN)
        
        self.play(Create(voronoi_lines), run_time=2)
        self.play(Write(prop_text))
        
        # Highlight Center Cell
        center_region = vor.regions[vor.point_region[4]]
        poly_points = [np.array([vor.vertices[i][0], vor.vertices[i][1], 0]) for i in center_region if i != -1]
        highlight = Polygon(*poly_points, color=BLUE, fill_opacity=0.3, stroke_width=0)
        
        self.play(FadeIn(highlight))
        self.next_slide()

        # 4. Duality (Delaunay)
        self.play(FadeOut(prop_text), FadeOut(highlight))
        dual_text = Tex(
            r"The \textbf{Delaunay Triangulation} is the dual graph.",
            font_size=32
        ).to_edge(DOWN)
        
        delaunay_lines = VGroup()
        for pair in vor.ridge_points:
            delaunay_lines.add(Line(sites[pair[0]], sites[pair[1]], color=ORANGE, stroke_opacity=0.8))
            
        self.play(Write(dual_text))
        self.play(voronoi_lines.animate.set_stroke(opacity=0.2), Create(delaunay_lines))
        self.next_slide()
        
        self.play(FadeOut(title), FadeOut(site_dots), FadeOut(voronoi_lines), FadeOut(delaunay_lines), FadeOut(dual_text))


# --- SCENE 2: FORTUNE'S ALGORITHM BASICS ---
class FortuneBasic(Slide):
    def construct(self):
        title = Title("Fortune's Sweep-Line Algorithm")
        algo_text = Tex(r"Sweep line moves down. 'Beach line' is lower envelope of parabolas.", font_size=30).next_to(title, DOWN)
        
        self.play(Write(title), Write(algo_text))
        
        # Setup Sites
        p1 = np.array([-2, 2, 0])
        p2 = np.array([2, 3, 0])
        sites = VGroup(Dot(p1, color=YELLOW), Dot(p2, color=YELLOW))
        
        sweep_y = ValueTracker(2.5) # Start above
        sweep_line = always_redraw(lambda: Line(LEFT*7, RIGHT*7, color=RED).set_y(sweep_y.get_value()))
        
        # Parabola Functions
        def get_para(focus, y_dir):
            if np.isclose(focus[1], y_dir): return lambda x: 100
            return lambda x: ((x - focus[0])**2) / (2 * (focus[1] - y_dir)) + (focus[1] + y_dir) / 2

        # Draw full parabolas (dashed) to show underlying geometry
        para1_full = always_redraw(lambda: FunctionGraph(get_para(p1, sweep_y.get_value()), x_range=[-7,7], color=BLUE_A, stroke_opacity=0.3))
        para2_full = always_redraw(lambda: FunctionGraph(get_para(p2, sweep_y.get_value()), x_range=[-7,7], color=BLUE_A, stroke_opacity=0.3))

        # The Beach Line (Lower Envelope)
        def beach_line_func(x):
            y1 = get_para(p1, sweep_y.get_value())(x)
            y2 = get_para(p2, sweep_y.get_value())(x)
            return min(y1, y2)
            
        beach_line = always_redraw(lambda: FunctionGraph(beach_line_func, x_range=[-7,7], color=YELLOW, stroke_width=4))

        self.play(Create(sites), Create(sweep_line))
        self.play(Create(para1_full), Create(para2_full), Create(beach_line))
        self.next_slide()
        
        # Animate Sweep
        self.play(sweep_y.animate.set_value(-2), run_time=4)
        self.next_slide()
        
        self.play(FadeOut(Group(sites, sweep_line, para1_full, para2_full, beach_line, title, algo_text)))


# --- SCENE 3: EVENTS (SITE & CIRCLE) ---
class FortuneEvents(Slide):
    def construct(self):
        title = Title("Algorithm Events")
        
        # 1. Site Event
        # -------------
        event_text = Tex(r"\textbf{Site Event}: New arc splits an existing arc.", font_size=32).next_to(title, DOWN)
        self.play(Write(title), Write(event_text))
        
        p_old = np.array([0, 2, 0])
        p_new = np.array([0.5, 0, 0]) # Site that will appear
        
        sweep_y = ValueTracker(1.0)
        
        # Setup visuals
        dot_old = Dot(p_old, color=YELLOW)
        dot_new = Dot(p_new, color=YELLOW, fill_opacity=0) # Invisible initially
        sweep_line = always_redraw(lambda: Line(LEFT*7, RIGHT*7, color=RED).set_y(sweep_y.get_value()))
        
        def get_para(f, y): return lambda x: ((x - f[0])**2) / (2 * (f[1] - y)) + (f[1] + y) / 2
        
        # We only draw the beach line logic here for simplicity
        beach_env = always_redraw(lambda: FunctionGraph(
            lambda x: min(get_para(p_old, sweep_y.get_value())(x), 
                          get_para(p_new, sweep_y.get_value())(x) if sweep_y.get_value() < 0 else 100),
            x_range=[-4,4], color=YELLOW
        ))

        self.play(Create(dot_old), Create(sweep_line), Create(beach_env))
        
        # Sweep down until just before p_new
        self.play(sweep_y.animate.set_value(0.1), run_time=1.5)
        
        # Trigger Site Event
        self.play(dot_new.animate.set_fill(opacity=1), run_time=0.2)
        self.play(sweep_y.animate.set_value(-2), run_time=3)
        self.next_slide()

        # 2. Circle Event
        # ---------------
        self.play(FadeOut(Group(dot_old, dot_new, sweep_line, beach_env, event_text)))
        
        circle_text = Tex(r"\textbf{Circle Event}: Arcs converge, creating a Voronoi Vertex.", font_size=32).next_to(title, DOWN)
        self.play(Write(circle_text))
        
        # Three sites arranged to cause a collapse
        s1 = np.array([-2, 1, 0])
        s2 = np.array([2, 1, 0])
        s3 = np.array([0, -1, 0]) # Lower down
        dots = VGroup(Dot(s1), Dot(s2), Dot(s3))
        
        sweep_y.set_value(0.5)
        
        # Redefine beach line for 3 sites
        def beach_3(x):
            y = sweep_y.get_value()
            v1 = get_para(s1, y)(x)
            v2 = get_para(s2, y)(x)
            v3 = get_para(s3, y)(x)
            return min(v1, v2, v3)

        beach_line_3 = always_redraw(lambda: FunctionGraph(beach_3, x_range=[-5,5], color=YELLOW))
        
        self.play(Create(dots), Create(sweep_line), Create(beach_line_3))
        
        # Animate sweep past the circle event
        self.play(sweep_y.animate.set_value(-3), run_time=5)
        
        # Highlight the vertex (approximate location for visual)
        vertex = Dot([0, 0.25, 0], color=ORANGE, radius=0.15)
        vertex_label = Text("Vertex", font_size=20, color=ORANGE).next_to(vertex, UP)
        self.play(FadeIn(vertex), FadeIn(vertex_label))
        
        self.next_slide()
        
        # 3. Complexity Summary
        self.play(FadeOut(Group(dots, sweep_line, beach_line_3, circle_text, vertex, vertex_label)))
        
        final_title = Title("Complexity & Summary")
        bullets = VGroup(
            Tex(r"$\bullet$ Naive Construction: $O(n^2)$", font_size=36),
            Tex(r"$\bullet$ Fortune's Algorithm: $O(n \log n)$", font_size=36),
            Tex(r"$\bullet$ Space Complexity: $O(n)$", font_size=36),
            Tex(r"$\bullet$ Events: $O(n)$ site/circle events processed in $O(\log n)$", font_size=36),
        ).arrange(DOWN, aligned_edge=LEFT).next_to(final_title, DOWN)
        
        self.play(Write(final_title))
        self.play(LaggedStart(*[Write(b) for b in bullets], lag_ratio=0.5))
        self.wait(2)