from manim import *
from manim_slides import Slide
import numpy as np
from scipy.spatial import Voronoi

class Presentation(Slide):
    def construct(self):
        title = Title("Vororoi Diagram and Fortune's algorithm")
        sub = Tex("by Romeu Esteve").next_to(title, DOWN)

        self.play(Write(title), Write(sub))
        self.next_slide()
        self.play(FadeOut(title), FadeOut(sub)) 


# --- SCENE 1: DEFINITIONS & PROPERTIES ---
class VoronoiIntro(Slide):
    def construct(self):
        # --- 1. Title & Definition ---
        title = Title("Voronoi Construction")
        def_text = Tex(
            r"For each site, we find the region closest to it\\by intersecting \textbf{perpendicular bisectors}.",
            font_size=32
        ).next_to(title, DOWN)

        self.play(Write(title), Write(def_text))
        self.next_slide()

        self.play(FadeOut(title), FadeOut(def_text))

        # --- Simple Bisector Example ---
        p1 = np.array([-2, 0.25, 0])
        p2 = np.array([2, -0.25, 0])
        dot1 = Dot(p1, color=YELLOW)
        dot2 = Dot(p2, color=YELLOW)

        # Draw the points
        self.play(GrowFromCenter(dot1), GrowFromCenter(dot2))
        
        # Draw perpendicular bisector
        midpoint = (p1 + p2) / 2
        tangent = p2 - p1
        normal = np.array([-tangent[1], tangent[0], 0], dtype=float)
        normal /= np.linalg.norm(normal)
        bisector_line = Line(midpoint - normal*3, midpoint + normal*3, color=ORANGE, stroke_width=3)
        
        self.play(Create(bisector_line))
        
        # Moving point that shows distance comparison
        test_point = Dot([-3, 1, 0], color=WHITE)
        self.add(test_point)

        # Lines from test point to the two sites
        line1 = always_redraw(lambda: Line(test_point.get_center(), dot1.get_center(), color=BLUE))
        line2 = always_redraw(lambda: Line(test_point.get_center(), dot2.get_center(), color=GREEN))

        self.add(line1, line2)

        # Distance labels
        dist_label1 = always_redraw(lambda: Tex(
            f"{np.linalg.norm(test_point.get_center() - dot1.get_center()):.2f}"
        ).next_to((test_point.get_center() + dot1.get_center())/2, UP))
        
        dist_label2 = always_redraw(lambda: Tex(
            f"{np.linalg.norm(test_point.get_center() - dot2.get_center()):.2f}"
        ).next_to((test_point.get_center() + dot2.get_center())/2, UP))

        self.add(dist_label1, dist_label2)

        # Color test point based on which site it's closer to
        def update_color(mob):
            d1 = np.linalg.norm(mob.get_center() - dot1.get_center())
            d2 = np.linalg.norm(mob.get_center() - dot2.get_center())
            mob.set_color(BLUE if d1 < d2 else GREEN)
        test_point.add_updater(update_color)

        # Animate the test point moving around
        self.play(test_point.animate.move_to([1, 2, 0]), run_time=2)
        self.wait(2)
        self.play(test_point.animate.move_to([-1, -1, 0]), run_time=2)
        self.wait(2)
        self.play(test_point.animate.move_to(midpoint), run_time=2)
        self.next_slide()

        # Fade everything out
        self.play(
            FadeOut(dot1), FadeOut(dot2), 
            FadeOut(bisector_line),
            FadeOut(test_point),
            FadeOut(line1),
            FadeOut(line2),
            FadeOut(dist_label1),
            FadeOut(dist_label2),
        )

        self.next_slide()

        
        # Define sites
        sites_coords = [[-3, -1, 0], [-1, 2, 0], [2, 1, 0], [3, -2, 0], [0, 0, 0]]
        sites = [np.array(p) for p in sites_coords]
        site_dots = VGroup(*[Dot(p, color=YELLOW, radius=0.08) for p in sites])
        
        # Calculate Voronoi
        points_2d = [p[:2] for p in sites]
        vor = Voronoi(points_2d)

        
        self.play(LaggedStart(*[GrowFromCenter(d) for d in site_dots], lag_ratio=0.1))
        self.next_slide()

        # --- 2. Construct Regions Iteratively ---
        
        # Store all final edges to keep them on screen
        final_diagram = VGroup() 

        # We need a center of mass to determine direction of infinite ridges correctly
        center_mass = vor.points.mean(axis=0)

        # Loop through every site to construct its cell
        for i, site in enumerate(sites):
            
            # 1. Highlight current site
            current_dot = site_dots[i]
            self.play(current_dot.animate.set_color(RED).scale(1.5), run_time=0.3)
            
            construction_lines = VGroup()
            cell_edges = VGroup()
            neighbor_lines = VGroup()

            # Find ridges (edges) associated with this specific point index 'i'
            # vor.ridge_points is a list of pairs [p1, p2]
            relevant_ridges_indices = []
            for ridge_idx, point_pair in enumerate(vor.ridge_points):
                if i in point_pair:
                    relevant_ridges_indices.append(ridge_idx)

            # Process each neighbor/ridge for this cell
            for ridge_idx in relevant_ridges_indices:
                point_pair = vor.ridge_points[ridge_idx]
                
                # Identify neighbor
                neighbor_idx = point_pair[1] if point_pair[0] == i else point_pair[0]
                neighbor_pos = sites[neighbor_idx]
                
                # Create visual line to neighbor
                n_line = DashedLine(site, neighbor_pos, color=GRAY, stroke_opacity=0.5)
                neighbor_lines.add(n_line)

                # --- Calculate Bisector Geometry ---
                p0_2d = vor.points[point_pair[0]]
                p1_2d = vor.points[point_pair[1]]
                tangent = p1_2d - p0_2d
                # Normal vector
                normal = np.array([-tangent[1], tangent[0]])
                normal /= np.linalg.norm(normal)
                
                midpoint_2d = (p0_2d + p1_2d) / 2
                midpoint = np.array([midpoint_2d[0], midpoint_2d[1], 0])

                # Draw Infinite Bisector (The "Construction" line)
                # We make it very long to simulate infinity
                bisector_direction = np.array([normal[0], normal[1], 0])
                start_inf = midpoint - bisector_direction * 10
                end_inf = midpoint + bisector_direction * 10
                inf_line = Line(start_inf, end_inf, color=ORANGE, stroke_width=2, stroke_opacity=0.5)
                construction_lines.add(inf_line)

                # --- Calculate Finite Edge (The Final Result) ---
                # Check vertex indices for this ridge
                v_pair = vor.ridge_vertices[ridge_idx]
                
                start_pos, end_pos = None, None

                # Case A: Finite Ridge (bounded by two vertices)
                if v_pair[0] >= 0 and v_pair[1] >= 0:
                    v0 = vor.vertices[v_pair[0]]
                    v1 = vor.vertices[v_pair[1]]
                    start_pos = np.array([v0[0], v0[1], 0])
                    end_pos = np.array([v1[0], v1[1], 0])
                
                # Case B: Infinite Ridge (bounded by one vertex, goes to infinity)
                else:
                    # Get the known vertex
                    known_v_idx = v_pair[0] if v_pair[0] >= 0 else v_pair[1]
                    v0 = vor.vertices[known_v_idx]
                    start_pos = np.array([v0[0], v0[1], 0])
                    
                    # Determine direction for the infinite part
                    # Check dot product with vector from center to ensure it points outwards
                    if np.dot(normal, v0 - center_mass) < 0:
                        normal = -normal
                    
                    direction = np.array([normal[0], normal[1], 0])
                    end_pos = start_pos + direction * 10

                edge = Line(start_pos, end_pos, color=BLUE, stroke_width=4)
                cell_edges.add(edge)

            # --- Animation Sequence for this Cell ---
            
            # 1. Show neighbors and infinite bisectors
            if (i == 0):
                self.play(Create(neighbor_lines))
                self.next_slide()
                self.play(Create(construction_lines))
                self.next_slide()
            else:
                self.play(
                    Create(neighbor_lines),
                    Create(construction_lines),
                    run_time=0.7
                )
            
            

            # 2. "Cut" the bisectors into the cell edges
            # We transform the long orange lines into the specific blue edges
            self.play(
                ReplacementTransform(construction_lines, cell_edges),
                FadeOut(neighbor_lines),
                run_time=0.7
            )

            if (i == 0): self.next_slide()
            
            # 3. Add to final group and reset dot
            final_diagram.add(cell_edges)
            self.play(current_dot.animate.set_color(YELLOW).scale(1/1.5), run_time=0.2)
            
            # Optional: Pause briefly after each cell
            # self.wait(0.2) 

        # --- 3. Final Polish ---
        self.add(final_diagram)
        
        # Highlight the center polygon
        center_region_idx = vor.point_region[4]
        region_indices = vor.regions[center_region_idx]
        if -1 not in region_indices and len(region_indices) > 0:
             poly_points = [np.array([vor.vertices[i][0], vor.vertices[i][1], 0]) for i in region_indices]
             center_poly = Polygon(*poly_points, color=BLUE, fill_color=BLUE, fill_opacity=0.2)
             self.play(FadeIn(center_poly))

        final_text = Tex(r"The complete \textbf{Voronoi Diagram}", font_size=36).to_edge(DOWN)
        self.play(Write(final_text))
        
        self.next_slide()
        
        # Cleanup
        self.play(
            FadeOut(site_dots), 
            FadeOut(final_diagram), 
            FadeOut(final_text),
            FadeOut(center_poly) if 'center_poly' in locals() else Wait(0)
        )

# --- SCENE 2: FORTUNE'S ALGORITHM BASICS ---
class FortuneBasic(Slide):
    def construct(self):
        # --- 1. Title Sequence ---
        title = Title("Fortune's Sweep-Line Algorithm", include_underline=True)
        # Using Tex for better formatting of text
        algo_text = Tex(
            r"The \textbf{Beach Line} is the lower envelope of parabolas defined by the sites.",
            font_size=34
        ).next_to(title, DOWN)
        
        self.play(Write(title))
        self.play(FadeIn(algo_text, shift=UP))
        self.next_slide()
        self.play(FadeOut(title), FadeOut(algo_text))

        # --- 2. Setup Data ---
        # Using slightly asymmetric points to make the resulting edge interesting
        p1 = np.array([-2.5, 0.75, 0])
        p2 = np.array([2.0, 1.75, 0])
        
        # Determine Perpendicular Bisector (The Voronoi Edge)
        # Midpoint
        mid = (p1 + p2) / 2
        # Vector connecting sites
        vec = p2 - p1
        # Perpendicular vector (rotate 90 deg)
        perp = np.array([-vec[1], vec[0], 0])
        # Function for the Voronoi Edge Line
        def get_voronoi_edge_point(t):
            return mid + perp * t

        # Objects
        sites = VGroup(
            Dot(p1, color=ORANGE, radius=0.12), 
            Dot(p2, color=ORANGE, radius=0.12)
        )
        site_labels = VGroup(
            MathTex("P_1", color=ORANGE, font_size=24).next_to(p1, LEFT),
            MathTex("P_2", color=ORANGE, font_size=24).next_to(p2, RIGHT)
        )
        
        # The Sweep Line (ValueTracker drives the animation)
        sweep_y = ValueTracker(0.0) 
        
        sweep_line_obj = always_redraw(lambda: VGroup(
            Line(LEFT*7, RIGHT*7, color=RED, stroke_width=2).set_y(sweep_y.get_value()),
            Text("Sweep Line", font_size=20, color=RED).next_to(RIGHT*6 + UP*sweep_y.get_value(), UP, buff=0.1)
        ))

        self.play(
            Create(sites), 
            Write(site_labels), 
            Create(sweep_line_obj)
        )

        # --- 3. Parabolas & Beach Line Logic ---
        
        def get_y_on_parabola(x, focus, directrix_y):
            # Check for divide by zero (when sweep line hits site)
            denom = 2 * (focus[1] - directrix_y)
            if abs(denom) < 0.0001: 
                return 100 # Shoot off to infinity briefly
            return ((x - focus[0])**2) / denom + (focus[1] + directrix_y) / 2

        # Dashed individual parabolas
        para1 = always_redraw(lambda: FunctionGraph(
            lambda x: get_y_on_parabola(x, p1, sweep_y.get_value()), 
            x_range=[-7, 7], color=BLUE_A, stroke_opacity=0.3, stroke_width=2
        ))
        para2 = always_redraw(lambda: FunctionGraph(
            lambda x: get_y_on_parabola(x, p2, sweep_y.get_value()), 
            x_range=[-7, 7], color=BLUE_A, stroke_opacity=0.3, stroke_width=2
        ))

        # The Beach Line (Lower Envelope)
        beach_line = always_redraw(lambda: FunctionGraph(
            lambda x: np.minimum(
                get_y_on_parabola(x, p1, sweep_y.get_value()),
                get_y_on_parabola(x, p2, sweep_y.get_value())
            ),
            x_range=[-7, 7], color=YELLOW, stroke_width=5
        ))
        
        beach_label = Text("Beach Line", color=YELLOW, font_size=24).to_corner(UL).shift(DOWN*1.5)

        self.play(Create(para1), Create(para2))
        self.play(Create(beach_line), Write(beach_label))

        # --- 4. The "Geometry Probe" (Educational Visualizer) ---
        # Shows distance to Focus == distance to Sweep Line
        probe_x_tracker = ValueTracker(-3.5)
        
        probe_lines = always_redraw(lambda: VGroup(
            DashedLine(
                start=p1,
                end=np.array([probe_x_tracker.get_value(),
                            get_y_on_parabola(probe_x_tracker.get_value(), p1, sweep_y.get_value()),
                            0]),
                color=BLUE
            ),
            DashedLine(
                start=np.array([probe_x_tracker.get_value(),
                                get_y_on_parabola(probe_x_tracker.get_value(), p1, sweep_y.get_value()),
                                0]),
                end=np.array([probe_x_tracker.get_value(), sweep_y.get_value(), 0]),
                color=BLUE
            )
        ))
        probe_text = Tex("Equidistant", font_size=20, color=BLUE).next_to(probe_lines, LEFT)

        self.play(Create(probe_lines), FadeIn(probe_text))

        self.play(probe_x_tracker.animate.set_value(-1.0), run_time=2, rate_func=smooth)
        self.play(probe_x_tracker.animate.set_value(-5.0), run_time=2, rate_func=smooth)
        self.play(probe_x_tracker.animate.set_value(-3.5), run_time=2, rate_func=smooth)

        self.wait()
        self.next_slide()

        # --- 5. Tracing the Intersection (Voronoi Edge) ---
        
        # Calculate intersection X of two parabolas analytically or via bisector logic
        # The intersection of two parabolas (directrix y=L) is on the perp bisector of foci.
        # We find the intersection of the Bisector Line and the Parabola Function.
        def get_breakpoint():
            # Current directrix
            L = sweep_y.get_value()
            
            # Simple numerical approach to find x where para1(x) approx para2(x)
            # Or use geometry: The intersection is the center of the circle passing through P1, P2 tangent to L.
            # Easier: Just intersect the Perp Bisector line with Para1.
            
            # Line eq: P = mid + perp * t
            # x(t) = mid[0] + perp[0]*t
            # y(t) = mid[1] + perp[1]*t
            # Parabola constraint: distance(P, p1) = distance(P, y=L)
            # This is complex to solve dynamically in a lambda efficiently. 
            # Let's use the property that x is the root of equal parabola equations.
            
            dp1 = 2 * (p1[1] - L)
            dp2 = 2 * (p2[1] - L)
            if abs(dp1) < 0.01 or abs(dp2) < 0.01: return np.array([0, 10, 0])

            # Solving a1 x^2 + b1 x + c1 = a2 x^2 + b2 x + c2 is messy.
            # Let's cheat slightly for visual smoothness using the Bisector Logic:
            # The breakpoint X is determined by solving (x-x1)^2 + y1^2 - L^2 / ...
            # Let's just scan a small range near the midpoint X for min(abs(y1-y2)).
            # (Note: For production math videos, use the analytic quadratic solution).
            
            # Analytic solution for X of intersection of two parabolas with same directrix:
            # x = [ x1(y2 - L) - x2(y1 - L) +/- sqrt( (y1-L)(y2-L)((x1-x2)^2 + (y1-y2)^2) ) ] / (y2 - y1)
            # If y1 = y2, x = (x1+x2)/2.
            
            if np.isclose(p1[1], p2[1]):
                x_sol = (p1[0] + p2[0]) / 2
            else:
                # To ensure we pick the correct branch, we want the one between the points usually
                # But actually, standard formula:
                num_term = np.sqrt((p1[1]-L)*(p2[1]-L)*((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2))
                # Note: This square root can be negative in the quadratic formula, but geometrically it's distinct
                # Let's simplify: Manim visualization doesn't need perfect float precision.
                # Let's rely on finding the Y on the Bisector that is equidistant to P1 and L.
                pass 

            # VISUAL HACK for stability:
            # We know the intersection lies on the perpendicular bisector.
            # We just need to find the point on the bisector where dist(pt, p1) = dist(pt, L)
            # Let pt = mid + perp * t
            # dist_sq_point = |mid + perp*t - p1|^2
            # dist_sq_line  = ( (mid[1] + perp[1]*t) - L )^2
            # solve for t. This is quadratic in t.
            
            # This is too heavy for always_redraw loop. 
            # Let's use a simpler visual approximation or assume the math works out to standard edge.
            
            # Let's use the "Newton's method" on the difference of the two parabola functions
            # starting from the midpoint x.
            guess_x = (p1[0] + p2[0])/2
            for _ in range(5):
                y1 = ((guess_x - p1[0])**2) / (2 * (p1[1] - L)) + (p1[1] + L) / 2
                y2 = ((guess_x - p2[0])**2) / (2 * (p2[1] - L)) + (p2[1] + L) / 2
                diff = y1 - y2
                # simplistic derivative approx
                dy1 = (guess_x - p1[0]) / (p1[1] - L)
                dy2 = (guess_x - p2[0]) / (p2[1] - L)
                deriv = dy1 - dy2
                if abs(deriv) > 0.001:
                    guess_x = guess_x - diff / deriv
            
            final_y = ((guess_x - p1[0])**2) / (2 * (p1[1] - L)) + (p1[1] + L) / 2
            return np.array([guess_x, final_y, 0])

        breakpoint_dot = always_redraw(lambda: Dot(get_breakpoint(), color=RED))
        
        # Trace the path of the breakpoint (The Voronoi Edge)
        trace = TracedPath(breakpoint_dot.get_center, stroke_color=GREEN, stroke_width=4)

        self.add(trace, breakpoint_dot)
        self.play(FadeOut(probe_lines), FadeOut(probe_text))

        # --- 6. Animate Sweep ---
        self.play(sweep_y.animate.set_value(-2), run_time=6, rate_func=linear)
        self.next_slide()
        
        # --- 7. Cleanup ---
        self.play(
            FadeOut(sites), FadeOut(site_labels), 
            FadeOut(sweep_line_obj), FadeOut(para1), FadeOut(para2), 
            FadeOut(beach_line), FadeOut(breakpoint_dot), FadeOut(trace),
            FadeOut(beach_label)
        )

# --- SCENE 3: EVENTS (SITE & CIRCLE) ---
class FortuneEvents(Slide):
    def construct(self):
        # --- Helper: Parabola Function ---
        def get_parabola_y(x, focus, directrix_y):
            dist = focus[1] - directrix_y
            if abs(dist) < 0.0001: return 100 # Avoid div/0, push to infinity
            return ((x - focus[0])**2) / (2 * dist) + (focus[1] + directrix_y) / 2

        # --- Helper: Circumcenter Calculation ---
        def get_circumcenter(p1, p2, p3):
            # Algorithm to find center of circle passing through 3 points
            d = 2 * (p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
            if abs(d) < 0.001: return np.array([0,0,0]) # Collinear
            ux = ((p1[0]**2 + p1[1]**2) * (p2[1] - p3[1]) + \
                  (p2[0]**2 + p2[1]**2) * (p3[1] - p1[1]) + \
                  (p3[0]**2 + p3[1]**2) * (p1[1] - p2[1])) / d
            uy = ((p1[0]**2 + p1[1]**2) * (p3[0] - p2[0]) + \
                  (p2[0]**2 + p2[1]**2) * (p1[0] - p3[0]) + \
                  (p3[0]**2 + p3[1]**2) * (p2[0] - p1[0])) / d
            return np.array([ux, uy, 0])

        # ==========================================
        # SCENE START
        # ==========================================
        title = Title("Fortune's Algorithm: The Events", include_underline=True)
        self.play(Write(title))

        # ------------------------------------------
        # 1. SITE EVENT
        # ------------------------------------------
        # Text Setup
        subtitle_site = Tex(r"\textbf{1. Site Event}: The sweep line hits a new site.", font_size=36, color=BLUE).next_to(title, DOWN)
        desc_site = Tex(r"A new narrow parabola appears and splits the arc above it.", font_size=28).next_to(subtitle_site, DOWN)
        
        self.play(FadeIn(subtitle_site, shift=UP), FadeIn(desc_site, shift=UP))
        self.next_slide()

        # Data
        p_old = np.array([0, 1.0, 0])
        p_new = np.array([0.5, -1.0, 0])
        
        sweep_y = ValueTracker(0.75)
        
        # Objects
        dot_old = Dot(p_old, color=YELLOW)
        dot_new = Dot(p_new, color=GREEN) # The new site
        
        # We start with dot_new hidden or outlined
        dot_new_ring = Circle(radius=0.15, color=GREEN).move_to(p_new)
        
        sweep_line = always_redraw(lambda: Line(LEFT*7, RIGHT*7, color=RED).set_y(sweep_y.get_value()))
        
        # Beach Line logic with "Existence check" for the second parabola
        beach_line_site = always_redraw(lambda: FunctionGraph(
            lambda x: min(
                get_parabola_y(x, p_old, sweep_y.get_value()),
                # Only include second parabola if sweep line is below the site
                get_parabola_y(x, p_new, sweep_y.get_value()) if sweep_y.get_value() < p_new[1] - 0.01 else 100
            ),
            x_range=[-6,6], color=YELLOW, stroke_width=4
        ))

        self.play(Create(dot_old), Create(dot_new_ring), Create(sweep_line), Create(beach_line_site))
        
        # Animate Sweep to just before the site
        self.play(sweep_y.animate.set_value(-1.01), run_time=1.5)
        
        # THE EVENT HAPPENS
        self.play(
            ReplacementTransform(dot_new_ring, dot_new),
            Flash(p_new, color=GREEN, flash_radius=0.5)
        )
        
        # Continue Sweep (Parabola emerges)
        self.play(sweep_y.animate.set_value(-1.5), run_time=3)
        
        self.next_slide()

        # ------------------------------------------
        # 2. CIRCLE EVENT
        # ------------------------------------------
        # Cleanup
        self.play(
            FadeOut(Group(subtitle_site, desc_site, dot_old, dot_new, beach_line_site)),
            sweep_y.animate.set_value(2.5) # Reset sweep line upwards
        )
        
        subtitle_circ = Tex(r"\textbf{2. Circle Event}: An arc disappears.", font_size=36, color=BLUE).next_to(title, DOWN)
        desc_circ = Tex(r"Occurs when the sweep line is tangent to the circle formed by 3 sites.", font_size=28).next_to(subtitle_circ, DOWN)
        
        self.play(Write(subtitle_circ), Write(desc_circ))

        # Data for 3 sites that form a nice triangle
        s1 = np.array([-2, 0.5, 0])
        s2 = np.array([2, 0.5, 0])
        s3 = np.array([0, -1.0, 0])
        
        sites_group = VGroup(Dot(s1), Dot(s2), Dot(s3)).set_color(ORANGE)
        
        # Math: Calculate exact event location
        center = get_circumcenter(s1, s2, s3)
        radius = np.linalg.norm(s1 - center)
        event_y = center[1] - radius # Bottom of the circle
        
        # Redefine beach line for 3 sites
        beach_line_circ = always_redraw(lambda: FunctionGraph(
            lambda x: min(
                get_parabola_y(x, s1, sweep_y.get_value()),
                get_parabola_y(x, s2, sweep_y.get_value()),
                get_parabola_y(x, s3, sweep_y.get_value())
            ),
            x_range=[-6,6], color=YELLOW
        ))

        self.play(Create(sites_group), Create(beach_line_circ))

        # Sweep down to near the event
        self.play(sweep_y.animate.set_value(event_y + 0.5), run_time=2)
        
        # Show the "Ghost Circle" to explain why it happens here
        ghost_circle = DashedVMobject(Circle(radius=radius, color=BLUE_B).move_to(center))
        vertex_dot = Dot(center, color=RED)
        vertex_label = Text("Voronoi Vertex", font_size=20, color=RED).next_to(vertex_dot, UP)

        self.play(Create(ghost_circle))
        
        # Animate to exact event moment
        self.play(
            sweep_y.animate.set_value(event_y), 
            run_time=2, 
            rate_func=linear
        )
        
        # Mark the vertex
        self.play(FadeIn(vertex_dot), FadeIn(vertex_label), Flash(center, color=RED))
        
        # Continue past event (Arc disappears)
        self.play(sweep_y.animate.set_value(event_y - 1.5), run_time=2)
        
        self.next_slide()

        # ------------------------------------------
        # 3. CONCLUSION
        # ------------------------------------------
        self.play(FadeOut(Group(sites_group, beach_line_circ, ghost_circle, vertex_dot, vertex_label, sweep_line, subtitle_circ, desc_circ)))
        
        final_text = Tex(r"Site Events $\rightarrow$ Create Arcs \\ Circle Events $\rightarrow$ Delete Arcs", font_size=40)
        self.play(Write(final_text))
        
        self.wait()
        self.play(FadeOut(final_text), FadeOut(title))