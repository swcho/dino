A clean flat-design educational infographic, 16:9 landscape, titled "DINO: One Training Iteration — 12 Steps" at the top center in a bold sans-serif.

Layout: one large serpentine (boustrophedon) flowchart filling the canvas — 3 rows by 4 columns of numbered rounded-square nodes. Row 1 reads left to right (nodes 1,2,3,4), then a curved arrow drops down at the right edge; Row 2 reads right to left (nodes 5,6,7,8), then a curved arrow drops down at the left edge; Row 3 reads left to right (nodes 9,10,11,12). Every consecutive node is joined by a thick solid arrow following this snake path, so the eye flows continuously from node 1 to node 12.

Each node shows a large number badge in a circle at its top-left corner, plus a short label and a tiny icon. Node labels, exactly:
1 "Global iteration index" — icon: counter digits
2 "Inject lr and wd" — icon: two sliders
3 "Crops to GPU" — icon: chip with upward arrow
4 "Teacher forward (2 global)" — icon: a professor-style head silhouette with a small padlock
5 "Student forward (all 10)" — icon: a student head silhouette with graduation cap
6 "DINO loss" — icon: two overlapping bar-histograms
7 "NaN guard" — icon: warning triangle
8 "Backward" — icon: backward-curving arrow
9 "clip_gradients (per tensor)" — icon: scissors over stacked bars
10 "cancel_last_layer grad" — icon: snowflake / freeze
11 "optimizer.step (AdamW)" — icon: footstep forward
12 "EMA teacher update" — icon: circular arrow

Color-group the nodes into four families, shown in a small legend strip under the title with four colored chips:
- Setup (nodes 1-3): soft blue
- Forward (nodes 4-6): warm amber
- Backward (nodes 7-10): coral red
- Update (nodes 11-12): fresh green
Node fills are pale tints of these colors with a saturated border and a matching number badge.

Special visual emphasis:
- Node 4 carries a small grey padlock icon with the tiny label "no grad" beside it, and a faint grey shading to signal frozen weights.
- Node 5 has small stacked rectangles at its side labeled "2 global + 8 local" to contrast with node 4.
- Between nodes 4/5 and node 6, thin converging lines meet at node 6.
- A prominent dashed curved arrow leaves node 12, sweeps across open space, and loops back into node 4, labeled along its path "EMA: teacher follows student" with a small "m = 0.996" tag.
- A thin vertical bracket spans nodes 8, 9, 10, 11 with the small caption "order matters: unscale, clip, cancel, step".

Bottom-right corner: a small side box titled "AMP branch" listing four tiny stacked chips: "scale(loss)", "unscale_", "scaler.step", "scaler.update", drawn in a muted grey-purple.
Bottom-left corner: a small side box titled "Shapes (B=4)" with three tiny rows: "images: 10 crops", "teacher: (8, K)", "student: (40, K)".

Style: clean flat vector illustration, educational infographic, generous white space, off-white background, rounded corners, subtle soft drop shadows, consistent 2px strokes, muted modern palette, crisp legible sans-serif typography, no photorealism, no clutter. All text must be rendered clearly and correctly. Aspect ratio 16:9.
