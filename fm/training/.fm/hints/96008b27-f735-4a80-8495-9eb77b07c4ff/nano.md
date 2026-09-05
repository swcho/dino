Create a clean flat-design educational infographic, 16:9 landscape, titled "DINO Training Pipeline in One Line" with a smaller subtitle "Six Beats, One Sentence".

Style: modern flat vector infographic, off-white background, thin rounded panel outlines, generous white space, crisp sans-serif labels, limited palette — deep indigo for student, warm amber for teacher, teal for data, coral for loss, grey for optimizer. No photographs, no 3D, no heavy shading. All text in English only — no Korean characters anywhere. Spell the title exactly "DINO Training Pipeline in One Line". Render only the labels listed below — no layout notes, no band names, no percentages, no placeholder text anywhere in the image. Every label short, sharp and correctly spelled.

Composition: three stacked horizontal zones, read top to bottom, with no row headings.

TOP ZONE, the dominant element — a left-to-right numbered flow of six boxes joined by bold arrows:
1. "ImageFolder", sub-label "labels discarded"; icon: a folder with a class tag crossed out.
2. "Multi-Crop Augment", sub-label "2 x 224 + 8 x 96"; icon: one photo splitting into two large squares and eight tiny squares, the two large ones visibly different (one blurred, one inverted).
3. A vertically split box titled "Two Networks": top half amber "Teacher: 2 global views" with tag "centering + sharpening + detach"; bottom half indigo "Student: all 10 views" with tag "gradients flow here".
4. Coral box "DINO Loss" with a large bold "18 CE terms" and the small line "2 x (2+8) - 2 = 18"; icon: a 2-row by 10-column grid of dots with two dots crossed out.
5. Grey box "Clip + AdamW", sub-label "per-tensor clip 3.0".
6. Amber box "EMA Teacher Update" with the formula "θt ← m θt + (1−m) θs" and tag "m: 0.996 → 1.0".

Below that flow, draw two curved dashed feedback arrows: a long amber dashed arrow from box 6 back to the Teacher half of box 3, labeled "EMA params, no gradient"; and a shorter dashed arrow from Teacher to box 4, labeled "center EMA, m = 0.9". These loops should be clearly visible.

MIDDLE ZONE — two panels side by side.
Left panel titled "Layers of Asymmetry": five thin colored rows, each with a tiny icon and a short label:
 "Augmentation: blur vs solarize"
 "View: 224 global vs 96 local"
 "Path: teacher 2, student 10"
 "Gradient: teacher frozen"
 "Temperature: 0.04 vs 0.1"
Right panel titled "Why It Never Collapses": a tug-of-war balance beam on a triangular pivot. Left end labeled "Uniform collapse" with a flat bar chart icon; right end labeled "Single prototype" with a one-spike bar icon. Two arrows point inward and meet above the pivot: the one from the left tagged "sharpening", the one from the right tagged "centering". Centered under the pivot: "balanced = healthy targets".

BOTTOM ZONE — a slim full-width strip titled "Four Schedules" holding four small cards side by side, each with a tiny sparkline curve and a two-line label:
 "lr" / "warmup, then cosine down"
 "weight decay" / "0.04 up to 0.4"
 "teacher momentum" / "0.996 up to 1.0"
 "teacher temp" / "0.04 up to 0.07"

Visual hierarchy: the six-box flow reads first and occupies roughly the upper half; the two comparison panels are lighter; the schedule strip is smallest. Keep every label under five words.
