Create a clean flat-design educational infographic, 16:9 landscape, white background, light blue / coral / warm gray accent palette, crisp sans-serif labels, generous whitespace, thin rounded arrows.

Title at top center, large bold: "DINO: Self-Distillation with No Labels"
Small subtitle under the title: "Same image, different crops, same distribution"

Layout: three horizontal panels left to right, connected by arrows, plus a small bottom strip.

PANEL A (left) — label above panel: "1. Multi-Crop"
Show one photo thumbnail of a dog at top. An arrow fans out downward into two large squares labeled "2 global 224px" and a row of four small squares labeled "8 local 96px". Caption under panel: "One image, many views".

PANEL B (center) — label above panel: "2. Match Distributions"
Two rounded boxes stacked vertically. Top box coral, labeled "TEACHER" with Korean "(교사)" beside it, fed only by the two global crops. Bottom box blue, labeled "STUDENT" with Korean "(학생)" beside it, fed by all crops. Each box outputs a small bar-chart histogram icon labeled "K-dim distribution". A thick double-headed arrow between the two histograms labeled "cross-entropy". A small tag on the teacher output arrow reads "stop-gradient". A curved dashed arrow loops from the STUDENT box back up to the TEACHER box, labeled "EMA update only" with a small formula chip beneath: "θt ← m·θt + (1−m)·θs".
Caption under panel: "Teacher sees global, student sees all".

PANEL C (right) — label above panel: "3. Avoid Collapse"
A horizontal balance beam / seesaw. Left pan labeled "CENTERING" with a small flat uniform histogram icon and tiny text "pushes to uniform". Right pan labeled "SHARPENING" with a small spiky one-hot histogram icon and tiny text "pushes to one-hot". The beam is level and balanced. Caption under panel: "Two opposing forces".

BOTTOM STRIP — a narrow band spanning the full width, split into three equal cells, each with a large red circle-slash icon and a short label:
"NO labels", "NO negative pairs", "NO contrastive loss"

Keep every label under five words. Do not add any text beyond the labels specified above.
