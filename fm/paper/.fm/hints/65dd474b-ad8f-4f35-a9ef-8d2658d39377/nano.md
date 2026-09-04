Create a clean, flat-design educational infographic, 16:9 landscape, on a light off-white background with a restrained palette: deep navy for text, muted orange for "k-NN", cool blue for "linear", light gray gridlines.

Title at the top center, large and bold: "k-NN almost equals Linear" with a smaller Korean subtitle underneath: "DINO ViT-S: 특징이 이미 뭉쳐 있다".

Layout: an upper row split into two equal side-by-side panels separated by a thin vertical divider, and a lower full-width horizontal band. Eye flow goes left panel to right panel, then down to the bar band.

UPPER LEFT PANEL, header label "Other SSL / ResNet-50":
A 2D scatter of feature points in two classes (blue circles and orange triangles). Draw them as two long, stretched, interleaved streaks so that many nearest neighbors belong to the other class. Overlay one bold dashed straight diagonal line labeled "Learned linear boundary" showing the classes ARE separable by a line. Add a small dotted circle around one query point, with 3 mixed-color points inside it, labeled "Neighbors are mixed". Bottom of panel, a small badge: "Gap 7-12%p".

UPPER RIGHT PANEL, header label "DINO + ViT":
The same two classes, but now drawn as two tight, round, clearly separated clusters. No decision boundary line needed. Add a small dotted circle around one query point with 3 same-color points inside it, labeled "Neighbors agree". Bottom of panel, a small badge: "Gap 2.5%p". Put a small check-mark icon next to this badge and a small warning icon next to the left panel badge.

LOWER FULL-WIDTH BAND, header label "Linear vs k-NN gap (ImageNet top-1)":
A horizontal grouped bar chart, one row per method, each row showing a blue "Linear" bar and an orange "k-NN" bar with the numeric gap printed at the right end of the row. Rows top to bottom:
"DINO ViT-S  77.0 / 74.5  → 2.5%p" (highlight this row with a soft orange background tint)
"BYOL ViT-S  71.4 / 66.6  → 4.8%p"
"SwAV ViT-S  73.5 / 66.3  → 7.2%p"
"MoCo-v2 ViT-S  72.7 / 64.4  → 8.3%p"
"DINO ResNet-50  75.3 / 67.5  → 7.8%p"
"SwAV ResNet-50  75.3 / 65.7  → 9.6%p"
Add a small legend with two swatches: "Linear (trained)" and "k-NN (no training)".

Bottom strip, one short caption line in small text: "Small gap = classes already locally clustered" with a Korean tag "국소 클러스터링".

Style: modern educational infographic, flat vector look, generous white space, thin 1px strokes, rounded panel corners, no photographic textures, no 3D effects, no drop shadows. All labels short (3-5 words max), crisp sans-serif type, high legibility. Aspect ratio 16:9.
